"""
FACMAC: Factored Multi-Agent Centralised Policy Gradients

Cooperative multi-agent actor-critic with:
- Decentralised deterministic actors (one per agent)
- Centralised but factored critic (per-agent utilities + QMIX-style mixer)
- Centralised policy gradient over the joint action space
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from src.agents.device import get_torch_device


def _mlp(
    input_size: int,
    hidden_layers: Sequence[int],
    output_size: int,
    activation: str = "relu",
    dropout: float = 0.0,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    act_cls = act_map.get(activation.lower(), nn.ReLU)
    layers: List[nn.Module] = []
    prev = input_size
    for hidden in hidden_layers:
        layers.append(nn.Linear(prev, hidden))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden
    layers.append(nn.Linear(prev, output_size))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Per-agent deterministic policy μ_a(o_a) -> continuous action in [-1, 1]."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.network = _mlp(
            obs_size,
            hidden_layers,
            action_size,
            activation=activation,
            dropout=dropout,
            final_activation=nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class AgentUtilityCritic(nn.Module):
    """Per-agent utility Q_a(o_a, u_a)."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.network = _mlp(
            obs_size + action_size,
            hidden_layers,
            1,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([obs, action], dim=-1))


class MixingNetwork(nn.Module):
    """
    QMIX-style mixer: combines per-agent utilities into Q_tot using a
    state-conditioned hypernetwork. Non-negative weights enforce monotonicity
    when monotonic=True (default FACMAC).
    """

    def __init__(
        self,
        n_agents: int,
        state_size: int,
        mixing_embed_dim: int = 32,
        hypernet_hidden: int = 64,
        monotonic: bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.monotonic = monotonic
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, n_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_size, mixing_embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (batch, n_agents)
        state:    (batch, state_size)
        returns:  (batch, 1)
        """
        batch_size = agent_qs.size(0)
        qs = agent_qs.view(batch_size, 1, self.n_agents)

        w1 = self.hyper_w1(state).view(batch_size, self.n_agents, self.mixing_embed_dim)
        if self.monotonic:
            w1 = torch.abs(w1)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(qs, w1) + b1)

        w2 = self.hyper_w2(state).view(batch_size, self.mixing_embed_dim, 1)
        if self.monotonic:
            w2 = torch.abs(w2)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, 1)


class FACMACReplayBuffer:
    """Stores joint multi-agent transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        global_state,
        local_obs,
        actions,
        reward,
        next_global_state,
        next_local_obs,
        done,
    ):
        self.buffer.append(
            (
                np.asarray(global_state, dtype=np.float32),
                np.asarray(local_obs, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                float(reward),
                np.asarray(next_global_state, dtype=np.float32),
                np.asarray(next_local_obs, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        (
            global_state,
            local_obs,
            actions,
            reward,
            next_global_state,
            next_local_obs,
            done,
        ) = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(global_state, dtype=torch.float32, device=device),
            torch.as_tensor(local_obs, dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.float32, device=device),
            torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(next_global_state, dtype=torch.float32, device=device),
            torch.as_tensor(next_local_obs, dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def to_numpy_dict(self):
        """Serialize buffer contents for checkpointing."""
        if not self.buffer:
            return None
        (
            global_state,
            local_obs,
            actions,
            reward,
            next_global_state,
            next_local_obs,
            done,
        ) = zip(*self.buffer)
        return {
            "global_state": np.stack(global_state).astype(np.float32),
            "local_obs": np.stack(local_obs).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_global_state": np.stack(next_global_state).astype(np.float32),
            "next_local_obs": np.stack(next_local_obs).astype(np.float32),
            "done": np.asarray(done, dtype=np.float32),
        }

    def load_numpy_dict(self, data):
        """Restore buffer contents from a checkpoint payload."""
        self.buffer.clear()
        if not data:
            return
        n = len(data["global_state"])
        for i in range(n):
            self.push(
                data["global_state"][i],
                data["local_obs"][i],
                data["actions"][i],
                float(data["reward"][i]),
                data["next_global_state"][i],
                data["next_local_obs"][i],
                float(data["done"][i]),
            )


class FACMACAgent:
    """
    Factored Multi-Agent Centralised Policy Gradients agent.

    Agents share a team reward. Actors condition on local observations;
    the factored critic mixes per-agent utilities with a global state.
    """

    def __init__(
        self,
        n_agents: int,
        obs_size: int,
        action_size: int,
        state_size: int,
        config_path: str,
        agent_names: Optional[Sequence[str]] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.n_agents = n_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.state_size = state_size
        self.agent_names = list(agent_names) if agent_names else [f"agent_{i}" for i in range(n_agents)]

        device_config = self.config.get("training", {}).get("device", "auto")
        self.device = get_torch_device(device_config)

        facmac_cfg = self.config.get("facmac", {})
        self.learning_rate = facmac_cfg.get("learning_rate", 3e-4)
        self.gamma = facmac_cfg.get("gamma", 0.99)
        self.tau = facmac_cfg.get("tau", 0.005)
        self.batch_size = facmac_cfg.get("batch_size", 256)
        self.target_update_interval = facmac_cfg.get("target_update_interval", 1)
        self.exploration_noise = facmac_cfg.get("exploration_noise", 0.1)
        self.exploration_noise_clip = facmac_cfg.get("exploration_noise_clip", 0.5)
        self.grad_clip = facmac_cfg.get("grad_clip", 10.0)
        monotonic = facmac_cfg.get("monotonic_mixing", True)
        mixing_embed_dim = facmac_cfg.get("mixing_embed_dim", 32)
        hypernet_hidden = facmac_cfg.get("hypernet_hidden", 64)

        model_cfg = self.config.get("model", {})
        hidden_layers = model_cfg.get("hidden_layers", [256, 256])
        dropout = model_cfg.get("dropout", 0.0)
        activation = model_cfg.get("activation", "relu")

        self.actors = nn.ModuleList(
            [
                DeterministicActor(obs_size, action_size, hidden_layers, dropout, activation)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.target_actors = nn.ModuleList(
            [
                DeterministicActor(obs_size, action_size, hidden_layers, dropout, activation)
                for _ in range(n_agents)
            ]
        ).to(self.device)

        self.critics = nn.ModuleList(
            [
                AgentUtilityCritic(obs_size, action_size, hidden_layers, dropout, activation)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.target_critics = nn.ModuleList(
            [
                AgentUtilityCritic(obs_size, action_size, hidden_layers, dropout, activation)
                for _ in range(n_agents)
            ]
        ).to(self.device)

        self.mixer = MixingNetwork(
            n_agents, state_size, mixing_embed_dim, hypernet_hidden, monotonic=monotonic
        ).to(self.device)
        self.target_mixer = MixingNetwork(
            n_agents, state_size, mixing_embed_dim, hypernet_hidden, monotonic=monotonic
        ).to(self.device)

        self._hard_update_all()

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critics.parameters()) + list(self.mixer.parameters()),
            lr=self.learning_rate,
        )

        memory_size = facmac_cfg.get("memory_size", 1_000_000)
        self.memory = FACMACReplayBuffer(memory_size)
        self.training_step = 0

    def _hard_update_all(self):
        for target, source in zip(self.target_actors, self.actors):
            target.load_state_dict(source.state_dict())
        for target, source in zip(self.target_critics, self.critics):
            target.load_state_dict(source.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def select_actions(
        self,
        local_obs: np.ndarray,
        evaluate: bool = False,
    ) -> np.ndarray:
        """
        local_obs: (n_agents, obs_size)
        returns:   (n_agents, action_size) in [-1, 1]
        """
        obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                action = actor(obs_t[i].unsqueeze(0)).squeeze(0)
                if not evaluate and self.exploration_noise > 0:
                    noise = torch.randn_like(action) * self.exploration_noise
                    noise = torch.clamp(
                        noise, -self.exploration_noise_clip, self.exploration_noise_clip
                    )
                    action = torch.clamp(action + noise, -1.0, 1.0)
                actions.append(action.cpu().numpy())
        return np.stack(actions, axis=0).astype(np.float32)

    def store_transition(
        self,
        global_state,
        local_obs,
        actions,
        reward,
        next_global_state,
        next_local_obs,
        done,
    ):
        self.memory.push(
            global_state, local_obs, actions, reward, next_global_state, next_local_obs, done
        )

    def _agent_qs(
        self,
        critics: nn.ModuleList,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return stacked utilities (batch, n_agents)."""
        qs = []
        for i, critic in enumerate(critics):
            qs.append(critic(local_obs[:, i, :], actions[:, i, :]))
        return torch.cat(qs, dim=1)

    def _policy_actions(self, actors: nn.ModuleList, local_obs: torch.Tensor) -> torch.Tensor:
        """Return joint actions from current policies (batch, n_agents, action_size)."""
        actions = []
        for i, actor in enumerate(actors):
            actions.append(actor(local_obs[:, i, :]))
        return torch.stack(actions, dim=1)

    def update_parameters(self) -> Optional[Dict[str, float]]:
        if len(self.memory) < self.batch_size:
            return None

        (
            global_state,
            local_obs,
            actions,
            reward,
            next_global_state,
            next_local_obs,
            done,
        ) = self.memory.sample(self.batch_size, self.device)

        # ---- Critic update (TD on factored Q_tot) ----
        with torch.no_grad():
            next_actions = self._policy_actions(self.target_actors, next_local_obs)
            next_agent_qs = self._agent_qs(self.target_critics, next_local_obs, next_actions)
            next_q_tot = self.target_mixer(next_agent_qs, next_global_state)
            target_q = reward + (1.0 - done) * self.gamma * next_q_tot

        current_agent_qs = self._agent_qs(self.critics, local_obs, actions)
        current_q_tot = self.mixer(current_agent_qs, global_state)
        critic_loss = F.mse_loss(current_q_tot, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                list(self.critics.parameters()) + list(self.mixer.parameters()),
                self.grad_clip,
            )
        self.critic_optimizer.step()

        # ---- Centralised actor update (joint action space, current policies) ----
        # Sample ALL agents' actions from current policies (FACMAC), not replay actions.
        policy_actions = self._policy_actions(self.actors, local_obs)
        policy_agent_qs = self._agent_qs(self.critics, local_obs, policy_actions)
        q_tot = self.mixer(policy_agent_qs, global_state)
        actor_loss = -q_tot.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.actors.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        if self.training_step % self.target_update_interval == 0:
            for target, source in zip(self.target_actors, self.actors):
                self.soft_update(target, source, self.tau)
            for target, source in zip(self.target_critics, self.critics):
                self.soft_update(target, source, self.tau)
            self.soft_update(self.target_mixer, self.mixer, self.tau)

        self.training_step += 1
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_tot": float(current_q_tot.mean().item()),
        }

    def save(self, filepath: str, extra_state: Optional[dict] = None):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        replay_payload = self.memory.to_numpy_dict()
        torch.save(
            {
                "actors": self.actors.state_dict(),
                "critics": self.critics.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_actors": self.target_actors.state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "n_agents": self.n_agents,
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "state_size": self.state_size,
                "agent_names": self.agent_names,
                "replay_buffer": replay_payload,
                "extra_state": extra_state or {},
            },
            filepath,
        )
        n_replay = 0 if replay_payload is None else len(replay_payload["global_state"])
        print(f"FACMAC model saved to {filepath} (replay_buffer={n_replay} transitions)")

    def load(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            print(f"No FACMAC model found at {filepath}")
            return {}
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actors.load_state_dict(checkpoint["actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint.get("training_step", 0)

        replay_payload = checkpoint.get("replay_buffer")
        if replay_payload is not None:
            self.memory.load_numpy_dict(replay_payload)
            print(
                f"FACMAC model loaded from {filepath} "
                f"(restored replay_buffer={len(self.memory)} transitions)"
            )
        else:
            print(
                f"FACMAC model loaded from {filepath} "
                "(no replay_buffer in checkpoint — warmup refill required)"
            )
        return checkpoint.get("extra_state", {}) or {}
