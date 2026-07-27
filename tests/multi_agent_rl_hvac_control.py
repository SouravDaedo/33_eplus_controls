"""
Multi-Agent RL HVAC Control with FACMAC

Three floor agents (bottom / mid / top) cooperatively control:
- Per-floor heating and cooling setpoints (sp_offset + deadband)
- Per-floor AHU outdoor-air mass flow

Uses FACMAC (Factored Multi-Agent Centralised Policy Gradients):
- Decentralised actors with local floor observations
- Centralised factored critic (per-agent utilities + QMIX-style mixer)
- Centralised policy gradient over the joint action space

Objectives and EnergyPlus simulation setup match tests/rl_hvac_control.py.

Usage:
    python tests/multi_agent_rl_hvac_control.py --training --save-model
    python tests/multi_agent_rl_hvac_control.py --model outputs/.../facmac_hvac_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from pyenergyplus.api import EnergyPlusAPI
except ImportError:
    print("Warning: EnergyPlus API not found. Install with: pip install pyenergyplus-lbnl")
    sys.exit(1)

from src.agents.facmac_agent import FACMACAgent
from src.utils.hvac_config import get_hvac_config
from src.utils.idf_modifier import (
    calculate_simulation_days,
    create_custom_idf,
    get_season_info,
)

from tests.rl_hvac_control import (
    AHU_OA_CONTROLS,
    CONTROLLED_ZONE_NAMES,
    FLOOR_ORDER,
    FLOOR_ZONE_GROUPS,
    HVACEnvironment,
    LiveEpisodeDollarKPIPlotter,
    LiveLossPlotter,
    LiveRLPlotter,
    _in_training_window,
    _sample_random_start_in_window,
)


AGENT_FLOORS = list(FLOOR_ORDER)  # bottom, mid, top
ACTIONS_PER_AGENT = 3  # sp_offset, deadband, oa_mass_flow


class LiveFACMACPlotter(LiveRLPlotter):
    """Live plot that uses per-floor setpoints and commanded OA from FACMAC logs."""

    def __init__(self, output_dir, update_every=1, episode_scope="current"):
        super().__init__(output_dir, update_every=update_every, episode_scope=episode_scope)
        self.fig.suptitle("FACMAC Multi-Agent HVAC Live Plot")

    def update(self, log_entry):
        # Parent expects a single commanded_oa_mass_flow; use mean for compatibility,
        # then overwrite per-floor commanded series after the parent append.
        patched = dict(log_entry)
        if "commanded_oa_mass_flow" not in patched:
            cmds = [
                log_entry.get(f"{floor}_commanded_oa_mass_flow")
                for floor in ("bottom", "mid", "top")
                if log_entry.get(f"{floor}_commanded_oa_mass_flow") is not None
            ]
            if cmds:
                patched["commanded_oa_mass_flow"] = float(np.mean(cmds))
        super().update(patched)

        for floor, series in (
            ("bottom", "bottom_floor_temp"),
            ("mid", "mid_floor_temp"),
            ("top", "top_floor_temp"),
        ):
            htg = log_entry.get(f"{floor}_heating_setpoint")
            clg = log_entry.get(f"{floor}_cooling_setpoint")
            if htg is not None and self.series[f"{series}_heating_setpoint"]:
                self.series[f"{series}_heating_setpoint"][-1] = float(htg)
            if clg is not None and self.series[f"{series}_cooling_setpoint"]:
                self.series[f"{series}_cooling_setpoint"][-1] = float(clg)

        for floor, series in (
            ("bottom", "bottom_floor_airflow"),
            ("mid", "mid_floor_airflow"),
            ("top", "top_floor_airflow"),
        ):
            cmd = log_entry.get(f"{floor}_commanded_oa_mass_flow")
            if cmd is not None and self.series.get(f"{series}_commanded"):
                self.series[f"{series}_commanded"][-1] = float(cmd)

        if self.step_count % self.update_every == 0:
            x = self.series["step"]
            for floor, series in (
                ("bottom", "bottom_floor_temp"),
                ("mid", "mid_floor_temp"),
                ("top", "top_floor_temp"),
            ):
                self.lines[f"{series}_heating_setpoint"].set_data(
                    x, self.series[f"{series}_heating_setpoint"]
                )
                self.lines[f"{series}_cooling_setpoint"].set_data(
                    x, self.series[f"{series}_cooling_setpoint"]
                )
            for series in ("bottom_floor_airflow", "mid_floor_airflow", "top_floor_airflow"):
                cmd_key = f"{series}_commanded"
                if cmd_key in self.lines:
                    self.lines[cmd_key].set_data(x, self.series[cmd_key])

    def finish(self, output_dir, hold=False):
        path = Path(output_dir) / "facmac_hvac_live_plot.png"
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved live plot snapshot to {path}")
        if hold:
            print("Close the plot window to finish.")
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class LiveFACMACLossPlotter(LiveLossPlotter):
    """FACMAC losses (+ optional adaptive weights) using LiveLossPlotter infrastructure."""

    def __init__(self, output_dir, update_every=1, show_losses=True, show_adaptive_weights=False):
        self._use_facmac_loss_layout = True
        super().__init__(
            output_dir,
            update_every=update_every,
            show_losses=show_losses,
            show_adaptive_weights=show_adaptive_weights,
        )
        # Extra FACMAC series (parent already has actor_loss / critic1 / etc.)
        self.series.setdefault("critic_loss", [])
        self.series.setdefault("q_tot", [])

    def _setup_figure(self):
        if not getattr(self, "_use_facmac_loss_layout", False):
            return super()._setup_figure()

        title_bits = []
        if self.show_losses:
            title_bits.append("FACMAC Losses")
        if self.show_adaptive_weights:
            title_bits.append("Adaptive Weights")
        title = " / ".join(title_bits) or "FACMAC"

        if self.show_losses and self.show_adaptive_weights:
            self.fig = self.plt.figure(figsize=(12, 10), num="FACMAC Losses + Adaptive Weights")
            gs = self.fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.35, wspace=0.25)
            self._setup_loss_axis(self.fig.add_subplot(gs[0, 0]), "actor_loss", "Actor Loss", "tab:purple")
            self._setup_loss_axis(self.fig.add_subplot(gs[0, 1]), "critic_loss", "Critic Loss", "tab:red")
            self._setup_loss_axis(self.fig.add_subplot(gs[1, :]), "q_tot", "Batch Mean Q_tot", "tab:orange")
            self.ax_scale = self.fig.add_subplot(gs[2, 0])
            self.ax_ema = self.fig.add_subplot(gs[2, 1])
            self._setup_weight_axes()
        elif self.show_losses:
            self.fig, axes = self.plt.subplots(1, 3, figsize=(12, 4.5), num="FACMAC Training Losses")
            self._setup_loss_axis(axes[0], "actor_loss", "Actor Loss", "tab:purple")
            self._setup_loss_axis(axes[1], "critic_loss", "Critic Loss", "tab:red")
            self._setup_loss_axis(axes[2], "q_tot", "Batch Mean Q_tot", "tab:orange")
            self.fig.tight_layout(pad=0.8, w_pad=0.5, h_pad=0.7)
        else:
            self.fig, (self.ax_scale, self.ax_ema) = self.plt.subplots(
                1, 2, figsize=(12, 4.5), num="Adaptive Reward Weights"
            )
            self._setup_weight_axes()
            self.fig.tight_layout(pad=0.9, w_pad=0.5)

        self.fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update(self, update_no, losses):
        if not self.show_losses:
            return
        self.step_count += 1
        self.series["update"].append(int(update_no))
        self.series["actor_loss"].append(float(losses["actor_loss"]))
        self.series["critic_loss"].append(float(losses["critic_loss"]))
        self.series["q_tot"].append(float(losses.get("q_tot", np.nan)))

        if self.step_count % self.update_every != 0:
            return

        x = self.series["update"]
        for name in ("actor_loss", "critic_loss", "q_tot"):
            if name in self.lines:
                self.lines[name].set_data(x, self.series[name])
        for ax in self.loss_axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def finish(self, output_dir, hold=False):
        if self.show_losses and self.show_adaptive_weights:
            name = "facmac_losses_and_adaptive_weights.png"
        elif self.show_adaptive_weights:
            name = "facmac_adaptive_weights.png"
        else:
            name = "facmac_training_losses.png"
        path = Path(output_dir) / name
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved FACMAC loss/weight plot snapshot to {path}")
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class MultiAgentHVACEnvironment(HVACEnvironment):
    """
    Floor-factored multi-agent wrapper around HVACEnvironment.

    Each agent controls one floor's thermostat band and AHU OA flow.
    Team reward matches the single-agent objective (energy, comfort, demand, etc.).
    """

    def __init__(self, api, state, config_path=None, simulation_overrides=None):
        super().__init__(api, state, config_path, simulation_overrides=simulation_overrides)
        self.n_agents = len(AGENT_FLOORS)
        self.agent_action_size = ACTIONS_PER_AGENT
        self.current_joint_action = None  # (n_agents, action_size)
        self._prev_joint_action = None
        self._obs_layout = self._build_obs_layout()
        self.local_obs_size = self._obs_layout["local_obs_size"]

    def _build_obs_layout(self):
        ss = self.hvac_config.config["state_space"]
        zone_count = ss["zone_temps"]["count"]
        # State zone order matches CONTROLLED_ZONE_NAMES[:zone_count] (not floor-contiguous).
        zone_names = CONTROLLED_ZONE_NAMES[:zone_count]
        zone_index = {name: i for i, name in enumerate(zone_names)}
        floor_zone_indices = {
            floor: [zone_index[z] for z in FLOOR_ZONE_GROUPS[floor] if z in zone_index]
            for floor in AGENT_FLOORS
        }
        zones_per_floor = max(len(idxs) for idxs in floor_zone_indices.values()) if floor_zone_indices else 0

        weather_count = len(ss["current_weather"])
        forecast_count = sum(
            len(self.weather_forecast_offsets[field])
            for field in ("oat", "humidity", "cloud_cover")
        )
        past_count = (
            self.weather_history_horizon * 3 if self.weather_history_enabled else 0
        )
        time_count = len(ss["time_features"])
        prev_count = ss.get("previous_actions", {}).get("count", 0)
        outdoor_co2_count = ss.get("outdoor_co2_ppm", {}).get("count", 0)
        floor_co2_count = ss.get("floor_co2_ppm", {}).get("count", 0)
        rtp_count = ss.get("rtp_price", {}).get("count", 0)

        idx = 0
        layout = {
            "zone_temps": (idx, idx + zone_count),
            "zone_names": zone_names,
            "floor_zone_indices": floor_zone_indices,
            "zones_per_floor": zones_per_floor,
        }
        idx += zone_count
        layout["weather"] = (idx, idx + weather_count)
        idx += weather_count
        layout["forecast"] = (idx, idx + forecast_count)
        idx += forecast_count
        layout["past"] = (idx, idx + past_count)
        idx += past_count
        layout["time"] = (idx, idx + time_count)
        idx += time_count
        layout["prev_action"] = (idx, idx + prev_count)
        idx += prev_count
        layout["outdoor_co2"] = (idx, idx + outdoor_co2_count)
        idx += outdoor_co2_count
        layout["floor_co2"] = (idx, idx + floor_co2_count)
        idx += floor_co2_count
        layout["rtp_price"] = (idx, idx + rtp_count)

        # Per-agent previous actions live in local obs (not the global 3-D prev slot).
        local_prev_count = self.agent_action_size if prev_count > 0 else 0
        shared_len = (
            weather_count
            + forecast_count
            + past_count
            + time_count
            + outdoor_co2_count
            + rtp_count
        )
        # local = floor zone temps + shared features + this floor CO2 + optional prev action
        layout["local_prev_count"] = local_prev_count
        layout["local_obs_size"] = zones_per_floor + shared_len + 1 + local_prev_count
        layout["zone_count"] = zone_count
        return layout

    def get_global_state(self):
        """Centralised critic state (same vector as single-agent SAC)."""
        return self.get_current_state()

    def get_local_observations(self, global_state=None):
        """
        Build per-agent observations from the global state vector.

        Each floor agent sees its own zone temperatures (looked up by zone name,
        since the global state packs zones in CONTROLLED_ZONE_NAMES order), plus
        shared weather/time/CO2 features and that floor's average CO2.

        Returns array of shape (n_agents, local_obs_size).
        """
        if global_state is None:
            global_state = self.get_global_state()
        state = np.asarray(global_state, dtype=np.float32)
        layout = self._obs_layout
        zones_per_floor = layout["zones_per_floor"]

        z0, z1 = layout["zone_temps"]
        zone_temps = state[z0:z1]
        shared_parts = []
        for key in ("weather", "forecast", "past", "time", "outdoor_co2", "rtp_price"):
            a, b = layout[key]
            if b > a:
                shared_parts.append(state[a:b])
        shared = np.concatenate(shared_parts) if shared_parts else np.zeros(0, dtype=np.float32)

        f0, f1 = layout["floor_co2"]
        floor_co2 = state[f0:f1] if f1 > f0 else np.zeros(self.n_agents, dtype=np.float32)
        local_prev_count = layout.get("local_prev_count", 0)

        local_obs = []
        for i, floor in enumerate(AGENT_FLOORS):
            idxs = layout["floor_zone_indices"].get(floor, [])
            floor_temps = zone_temps[idxs] if idxs else np.zeros(0, dtype=np.float32)
            if len(floor_temps) < zones_per_floor:
                floor_temps = np.pad(floor_temps, (0, zones_per_floor - len(floor_temps)))
            this_floor_co2 = (
                np.array([floor_co2[i]], dtype=np.float32)
                if i < len(floor_co2)
                else np.array([0.0], dtype=np.float32)
            )
            parts = [floor_temps, shared, this_floor_co2]
            if local_prev_count > 0:
                if self._prev_joint_action is not None:
                    prev = np.asarray(self._prev_joint_action[i], dtype=np.float32).reshape(-1)
                else:
                    prev_cfg = self.hvac_config.config["state_space"].get("previous_actions", {})
                    init = prev_cfg.get("initial_value") or [0.0] * local_prev_count
                    prev = np.asarray(init[:local_prev_count], dtype=np.float32)
                if len(prev) < local_prev_count:
                    prev = np.pad(prev, (0, local_prev_count - len(prev)))
                parts.append(prev[:local_prev_count])
            obs = np.concatenate(parts).astype(np.float32)
            if len(obs) < self.local_obs_size:
                obs = np.pad(obs, (0, self.local_obs_size - len(obs)))
            local_obs.append(obs[: self.local_obs_size])
        return np.stack(local_obs, axis=0)

    def _scale_action_dim(self, a, low, high):
        return low + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (high - low)

    def decode_agent_action(self, action):
        """Map one agent tanh action to physical setpoints / OA flow."""
        bounds = self.hvac_config.get_action_bounds()
        sp_offset = self._scale_action_dim(
            action[0], bounds["sp_offset"][0], bounds["sp_offset"][1]
        )
        deadband = self._scale_action_dim(
            action[1], bounds["deadband"][0], bounds["deadband"][1]
        )
        oa_mass_flow = max(
            0.0,
            self._scale_action_dim(
                action[2],
                bounds["airflow_multiplier"][0],
                bounds["airflow_multiplier"][1],
            ),
        )
        heating_sp = self.base_temp + sp_offset - deadband / 2
        cooling_sp = self.base_temp + sp_offset + deadband / 2
        return {
            "sp_offset": sp_offset,
            "deadband": deadband,
            "oa_mass_flow": oa_mass_flow,
            "heating_sp": heating_sp,
            "cooling_sp": cooling_sp,
        }

    def apply_joint_action(self, joint_action):
        """
        Apply per-floor actions to EnergyPlus.

        joint_action: (n_agents, 3) in [-1, 1]
        """
        if not self.handles_initialized:
            return

        exchange = self.api.exchange
        joint_action = np.asarray(joint_action, dtype=np.float32)
        if joint_action.shape != (self.n_agents, self.agent_action_size):
            raise ValueError(
                f"Expected joint action shape {(self.n_agents, self.agent_action_size)}, "
                f"got {joint_action.shape}"
            )

        decoded = []
        for i, floor in enumerate(AGENT_FLOORS):
            ctrl = self.decode_agent_action(joint_action[i])
            decoded.append(ctrl)
            for zone_name in FLOOR_ZONE_GROUPS[floor]:
                handles = self.handles["zones"].get(zone_name)
                if not handles:
                    continue
                if handles["heating_sp"] > 0:
                    exchange.set_actuator_value(
                        self.state, handles["heating_sp"], ctrl["heating_sp"]
                    )
                if handles["cooling_sp"] > 0:
                    exchange.set_actuator_value(
                        self.state, handles["cooling_sp"], ctrl["cooling_sp"]
                    )
            oa_handles = self.handles.get("ahu_oa", {}).get(floor)
            if oa_handles and oa_handles["commanded_mass_flow"] > 0:
                exchange.set_actuator_value(
                    self.state, oa_handles["commanded_mass_flow"], ctrl["oa_mass_flow"]
                )

        self.current_joint_action = joint_action
        self._prev_joint_action = np.asarray(joint_action, dtype=np.float32).copy()
        # Keep parent current_action as mean 3-D action so global prev_action (if enabled)
        # stays SAC-compatible; per-agent prev lives in local observations.
        self.current_action = joint_action.mean(axis=0)
        return decoded

    def compute_reward_components(self, reward_config, zone_temps, action):
        """
        Same team reward as single-agent, with setpoint penalty averaged over floors.

        `action` may be a single 3-vector (legacy) or joint (n_agents, 3).
        """
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 1:
            return super().compute_reward_components(reward_config, zone_temps, action_arr)

        # Use mean physical action for energy/demand path compatibility, then
        # replace setpoint penalty with multi-agent average.
        mean_action = action_arr.mean(axis=0)
        components = super().compute_reward_components(reward_config, zone_temps, mean_action)

        SETPOINT_WEIGHT = reward_config["setpoint_weight"]
        setpoint_penalty = 0.0
        for agent_action in action_arr:
            ctrl = self.decode_agent_action(agent_action)
            if abs(ctrl["sp_offset"]) > 2.0:
                setpoint_penalty += 0.1
            if ctrl["deadband"] < 1.0:
                setpoint_penalty += 0.05
        setpoint_penalty = (setpoint_penalty / max(1, len(action_arr))) * SETPOINT_WEIGHT
        components["setpoint_penalty"] = setpoint_penalty

        # Rebuild total including CO2. Dollar terms are already normalized by the parent;
        # setpoint_penalty is intentionally left unnormalized (same as SAC).
        co2_penalty = float(components.get("co2_penalty", 0.0))
        components["total_cost"] = (
            float(components["energy_cost"])
            + float(components["gas_cost"])
            + float(components["comfort_penalty"])
            + float(setpoint_penalty)
            + float(components["demand_penalty"])
            + co2_penalty
        )
        components["reward"] = -components["total_cost"]
        return components

    def step_multi(self, joint_action):
        """Execute one multi-agent environment step. Returns team reward."""
        prev_state = self.current_state.copy() if self.current_state is not None else None
        self.apply_joint_action(joint_action)
        self.current_state = self.get_current_state()
        local_obs = self.get_local_observations(self.current_state)

        if prev_state is not None:
            reward_config = self.hvac_config.config["reward"]
            if self.handles_initialized and self.handles.get("zones"):
                zone_temps = [
                    self._zone_temp_value(zone_name)
                    for zone_name in self.handles["zones"]
                ]
            else:
                zone_count = self.hvac_config.config["state_space"]["zone_temps"]["count"]
                zone_temps = list(self.current_state[:zone_count])
            components = self.compute_reward_components(
                reward_config, zone_temps, joint_action
            )
            reward = components["reward"]
            self._last_reward_components = components
        else:
            reward = 0.0
            self._last_reward_components = None

        self.episode_reward += reward
        self.timestep_count += 1
        steps_this_episode = (
            self._episode_duration_timesteps
            if self._episode_duration_timesteps is not None
            else self.episode_timesteps
        )
        current_day = self.api.exchange.day_of_month(self.state)
        done = (
            self.timestep_count >= steps_this_episode
            and current_day > 0
            and not self.api.exchange.warmup_flag(self.state)
        )
        return self.current_state, local_obs, reward, done, {}

    def reset(self):
        global_state = super().reset()
        self.current_joint_action = None
        self._prev_joint_action = None
        local_obs = self.get_local_observations(global_state)
        return global_state, local_obs


class FACMACHVACController:
    """Controller that uses FACMAC for multi-floor HVAC control."""

    def __init__(
        self,
        api,
        state,
        config_path=None,
        training_mode=False,
        live_plotter=None,
        loss_plotter=None,
        episode_kpi_plotter=None,
        output_dir=None,
        model_path=None,
        save_model=False,
        save_every=20,
        simulation_overrides=None,
        facmac_config_path=None,
    ):
        self.api = api
        self.state = state
        self.training_mode = training_mode
        self.live_plotter = live_plotter
        self.loss_plotter = loss_plotter
        self.episode_kpi_plotter = episode_kpi_plotter
        self.output_dir = output_dir
        self.save_model_enabled = save_model
        self.save_every = save_every

        self.env = MultiAgentHVACEnvironment(
            api, state, config_path, simulation_overrides=simulation_overrides
        )

        facmac_cfg = facmac_config_path or (project_root / "sac_config" / "facmac_config.yaml")
        self.agent = FACMACAgent(
            n_agents=self.env.n_agents,
            obs_size=self.env.local_obs_size,
            action_size=self.env.agent_action_size,
            state_size=self.env.state_size,
            config_path=str(facmac_cfg),
            agent_names=AGENT_FLOORS,
        )
        warmup_steps = self.agent.config.get("training", {}).get("warmup_steps", 0)
        self.training_start_memory = max(int(warmup_steps), int(self.agent.batch_size))
        if training_mode:
            print(
                "FACMAC training updates start when replay memory reaches "
                f"{self.training_start_memory} transitions "
                f"(batch_size={self.agent.batch_size}, warmup_steps={warmup_steps})."
            )

        if model_path:
            load_path = Path(model_path)
        elif not training_mode:
            load_path = project_root / "models" / "facmac_hvac_model.pth"
        else:
            load_path = None

        loaded_episode_count = 0
        if load_path is not None:
            if load_path.exists():
                extra_state = self.agent.load(str(load_path))
                loaded_episode_count = extra_state.get("episode_count", 0)
                print(
                    f"Loaded FACMAC model from {load_path}"
                    + (
                        f" (resuming at episode {loaded_episode_count})"
                        if loaded_episode_count
                        else ""
                    )
                )
                bal = getattr(self.env, "adaptive_cost_balancer", None)
                if bal is not None and bal.load_state_dict(extra_state.get("adaptive_balancer")):
                    print(
                        "Adaptive balancer restored: "
                        f"n_samples={bal.n_samples}  "
                        f"ppd_scale={bal.comfort_scale:.4f}  "
                        f"co2_scale={bal.co2_scale:.4f}  "
                        f"active={bal.n_samples >= bal.min_samples}"
                    )
                elif training_mode and bal is not None and bal.enabled:
                    print(
                        "No adaptive balancer state in checkpoint — "
                        "scales restart at initial_scale until min_samples."
                    )
                if training_mode:
                    mem = len(self.agent.memory)
                    if mem >= self.training_start_memory:
                        print(
                            f"Replay buffer restored ({mem} transitions) — "
                            "training updates will start on the first new step."
                        )
                    else:
                        print(
                            f"Replay buffer has {mem}/{self.training_start_memory} transitions — "
                            "collecting more before training updates resume."
                        )
            else:
                print(f"No pre-trained FACMAC model at {load_path}, using untrained agent")

        self.episode_count = loaded_episode_count
        self.max_episodes = 99999
        self.max_episodes_reached = False
        self.training_window = self.env.hvac_config.get_training_window()
        self.episode_duration_range = self.env.hvac_config.get_episode_duration_range()
        self._episode_started_this_run = False
        self._current_episode_duration_hours = None
        self._next_episode_start = None
        self._next_episode_duration_hours = None
        self.training_update_count = 0
        self.log_data = []
        self._co2_400_warned = False
        self.fatal_error = None
        self._fatal_error_reported = False

    def _handle_callback_error(self, exc):
        self.fatal_error = exc
        if not self._fatal_error_reported:
            self._fatal_error_reported = True
            print(f"\nFATAL FACMAC callback error: {exc}\n")
        self.max_episodes_reached = True
        self.api.runtime.stop_simulation(self.state)

    def power_cache_callback(self, state):
        if self.api.exchange.warmup_flag(state):
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if not self.env.handles_initialized:
            return
        if self.env.handles.get("total_power", -1) > 0:
            self.env._cached_power_w = self.api.exchange.get_variable_value(
                self.state, self.env.handles["total_power"]
            )

    def pre_timestep_callback(self, state):
        if self.api.exchange.warmup_flag(state):
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if not self.env.handles_initialized:
            return
        self.env.apply_outdoor_co2_for_current_timestep()
        self.env.apply_weather_override()
        self.env.apply_occupancy_events_for_current_timestep()

    def timestep_callback(self, state):
        try:
            self._timestep_callback_impl(state)
        except Exception as exc:
            self._handle_callback_error(exc)

    def _timestep_callback_impl(self, state):
        if self.api.exchange.warmup_flag(state):
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if self.max_episodes_reached:
            return

        current_month = self.api.exchange.month(self.state)
        current_day = self.api.exchange.day_of_month(self.state)
        current_hour = self.api.exchange.hour(self.state)
        if not _in_training_window(
            current_month, current_day, current_hour, self.training_window
        ):
            return

        if not self.env.handles_initialized:
            if not self.env.initialize_handles():
                return

        if self._next_episode_start is None:
            self._next_episode_start = _sample_random_start_in_window(self.training_window)
            min_h, max_h = self.episode_duration_range
            self._next_episode_duration_hours = (
                min_h if min_h == max_h else np.random.uniform(min_h, max_h)
            )

        if (current_month, current_day, current_hour) < self._next_episode_start:
            return

        if not self._episode_started_this_run:
            self.env.timestep_count = 0
            self.env.episode_reward = 0.0
            duration_hours = self._next_episode_duration_hours
            duration_timesteps = int(duration_hours * self.env.timesteps_per_hour)
            self.env.start_episode(duration_timesteps)
            self._episode_started_this_run = True
            self._current_episode_duration_hours = duration_hours
            nm, nd, nh = self._next_episode_start
            print(
                f"\n--- FACMAC Episode {self.episode_count + 1} started at "
                f"{nm}/{nd:02d} {nh:02d}:00 (duration: {duration_hours:.1f} h) ---\n"
            )

        global_state = self.env.get_global_state()
        local_obs = self.env.get_local_observations(global_state)
        joint_action = self.agent.select_actions(
            local_obs, evaluate=not self.training_mode
        )

        next_global, next_local, env_step_reward, done, _ = self.env.step_multi(joint_action)

        # Prefer components computed once inside step_multi (avoids double adaptive EMA).
        if self.env._last_reward_components is not None:
            components = self.env._last_reward_components
        else:
            reward_config = self.env.hvac_config.config["reward"]
            zone_count_for_reward = self.env.hvac_config.config["state_space"]["zone_temps"]["count"]
            raw_zone_temps_fallback = list(global_state[:zone_count_for_reward])
            components = self.env.compute_reward_components(
                reward_config, raw_zone_temps_fallback, joint_action
            )
        reward = components["reward"]
        # step_multi already added env_step_reward (== reward when components available)
        if abs(reward - env_step_reward) > 1e-9:
            self.env.episode_reward += reward - env_step_reward

        reward_config = self.env.hvac_config.config["reward"]
        zone_count_for_reward = self.env.hvac_config.config["state_space"]["zone_temps"]["count"]
        raw_zone_temps = []
        for zone_name, zhandles in list(self.env.handles["zones"].items())[:zone_count_for_reward]:
            h_temp = zhandles.get("temp", -1)
            if h_temp and h_temp > 0:
                raw_zone_temps.append(
                    self.api.exchange.get_variable_value(self.state, h_temp)
                )
        if not raw_zone_temps:
            raw_zone_temps = list(global_state[:zone_count_for_reward])

        if self.training_mode:
            self.agent.store_transition(
                global_state,
                local_obs,
                joint_action,
                reward,
                next_global,
                next_local,
                float(done),
            )
            if len(self.agent.memory) >= self.training_start_memory:
                losses = self.agent.update_parameters()
                if losses is not None:
                    self.training_update_count += 1
                    if self.loss_plotter is not None:
                        self.loss_plotter.update(self.training_update_count, losses)
                    if self.training_update_count == 1 or self.training_update_count % 10 == 0:
                        print(
                            "   FACMAC training: "
                            f"update={self.training_update_count}  "
                            f"memory_len={len(self.agent.memory)}  "
                            f"actor_loss={losses['actor_loss']:.4f}  "
                            f"critic_loss={losses['critic_loss']:.4f}  "
                            f"q_tot={losses['q_tot']:.4f}"
                        )

        current_hour = self.api.exchange.hour(self.state)
        current_day = self.api.exchange.day_of_month(self.state)
        current_month = self.api.exchange.month(self.state)
        episode_no = self.episode_count + 1

        decoded = [self.env.decode_agent_action(joint_action[i]) for i in range(self.env.n_agents)]
        avg_htg = float(np.mean([d["heating_sp"] for d in decoded]))
        avg_clg = float(np.mean([d["cooling_sp"] for d in decoded]))
        outdoor_co2 = self.env.get_outdoor_co2_ppm(current_month, current_day, current_hour)
        raw_outdoor_temp = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(
            self.state,
            self.api.exchange.hour(self.state),
            self.api.exchange.zone_time_step_number(self.state),
        )

        zone_temp = {}
        zone_co2 = {}
        zone_people = {}
        zone_ppd = {}
        zone_pmv = {}
        exchange = self.api.exchange
        for zone_name, zhandles in self.env.handles["zones"].items():
            h_temp = zhandles.get("temp", -1)
            zone_temp[f"temp_{zone_name}"] = (
                exchange.get_variable_value(self.state, h_temp)
                if h_temp and h_temp > 0
                else np.nan
            )
            h = zhandles.get("co2", -1)
            zone_co2[f"co2_{zone_name}"] = (
                exchange.get_variable_value(self.state, h) if h and h > 0 else np.nan
            )
            h_people = zhandles.get("people", -1)
            zone_people[f"people_{zone_name}"] = (
                exchange.get_variable_value(self.state, h_people)
                if h_people and h_people > 0
                else np.nan
            )
            h_ppd = zhandles.get("ppd", -1)
            zone_ppd[f"ppd_{zone_name}"] = (
                exchange.get_variable_value(self.state, h_ppd)
                if h_ppd and h_ppd > 0
                else np.nan
            )
            h_pmv = zhandles.get("pmv", -1)
            zone_pmv[f"pmv_{zone_name}"] = (
                exchange.get_variable_value(self.state, h_pmv)
                if h_pmv and h_pmv > 0
                else np.nan
            )

        ahu_oa_flows = {}
        for floor, handles in self.env.handles.get("ahu_oa", {}).items():
            h_oa = handles.get("mass_flow", -1)
            ahu_oa_flows[floor] = (
                exchange.get_variable_value(self.state, h_oa) if h_oa and h_oa > 0 else np.nan
            )
        valid_ahu_oa_flows = [v for v in ahu_oa_flows.values() if not np.isnan(v)]

        if not self._co2_400_warned and zone_co2:
            co2_vals = [v for v in zone_co2.values() if not np.isnan(v)]
            if co2_vals and all(abs(v - 400.0) < 1.0 for v in co2_vals):
                self._co2_400_warned = True
                print(
                    "\n  [CO2] All zone CO2 ≈ 400 ppm (outdoor default). "
                    "Occupancy is typically 0 until 07:00.\n"
                )

        floor_log = {}
        for i, floor in enumerate(AGENT_FLOORS):
            d = decoded[i]
            floor_log[f"{floor}_sp_offset"] = d["sp_offset"]
            floor_log[f"{floor}_deadband"] = d["deadband"]
            floor_log[f"{floor}_heating_setpoint"] = d["heating_sp"]
            floor_log[f"{floor}_cooling_setpoint"] = d["cooling_sp"]
            floor_log[f"{floor}_commanded_oa_mass_flow"] = d["oa_mass_flow"]
            floor_log[f"{floor}_raw_action"] = joint_action[i].tolist()

        log_entry = {
            "episode": episode_no,
            "timestep": self.env.timestep_count,
            "elapsed_hours": self.env.timestep_count / self.env.timesteps_per_hour,
            "action": joint_action.reshape(-1).tolist(),
            "reward": reward,
            "energy_cost": components["energy_cost"],
            "gas_cost": components["gas_cost"],
            "comfort_penalty": components["comfort_penalty"],
            "setpoint_penalty": components["setpoint_penalty"],
            "demand_penalty": components["demand_penalty"],
            "co2_penalty": components.get("co2_penalty", 0.0),
            "total_cost": components["total_cost"],
            "cost_normalization_mode": components.get("cost_normalization_mode", "absolute"),
            "cost_normalization_factor": components.get("cost_normalization_factor", 1.0),
            "adaptive_comfort_scale": components.get("adaptive_comfort_scale", 1.0),
            "adaptive_co2_scale": components.get("adaptive_co2_scale", 1.0),
            "adaptive_balancing_active": components.get("adaptive_balancing_active", False),
            "adaptive_n_samples": components.get("adaptive_n_samples", 0),
            "adaptive_ema_energy": components.get("adaptive_ema_energy"),
            "adaptive_ema_comfort": components.get("adaptive_ema_comfort"),
            "adaptive_ema_co2": components.get("adaptive_ema_co2"),
            "energy_price_used": components["energy_price_used"],
            "energy_kwh": components["energy_kwh"],
            "current_power": components["current_power"],
            "gas_kwh": components["gas_kwh"],
            "current_gas_power": components["current_gas_power"],
            "avg_zone_temp": float(np.mean(raw_zone_temps)),
            "outdoor_temp": raw_outdoor_temp,
            "outdoor_co2": outdoor_co2,
            # Averages for LiveRLPlotter compatibility
            "heating_setpoint": avg_htg,
            "cooling_setpoint": avg_clg,
            "setpoint_offset": float(np.mean([d["sp_offset"] for d in decoded])),
            "deadband": float(np.mean([d["deadband"] for d in decoded])),
            "commanded_oa_mass_flow": float(np.mean([d["oa_mass_flow"] for d in decoded])),
            "airflow": float(np.mean(valid_ahu_oa_flows)) if valid_ahu_oa_flows else np.nan,
            "bottom_floor_airflow": ahu_oa_flows.get("bottom", np.nan),
            "mid_floor_airflow": ahu_oa_flows.get("mid", np.nan),
            "top_floor_airflow": ahu_oa_flows.get("top", np.nan),
            "episode_reward": self.env.episode_reward,
            "hour": current_hour,
            "day": current_day,
            "month": current_month,
            "people": np.nansum(list(zone_people.values())) if zone_people else np.nan,
            **floor_log,
            **zone_temp,
            **zone_co2,
            **zone_people,
            **zone_ppd,
            **zone_pmv,
        }
        self.log_data.append(log_entry)
        if self.live_plotter is not None:
            self.live_plotter.update(log_entry)
        if self.loss_plotter is not None:
            self.loss_plotter.update_weights(log_entry)
        if self.episode_kpi_plotter is not None:
            self.episode_kpi_plotter.update_step(log_entry)

        episode_step = self.env.timestep_count
        actual_day = self.api.exchange.day_of_month(self.state)
        actual_hour = self.api.exchange.hour(self.state)
        timestep_minutes = 60 // self.env.timesteps_per_hour
        actual_minute = int(
            (
                self.api.exchange.minute_of_hour(self.state)
                if hasattr(self.api.exchange, "minute_of_hour")
                else (episode_step % self.env.timesteps_per_hour) * timestep_minutes
            )
        )
        datetime_str = f"2023-{current_month:02d}-{actual_day:02d} {actual_hour:02d}:{actual_minute:02d}"
        memory_len = len(self.agent.memory)
        memory_status = (
            f"{memory_len}/{self.training_start_memory}"
            if self.training_mode
            else str(memory_len)
        )

        print(f"{datetime_str} | Ep {episode_no:3d} | Step {episode_step:3d} | FACMAC")
        for i, floor in enumerate(AGENT_FLOORS):
            d = decoded[i]
            print(
                f"   [{floor:6s}] "
                f"raw=[{joint_action[i, 0]:+.2f},{joint_action[i, 1]:+.2f},{joint_action[i, 2]:+.2f}]  "
                f"htg={d['heating_sp']:.1f}°C  clg={d['cooling_sp']:.1f}°C  "
                f"db={d['deadband']:.2f}°C  oa_cmd={d['oa_mass_flow']:.3f}kg/s  "
                f"oa_flow={ahu_oa_flows.get(floor, np.nan):.3f}kg/s"
            )
        zone_str = ", ".join([f"{t:5.1f}" for t in raw_zone_temps[:5]])
        print(
            f"   States: zone_temps(first5)=[{zone_str}]°C  "
            f"outdoor_temp={raw_outdoor_temp:5.1f}°C  memory_len={memory_status}"
        )
        print(
            f"   Reward: {reward:7.3f}  episode_total={self.env.episode_reward:7.3f}  |  "
            f"elec_cost[$]={components['energy_cost']:.4f}  "
            f"gas_cost[$]={components['gas_cost']:.4f}  "
            f"comfort={components['comfort_penalty']:.4f}  "
            f"setpoint={components['setpoint_penalty']:.4f}  "
            f"demand_penalty[$]={components['demand_penalty']:.4f}  "
            f"co2={components.get('co2_penalty', 0.0):.4f}  "
            f"total_cost={components['total_cost']:.4f}  |  "
            f"price[$/kWh]={components['energy_price_used']:.3f}  "
            f"elec[kWh]={components['energy_kwh']:.4f}  "
            f"gas[kWh]={components['gas_kwh']:.4f}"
        )
        print()

        if done:
            self.episode_count += 1
            dur_h = getattr(self, "_current_episode_duration_hours", None)
            dur_str = f" (duration: {dur_h:.1f} h)" if dur_h is not None else ""
            print(f"\n--- FACMAC Episode {self.episode_count} completed{dur_str} ---\n")
            if self.episode_kpi_plotter is not None:
                self.episode_kpi_plotter.finalize_episode(self.episode_count)

            if (
                self.training_mode
                and self.save_model_enabled
                and self.save_every > 0
                and self.episode_count % self.save_every == 0
            ):
                ckpt_dir = (
                    os.path.join(self.output_dir, "checkpoints")
                    if self.output_dir
                    else "checkpoints"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    ckpt_dir, f"facmac_hvac_model_ep{self.episode_count}.pth"
                )
                self.save_model(ckpt_path)

            if self.episode_count >= self.max_episodes:
                print(f"Reached maximum episodes ({self.max_episodes}), logging disabled")
                self.max_episodes_reached = True
                return

            self._episode_started_this_run = False
            self._next_episode_start = _sample_random_start_in_window(
                self.training_window,
                after_month=current_month,
                after_day=current_day,
                after_hour=current_hour,
            )
            min_h, max_h = self.episode_duration_range
            self._next_episode_duration_hours = (
                min_h if min_h == max_h else np.random.uniform(min_h, max_h)
            )

    def get_summary(self):
        if not self.log_data:
            return "No control data logged"
        rewards = [d["reward"] for d in self.log_data]
        temps = [d["avg_zone_temp"] for d in self.log_data]
        return f"""
FACMAC Multi-Agent HVAC Control Summary:
  Agents: {', '.join(AGENT_FLOORS)} ({self.env.n_agents} floors)
  Total timesteps: {len(self.log_data)}
  Training mode: {self.training_mode}

  Performance:
    Total Reward: {self.env.episode_reward:.2f}
    Avg Reward/Step: {np.mean(rewards):.3f}
    Best Step Reward: {max(rewards):.3f}
    Worst Step Reward: {min(rewards):.3f}

  Temperature Control:
    Avg Zone Temp: {np.mean(temps):.2f}°C
    Temp Std Dev: {np.std(temps):.2f}°C
    Min Temp: {min(temps):.2f}°C
    Max Temp: {max(temps):.2f}°C
"""

    def save_model(self, path):
        bal = getattr(self.env, "adaptive_cost_balancer", None)
        extra = {"episode_count": self.episode_count}
        if bal is not None and bal.enabled:
            extra["adaptive_balancer"] = bal.state_dict()
        self.agent.save(path, extra_state=extra)
        print(f"Model saved to {path}")

    def save_log_to_csv(self, filepath):
        if not self.log_data:
            print("No data to save")
            return
        df = pd.DataFrame(self.log_data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.log_data)} timesteps to {filepath}")


def run_simulation(
    idf_path,
    epw_path,
    output_dir,
    config,
    max_episodes=None,
    training_mode=False,
    override_test=False,
    live_plot=False,
    loss_plot=False,
    adaptive_weight_plot=False,
    episode_kpi_plot=False,
    live_plot_every=1,
    live_plot_hold=False,
    live_plot_scope="current",
    model_path=None,
    save_model=False,
    save_every=20,
    simulation_overrides=None,
    facmac_config=None,
):
    """Run EnergyPlus with FACMAC multi-agent HVAC control."""
    hvac_config = get_hvac_config(config)
    if simulation_overrides:
        sim_override_cfg = hvac_config.config.setdefault("simulation", {})
        for key, value in simulation_overrides.items():
            if value is not None:
                sim_override_cfg[key] = value

    sim_cfg = hvac_config.config["simulation"]
    window = sim_cfg.get("training_window", {})
    start_month = window.get("start_month", sim_cfg.get("start_month", 6))
    start_day = window.get("start_day", sim_cfg.get("start_day", 1))
    end_month = window.get("end_month", start_month)
    end_day = window.get("end_day", start_day)
    start_hour = window.get("start_hour", 0)
    end_hour = window.get("end_hour", 24)
    ep_range = sim_cfg.get("episode_duration_hours", [24, 24])
    if isinstance(ep_range, list):
        ep_min, ep_max = ep_range[0], ep_range[1]
    else:
        ep_min = ep_max = sim_cfg.get("episode_hours", 24)

    if max_episodes is None:
        max_episodes = 99999

    if sim_cfg.get("custom_period", {}).get("enabled", False):
        season_info = get_season_info(start_month)
        sim_days = calculate_simulation_days(start_month, start_day, end_month, end_day)
        print("\n" + "=" * 60)
        print("TRAINING WINDOW (from config)")
        print("=" * 60)
        print(
            f"  Date range: {start_month}/{start_day:02d} to {end_month}/{end_day:02d} "
            f"({sim_days} days)"
        )
        print(f"  Time range: {start_hour}:00 to {end_hour}:00 (RL runs only in this window)")
        print(
            f"  Episode duration: [{ep_min}, {ep_max}] h (random per episode); "
            "start date/time: random in window"
        )
        print(f"  Season: {season_info['name']} ({season_info['description']})")
        print("=" * 60)
        outdoor_co2 = sim_cfg.get("outdoor_co2_ppm")
        outdoor_co2_csv = sim_cfg.get("outdoor_co2_csv_path")
        outdoor_co2_fallback = sim_cfg.get("outdoor_co2_fallback_ppm")
        occupancy_schedule_name = hvac_config.config.get("occupancy_events", {}).get("schedule_name")
        idf_path = create_custom_idf(
            idf_path,
            start_month,
            start_day,
            end_month,
            end_day,
            output_dir,
            outdoor_co2_ppm=outdoor_co2,
            outdoor_co2_csv_path=outdoor_co2_csv,
            outdoor_co2_fallback_ppm=outdoor_co2_fallback,
            occupancy_schedule_name=occupancy_schedule_name,
        )

    live_plotter = None
    loss_plotter = None
    episode_kpi_plotter = None
    if live_plot:
        live_plotter = LiveFACMACPlotter(
            output_dir=output_dir,
            update_every=live_plot_every,
            episode_scope=live_plot_scope,
        )
        print(
            f"Live plot enabled (updates every {max(1, int(live_plot_every))} "
            f"timestep(s), scope: {live_plot_scope})"
        )
    if loss_plot or adaptive_weight_plot:
        if training_mode or adaptive_weight_plot:
            loss_plotter = LiveFACMACLossPlotter(
                output_dir=output_dir,
                update_every=live_plot_every,
                show_losses=bool(loss_plot and training_mode),
                show_adaptive_weights=bool(adaptive_weight_plot),
            )
            bits = []
            if loss_plot and training_mode:
                bits.append("FACMAC losses")
            if adaptive_weight_plot:
                bits.append("adaptive weights")
            print(f"Live plot enabled: {', '.join(bits)}")
            if loss_plot and not training_mode:
                print("Warning: --loss-plot ignored without --training")
        else:
            print("Warning: --loss-plot requested without --training; no losses will be plotted")
    if episode_kpi_plot:
        episode_kpi_plotter = LiveEpisodeDollarKPIPlotter(output_dir=output_dir)
        print("Episode dollar KPI live plot enabled")

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    controller = FACMACHVACController(
        api,
        state,
        config,
        training_mode=training_mode,
        live_plotter=live_plotter,
        loss_plotter=loss_plotter,
        episode_kpi_plotter=episode_kpi_plotter,
        output_dir=output_dir,
        model_path=model_path,
        save_model=save_model,
        save_every=save_every,
        simulation_overrides=simulation_overrides,
        facmac_config_path=facmac_config,
    )
    controller.max_episodes = max_episodes

    if override_test:
        controller.env.weather_override = {
            "dry_bulb": 35.0,
            "humidity": 80.0,
            "wind_speed": 1.0,
        }
        controller.env.outdoor_co2_override = 650.0
        print("\n" + "=" * 60)
        print("[Override Test] Active — injecting fixed values every timestep")
        print("=" * 60 + "\n")

    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
        state, controller.pre_timestep_callback
    )
    api.runtime.callback_after_predictor_after_hvac_managers(
        state, controller.power_cache_callback
    )
    api.runtime.callback_end_zone_timestep_after_zone_reporting(
        state, controller.timestep_callback
    )

    print("\n" + "=" * 60)
    print("FACMAC Multi-Agent HVAC Control Configuration")
    print("=" * 60)
    print(f"  Agents:             {AGENT_FLOORS}")
    print(f"  Local obs size:     {controller.env.local_obs_size} (per floor; includes RTP)")
    print(f"  Global state size:  {controller.env.state_size} (mixer)")
    print(
        f"  Action per agent:   {controller.env.agent_action_size}  "
        "[sp_offset, deadband, oa_mass_flow]"
    )
    print(f"  Joint action size:  {controller.env.n_agents * controller.env.agent_action_size}")
    print(f"  Timesteps per hour: {controller.env.timesteps_per_hour}")
    print(f"  Training mode:      {controller.training_mode}")
    print(f"  Algorithm:          FACMAC (factored critic + centralised PG)")
    print("=" * 60)

    try:
        input("\nPress Enter to start simulation, or Ctrl+C to abort: ")
    except KeyboardInterrupt:
        print("\nSimulation aborted.")
        return 1
    except EOFError:
        print("\nNo interactive stdin detected; starting simulation.")

    eplus_args = ["-w", epw_path, "-d", output_dir, idf_path]
    exit_code = api.runtime.run_energyplus(state, eplus_args)
    if controller.fatal_error is not None:
        print(f"FACMAC controller failed: {controller.fatal_error}")
        exit_code = 1

    if controller.training_mode and controller.save_model_enabled and controller.fatal_error is None:
        final_model_path = os.path.join(output_dir, "facmac_hvac_model.pth")
        controller.save_model(final_model_path)

    if controller.fatal_error is None:
        controller.save_log_to_csv(os.path.join(output_dir, "facmac_hvac_log.csv"))
        if live_plotter is not None:
            live_plotter.finish(output_dir, hold=live_plot_hold)
        if loss_plotter is not None:
            loss_plotter.finish(output_dir, hold=live_plot_hold)
        if episode_kpi_plotter is not None:
            episode_kpi_plotter.finish(output_dir, hold=live_plot_hold)

    api.state_manager.delete_state(state)

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("SIMULATION COMPLETED SUCCESSFULLY")
    else:
        print(f"SIMULATION FAILED (exit code: {exit_code})")
    print("=" * 60)
    print(controller.get_summary())
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Run EnergyPlus with FACMAC multi-agent HVAC control."
    )
    parser.add_argument("--idf", type=str, default=None, help="Path to IDF file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hvac_config.yaml",
        help="Path to HVAC configuration file",
    )
    parser.add_argument(
        "--facmac-config",
        type=str,
        default="sac_config/facmac_config.yaml",
        help="Path to FACMAC agent hyperparameters",
    )
    parser.add_argument(
        "--epw",
        type=str,
        default="weather/chicago/TMY_lat41.88_lon-87.63.epw",
        help="Path to weather file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/multi_agent_rl_hvac_control",
        help="Output directory",
    )
    parser.add_argument("--training", action="store_true", help="Enable training mode")
    parser.add_argument("--model", type=str, help="Path to a pre-trained FACMAC model")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model checkpoints during training",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Episodes between periodic checkpoint saves",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument(
        "--override-test",
        action="store_true",
        help="Inject fixed weather/CO2 overrides every step",
    )
    parser.add_argument("--live-plot", action="store_true", help="Show live matplotlib plot")
    parser.add_argument(
        "--loss-plot",
        action="store_true",
        help="Show live FACMAC training loss plot (with --training)",
    )
    parser.add_argument(
        "--adaptive-weight-plot",
        action="store_true",
        help="Show live adaptive PPD/CO2 weight / EMA plot (combine with --loss-plot)",
    )
    parser.add_argument(
        "--episode-kpi-plot",
        action="store_true",
        help="Show live per-episode dollar and physical KPI plot",
    )
    parser.add_argument("--live-plot-every", type=int, default=1)
    parser.add_argument("--live-plot-hold", action="store_true")
    parser.add_argument(
        "--live-plot-scope", choices=["current", "all"], default="current"
    )
    parser.add_argument("--outdoor-co2-csv", type=str, default=None)
    parser.add_argument("--outdoor-co2-fallback", type=float, default=None)

    args = parser.parse_args()

    hvac_config = get_hvac_config(args.config)
    sim_cfg = hvac_config.config.get("simulation", {})
    idf_rel = args.idf or sim_cfg.get(
        "idf_path", "energyplus/control_models/MediumOffice_IAQ.idf"
    )
    args.idf = str(project_root / idf_rel) if not os.path.isabs(idf_rel) else idf_rel
    if not os.path.isabs(args.epw):
        args.epw = str(project_root / args.epw)
    if not os.path.isabs(args.output):
        args.output = str(project_root / args.output)
    if not os.path.isabs(args.facmac_config):
        args.facmac_config = str(project_root / args.facmac_config)

    if not os.path.exists(args.idf):
        print(f"Error: IDF file not found: {args.idf}")
        return 1
    if not os.path.exists(args.epw):
        print(f"Error: Weather file not found: {args.epw}")
        return 1
    if not os.path.exists(args.facmac_config):
        print(f"Error: FACMAC config not found: {args.facmac_config}")
        return 1

    simulation_overrides = {}
    if args.outdoor_co2_csv:
        co2_csv = args.outdoor_co2_csv
        if not os.path.isabs(co2_csv):
            co2_csv = str(project_root / co2_csv)
        if not os.path.exists(co2_csv):
            print(f"Error: Outdoor CO2 CSV not found: {co2_csv}")
            return 1
        simulation_overrides["outdoor_co2_csv_path"] = co2_csv
    if args.outdoor_co2_fallback is not None:
        simulation_overrides["outdoor_co2_fallback_ppm"] = args.outdoor_co2_fallback

    return run_simulation(
        args.idf,
        args.epw,
        args.output,
        args.config,
        max_episodes=args.episodes,
        training_mode=args.training,
        override_test=args.override_test,
        live_plot=args.live_plot,
        loss_plot=args.loss_plot,
        adaptive_weight_plot=args.adaptive_weight_plot,
        episode_kpi_plot=args.episode_kpi_plot,
        live_plot_every=args.live_plot_every,
        live_plot_hold=args.live_plot_hold,
        live_plot_scope=args.live_plot_scope,
        model_path=args.model,
        save_model=args.save_model,
        save_every=args.save_every,
        simulation_overrides=simulation_overrides or None,
        facmac_config=args.facmac_config,
    )


if __name__ == "__main__":
    sys.exit(main())
