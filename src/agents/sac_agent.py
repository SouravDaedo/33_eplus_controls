"""
Soft Actor-Critic (SAC) Agent for Trading
Implements SAC algorithm with continuous action space for trading decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import yaml
from typing import Tuple, List, Dict
import os

class Actor(nn.Module):
    """Actor network for SAC - outputs continuous actions."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int], 
                 dropout: float = 0.2, activations: List[str] = None, 
                 log_std_min: float = -20, log_std_max: float = 2):
        """
        Initialize Actor network.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            activations: List of activation functions per layer
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Default to ReLU if no activations specified
        if activations is None:
            activations = ["relu"] * len(hidden_layers)
        
        # Build shared layers
        layers = []
        prev_size = state_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activations[i]))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_size, action_size)
        self.log_std_layer = nn.Linear(prev_size, action_size)
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "prelu": nn.PReLU(),
            "softplus": nn.Softplus()
        }
        return activations[activation_name.lower()]
    
    def forward(self, state):
        """Forward pass through actor network."""
        x = self.shared_layers(state)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy distribution."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        
        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(x_t)
        
        # Calculate log probability with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean, log_std


class Critic(nn.Module):
    """Critic network for SAC - Q-function approximation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int],
                 dropout: float = 0.2, activations: List[str] = None):
        """Initialize Critic network."""
        super(Critic, self).__init__()
        
        # Default to ReLU if no activations specified
        if activations is None:
            activations = ["relu"] * len(hidden_layers)
        
        # Build network layers
        layers = []
        prev_size = state_size + action_size  # Concatenate state and action
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activations[i]))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (single Q-value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "prelu": nn.PReLU(),
            "softplus": nn.Softplus()
        }
        return activations[activation_name.lower()]
    
    def forward(self, state, action):
        """Forward pass through critic network."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class SACReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int, device: torch.device):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(reward).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).unsqueeze(1).to(self.device)
        )
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic agent for continuous control trading."""
    
    def __init__(self, state_size: int, action_size: int, config_path: str = "config/config.yaml"):
        """Initialize SAC agent."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Device configuration
        device_config = self.config.get('training', {}).get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available!")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # SAC hyperparameters
        sac_config = self.config.get('sac', {})
        self.learning_rate = sac_config.get('learning_rate', 3e-4)
        self.gamma = sac_config.get('gamma', 0.99)
        self.tau = sac_config.get('tau', 0.005)  # Soft update parameter
        self.alpha = sac_config.get('alpha', 0.2)  # Temperature parameter
        self.target_update_interval = sac_config.get('target_update_interval', 1)
        self.automatic_entropy_tuning = sac_config.get('automatic_entropy_tuning', True)
        self.batch_size = sac_config.get('batch_size', 256)
        
        # Network architecture
        hidden_layers = self.config['model']['hidden_layers']
        dropout = self.config['model']['dropout']
        activations = self._get_activations()
        
        # Actor network
        self.actor = Actor(state_size, action_size, hidden_layers, dropout, activations).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        
        # Critic networks (twin critics)
        self.critic1 = Critic(state_size, action_size, hidden_layers, dropout, activations).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_layers, dropout, activations).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)
        
        # Target critic networks
        self.target_critic1 = Critic(state_size, action_size, hidden_layers, dropout, activations).to(self.device)
        self.target_critic2 = Critic(state_size, action_size, hidden_layers, dropout, activations).to(self.device)
        
        # Initialize target networks
        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_size]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Replay buffer
        memory_size = sac_config.get('memory_size', 1000000)
        self.memory = SACReplayBuffer(memory_size, self.device)
        
        # Training stats
        self.training_step = 0
        self.actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.alpha_losses = []
        
    def _get_activations(self) -> List[str]:
        """Get activation functions from config."""
        model_config = self.config['model']
        hidden_layers = model_config['hidden_layers']
        
        if 'layer_activations' in model_config and model_config['layer_activations']:
            layer_activations = model_config['layer_activations']
            if len(layer_activations) == len(hidden_layers):
                return layer_activations
        
        # Fallback to single activation
        single_activation = model_config.get('activation', 'relu')
        return [single_activation] * len(hidden_layers)
    
    def select_action(self, state, evaluate: bool = False):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # Deterministic action for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        else:
            # Stochastic action for exploration
            with torch.no_grad():
                action, _, _, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""     
        self.memory.push(state, action, reward, next_state, done)
    
    def update_parameters(self):
        """Update SAC networks."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(self.batch_size)
        
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Update critics
        critic1_loss, critic2_loss = self._update_critics(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        
        # Update actor and alpha
        actor_loss, alpha_loss = self._update_actor_and_alpha(state_batch)
        
        # Soft update target networks
        if self.training_step % self.target_update_interval == 0:
            self.soft_update(self.target_critic1, self.critic1, self.tau)
            self.soft_update(self.target_critic2, self.critic2, self.tau)
        
        self.training_step += 1
        
        # Store losses
        self.critic1_losses.append(critic1_loss)
        self.critic2_losses.append(critic2_loss)
        self.actor_losses.append(actor_loss)
        if alpha_loss is not None:
            self.alpha_losses.append(alpha_loss)
        
        return {
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def _update_critics(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """Update critic networks."""
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_state_batch)
            target_q1 = self.target_critic1(next_state_batch, next_action)
            target_q2 = self.target_critic2(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(state_batch, action_batch)
        current_q2 = self.critic2(state_batch, action_batch)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def _update_actor_and_alpha(self, state_batch):
        """Update actor network and alpha (temperature)."""
        # Sample actions from current policy
        action, log_prob, _, _ = self.actor.sample(state_batch)
        
        # Q-values for sampled actions
        q1 = self.critic1(state_batch, action)
        q2 = self.critic2(state_batch, action)
        q = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature)
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
    def soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        """Hard update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_step': self.training_step,
            'actor_losses': self.actor_losses,
            'critic1_losses': self.critic1_losses,
            'critic2_losses': self.critic2_losses,
            'alpha_losses': self.alpha_losses
        }, filepath)
        print(f"SAC model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            
            if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                self.alpha = self.log_alpha.exp()
            
            self.training_step = checkpoint.get('training_step', 0)
            self.actor_losses = checkpoint.get('actor_losses', [])
            self.critic1_losses = checkpoint.get('critic1_losses', [])
            self.critic2_losses = checkpoint.get('critic2_losses', [])
            self.alpha_losses = checkpoint.get('alpha_losses', [])
            
            print(f"SAC model loaded from {filepath}")
        else:
            print(f"No SAC model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            'training_steps': self.training_step,
            'alpha': self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic1_loss': np.mean(self.critic1_losses[-100:]) if self.critic1_losses else 0,
            'avg_critic2_loss': np.mean(self.critic2_losses[-100:]) if self.critic2_losses else 0,
            'avg_alpha_loss': np.mean(self.alpha_losses[-100:]) if self.alpha_losses else 0,
            'memory_size': len(self.memory)
        }
