import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import yaml
from typing import Tuple, List
import os

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 dropout: float = 0.2, activations: List[str] = None):
        """
        Initialize the DQN network.
        
        Args:
            input_size: Size of input layer
            hidden_layers: List of hidden layer sizes
            output_size: Size of output layer
            dropout: Dropout rate
            activations: List of activation functions for each layer (optional)
        """
        super(DQNNetwork, self).__init__()
        
        # Default to ReLU if no activations specified
        if activations is None:
            activations = ["relu"] * len(hidden_layers)
        
        # Validate activations list length
        if len(activations) != len(hidden_layers):
            raise ValueError(f"Number of activations ({len(activations)}) must match "
                           f"number of hidden layers ({len(hidden_layers)})")
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with configurable activations
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activations[i]))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),  # SiLU is PyTorch's implementation of Swish
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "prelu": nn.PReLU(),
            "softplus": nn.Softplus()
        }
        
        if activation_name.lower() not in activations:
            available = list(activations.keys())
            raise ValueError(f"Activation '{activation_name}' not supported. "
                           f"Available: {available}")
        
        return activations[activation_name.lower()]
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self, state_size: int, action_size: int, config_path: str = "config/config.yaml"):
        """Initialize DQN agent."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.state_size = state_size
        self.action_size = action_size
        # Device configuration
        device_config = self.config['training'].get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available!")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Hyperparameters
        self.learning_rate = self.config['training']['learning_rate']
        self.gamma = self.config['training']['gamma']
        self.epsilon = self.config['training']['epsilon_start']
        self.epsilon_end = self.config['training']['epsilon_end']
        self.epsilon_decay = self.config['training']['epsilon_decay']
        self.batch_size = self.config['training']['batch_size']
        self.target_update = self.config['training']['target_update']
        
        # Networks
        hidden_layers = self.config['model']['hidden_layers']
        dropout = self.config['model']['dropout']
        
        # Get activation functions (per-layer or single)
        activations = self._get_activations()
        
        self.q_network = DQNNetwork(state_size, hidden_layers, action_size, dropout, activations).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_layers, action_size, dropout, activations).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Replay buffer
        memory_size = self.config['training']['memory_size']
        self.memory = ReplayBuffer(memory_size)
        
        # Training stats
        self.training_step = 0
        self.losses = []
        self.q_values = []
        
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer (standardized method name)."""
        self.remember(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        self.q_values.append(q_values.max().item())
        
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.update_target_network()
        
        # Store loss
        self.losses.append(loss.item())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses,
            'q_values': self.q_values
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.training_step = checkpoint.get('training_step', 0)
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'memory_size': len(self.memory)
        }
    
    def _get_activations(self) -> List[str]:
        """Get activation functions from config (per-layer or single)."""
        model_config = self.config['model']
        hidden_layers = model_config['hidden_layers']
        
        # Check if per-layer activations are specified
        if 'layer_activations' in model_config and model_config['layer_activations']:
            layer_activations = model_config['layer_activations']
            
            # Validate length matches hidden layers
            if len(layer_activations) != len(hidden_layers):
                print(f"Warning: layer_activations length ({len(layer_activations)}) "
                      f"doesn't match hidden_layers length ({len(hidden_layers)})")
                print("Falling back to single activation function")
                return [model_config.get('activation', 'relu')] * len(hidden_layers)
            
            print(f"Using per-layer activations: {layer_activations}")
            return layer_activations
        else:
            # Use single activation for all layers
            single_activation = model_config.get('activation', 'relu')
            print(f"Using single activation '{single_activation}' for all layers")
            return [single_activation] * len(hidden_layers)
