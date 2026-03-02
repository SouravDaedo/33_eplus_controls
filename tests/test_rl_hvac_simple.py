"""
Simple test of RL HVAC control without EnergyPlus dependency.
Tests the RL agent and environment logic in isolation.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing SAC agent and HVAC config
from src.agents.sac_agent import SACAgent
from src.utils.hvac_config import get_hvac_config


class MockHVACEnvironment:
    """Mock HVAC environment for testing RL agent without EnergyPlus."""
    
    def __init__(self, config_path=None):
        # Load HVAC configuration
        self.hvac_config = get_hvac_config(config_path)
        
        # Control parameters from config
        self.base_temp = self.hvac_config.base_temperature
        self.deadband_min = self.hvac_config.deadband_min
        self.deadband_max = self.hvac_config.deadband_max
        self.min_airflow = self.hvac_config.min_airflow
        self.max_airflow = self.hvac_config.max_airflow
        
        # State and action dimensions from config
        self.state_size = self.hvac_config.get_state_size()
        self.action_size = self.hvac_config.get_action_size()
        self.timesteps_per_hour = self.hvac_config.get_timesteps_per_hour()
        self.episode_timesteps = self.hvac_config.get_episode_timesteps()
        
        # Current state tracking
        self.current_state = None
        self.current_action = None
        self.prev_temp = None
        self.prev_energy = 0.0
        
        # Episode tracking
        self.timestep_count = 0
        self.episode_reward = 0.0
        self.energy_consumption = []
        
        # Weather simulation
        self.weather_history = []
        self.forecast_horizon = self.hvac_config.config['state_space']['weather_forecast']['horizon']
        
        # Initialize state
        self.reset()
    
    def get_current_state(self):
        """Get current environment state (simulated)."""
        # Get zone count from config
        zone_count = self.hvac_config.config['state_space']['zone_temps']['count']
        
        # Simulate zone temperatures
        zone_temps = np.random.normal(self.base_temp, 2.0, zone_count).tolist()
        
        # Simulate current weather
        oat = 15.0 + 10 * np.sin(self.timestep_count * 0.01)  # Daily temperature cycle
        humidity = 0.5 + 0.2 * np.sin(self.timestep_count * 0.005)
        cloud_cover = 0.3 + 0.4 * np.random.random()
        
        current_weather = [oat, humidity, cloud_cover]
        
        # Update weather history
        self.weather_history.append(current_weather)
        if len(self.weather_history) > self.forecast_horizon:
            self.weather_history.pop(0)
        
        # Forecasted weather (simple persistence forecast)
        forecast_weather = []
        for i in range(self.forecast_horizon):
            if i < len(self.weather_history):
                forecast_weather.extend(self.weather_history[-(i+1)])
            else:
                forecast_weather.extend(current_weather)
        
        # Time features
        hour = (self.timestep_count // self.timesteps_per_hour) % 24
        day = (self.timestep_count // (self.timesteps_per_hour * 24)) % 7
        month = ((self.timestep_count // (self.timesteps_per_hour * 24 * 30)) % 12) + 1
        
        time_features = [hour/24.0, day/7.0, month/12.0]
        
        # Previous action
        prev_action = self.current_action if self.current_action is not None else [0.0, 1.0, 0.5]
        
        # Combine all features (should match config state_size)
        all_features = zone_temps + list(current_weather) + list(forecast_weather) + list(time_features) + list(prev_action)
        state = np.array(all_features[:self.state_size])  # Trim to exact state size
        
        return state.astype(np.float32)
    
    def apply_action(self, action):
        """Apply action from RL agent (simulated)."""
        # Parse action using config bounds
        action_bounds = self.hvac_config.get_action_bounds()
        sp_offset = np.clip(action[0], action_bounds['sp_offset'][0], action_bounds['sp_offset'][1])
        deadband = np.clip(action[1], action_bounds['deadband'][0], action_bounds['deadband'][1])
        airflow_mult = np.clip(action[2], action_bounds['airflow_multiplier'][0], action_bounds['airflow_multiplier'][1])
        
        # Calculate setpoints
        heating_sp = self.base_temp + sp_offset - deadband/2
        cooling_sp = self.base_temp + sp_offset + deadband/2
        
        # Store action
        self.current_action = action
        
        return heating_sp, cooling_sp, deadband, airflow_mult
    
    def calculate_reward(self, prev_state, action, curr_state):
        """Calculate reward based on energy efficiency and thermal comfort."""
        # Simulate energy consumption based on action
        sp_offset = action[0]
        deadband = action[1]
        airflow_mult = action[2]
        
        # Energy cost (higher airflow and tighter deadband = more energy)
        energy_cost = 0.1 * airflow_mult + 0.05 * (3.0 - deadband)
        if abs(sp_offset) > 2.0:
            energy_cost += 0.1 * abs(sp_offset - 2.0)
        
        # Thermal comfort reward
        zone_temps = curr_state[:5]
        comfort_penalty = 0.0
        for temp in zone_temps:
            deviation = abs(temp - self.base_temp)
            if deviation > 1.0:  # Outside comfort band
                comfort_penalty += (deviation - 1.0) * 0.5
        
        # Setpoint reasonableness penalty
        setpoint_penalty = 0.0
        if abs(sp_offset) > 2.0:
            setpoint_penalty += 0.1
        if deadband < 1.0:
            setpoint_penalty += 0.05
        
        # Total reward (negative because we want to minimize costs)
        reward = -(energy_cost + comfort_penalty + setpoint_penalty)
        
        return reward
    
    def step(self, action):
        """Execute one environment step."""
        prev_state = self.current_state.copy() if self.current_state is not None else None
        
        # Apply action
        self.apply_action(action)
        
        # Get new state
        self.current_state = self.get_current_state()
        
        # Calculate reward
        if prev_state is not None:
            reward = self.calculate_reward(prev_state, action, self.current_state)
        else:
            reward = 0.0
        
        # Update tracking
        self.episode_reward += reward
        self.timestep_count += 1
        
        # Check if episode is done
        done = self.timestep_count >= self.episode_timesteps
        
        return self.current_state, reward, done, {}
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_state = self.get_current_state()
        self.current_action = None
        self.episode_reward = 0.0
        self.timestep_count = 0
        self.weather_history = []
        return self.current_state


def test_rl_agent():
    """Test the RL agent with mock HVAC environment."""
    print("=" * 60)
    print("RL HVAC Control Test (Mock Environment)")
    print("=" * 60)
    
    # Initialize environment with config
    env = MockHVACEnvironment()
    env.hvac_config.print_config_summary()
    
    # Initialize RL agent
    config_path = project_root / "sac_config" / "sac_config.yaml"
    agent = SACAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config_path=str(config_path)
    )
    print(f"Agent initialized from {config_path}")
    
    # Test episode
    timestep_minutes = 60 // env.timesteps_per_hour
    print(f"\nRunning test episode ({env.hvac_config.episode_hours} hours, {timestep_minutes}-min timesteps)...\n")
    
    state = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while True:
        # Select action (evaluation mode)
        action = agent.select_action(state, evaluate=True)
        
        # Execute step
        next_state, reward, done, info = env.step(action)
        
        # Store experience (for potential training)
        agent.store_transition(state, action, reward, next_state, float(done))
        
        total_reward += reward
        step_count += 1
        
        # Print progress every hour
        if step_count % env.timesteps_per_hour == 0:
            hour = step_count // env.timesteps_per_hour
            avg_temp = np.mean(state[:env.hvac_config.config['state_space']['zone_temps']['count']])
            print(f"  Hour {hour:2d} | Step {step_count:3d} | "
                  f"Temp: {avg_temp:5.1f}°C | "
                  f"Action: [{action[0]:5.2f}, {action[1]:4.2f}, {action[2]:4.2f}] | "
                  f"Reward: {reward:6.3f}")
        
        state = next_state
        
        if done:
            break
    
    print(f"\nEpisode completed:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward/step: {total_reward/step_count:.3f}")
    
    # Test agent statistics
    stats = agent.get_training_stats()
    print(f"\nAgent memory size: {stats['memory_size']}")
    
    # Test a few training updates
    if len(agent.memory) >= agent.batch_size:
        print("\nTesting training updates...")
        for i in range(5):
            agent.update_parameters()
        stats = agent.get_training_stats()
        print(f"  Actor loss: {stats['avg_actor_loss']:.4f}")
        print(f"  Critic1 loss: {stats['avg_critic1_loss']:.4f}")
        print(f"  Critic2 loss: {stats['avg_critic2_loss']:.4f}")
    
    return agent, env


def main():
    """Run the test."""
    try:
        agent, env = test_rl_agent()
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
