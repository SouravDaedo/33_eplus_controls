"""
HVAC Configuration Utility Module
Loads configuration and computes state/action space dimensions.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class HVACConfig:
    """HVAC configuration loader and state space calculator."""
    
    def __init__(self, config_path: str = None):
        """Initialize HVAC configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "hvac_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Compute dimensions
        self._compute_dimensions()
    
    def _compute_dimensions(self):
        """Compute state and action space dimensions from configuration."""
        # State space dimensions
        zone_count = self.config['state_space']['zone_temps']['count']
        current_weather_count = len(self.config['state_space']['current_weather'])
        forecast_horizon = self.config['state_space']['weather_forecast']['horizon']
        forecast_vars = self.config['state_space']['weather_forecast']['variables']
        time_features_count = len(self.config['state_space']['time_features'])
        prev_actions_count = self.config['state_space']['previous_actions']['count']
        
        self.state_size = (zone_count + 
                         current_weather_count + 
                         (forecast_horizon * forecast_vars) + 
                         time_features_count + 
                         prev_actions_count)
        
        # Action space dimensions
        self.action_size = len(self.config['action_space'])
        
        # Simulation parameters
        self.timesteps_per_hour = self.config['simulation']['timesteps_per_hour']
        self.episode_hours = self.config['simulation']['episode_hours']
        self.episode_timesteps = self.episode_hours * self.timesteps_per_hour
        
        # Control parameters
        self.base_temperature = self.config['control']['base_temperature']
        self.deadband_min = self.config['action_space']['deadband']['min']
        self.deadband_max = self.config['action_space']['deadband']['max']
        self.min_airflow = self.config['control']['min_airflow']
        self.max_airflow = self.config['control']['max_airflow']
        
        # Action bounds
        self.action_bounds = {
            'heating_offset': (
                self.config['action_space']['heating_offset']['min'],
                self.config['action_space']['heating_offset']['max']
            ),
            'deadband': (
                self.config['action_space']['deadband']['min'],
                self.config['action_space']['deadband']['max']
            ),
            'airflow_multiplier': (
                self.config['action_space']['airflow_multiplier']['min'],
                self.config['action_space']['airflow_multiplier']['max']
            )
        }
        
        # Reward weights
        self.reward_weights = self.config['reward']
        
        # Zone names and groups
        self.zone_names = self.config['zones']['names']
        self.zone_groups = self.config['zones']['groups']
    
    def get_state_size(self) -> int:
        """Get state space dimension."""
        return self.state_size
    
    def get_action_size(self) -> int:
        """Get action space dimension."""
        return self.action_size
    
    def get_timesteps_per_hour(self) -> int:
        """Get timesteps per hour."""
        return self.timesteps_per_hour
    
    def get_episode_timesteps(self) -> int:
        """Get total timesteps per episode."""
        return self.episode_timesteps
    
    def get_zone_names(self) -> List[str]:
        """Get list of zone names."""
        return self.zone_names
    
    def get_zone_groups(self) -> Dict[str, List[str]]:
        """Get zone group definitions."""
        return self.zone_groups
    
    def get_action_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get action bounds dictionary."""
        return self.action_bounds
    
    def get_reward_weights(self) -> Dict[str, float]:
        """Get reward function weights."""
        return self.reward_weights
    
    def print_config_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("HVAC Configuration Summary")
        print("=" * 60)
        print(f"Simulation:")
        print(f"  Timesteps per hour: {self.timesteps_per_hour}")
        print(f"  Episode duration: {self.episode_hours} hours")
        print(f"  Episode timesteps: {self.episode_timesteps}")
        print(f"\nState Space:")
        print(f"  State size: {self.state_size}")
        print(f"  Zone temperatures: {self.config['state_space']['zone_temps']['count']}")
        print(f"  Current weather: {len(self.config['state_space']['current_weather'])}")
        print(f"  Weather forecast: {self.config['state_space']['weather_forecast']['horizon']} × {self.config['state_space']['weather_forecast']['variables']}")
        print(f"  Time features: {len(self.config['state_space']['time_features'])}")
        print(f"  Previous actions: {self.config['state_space']['previous_actions']['count']}")
        print(f"\nAction Space:")
        print(f"  Action size: {self.action_size}")
        for action_name, bounds in self.action_bounds.items():
            print(f"  {action_name}: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
        print(f"\nControl:")
        print(f"  Base temperature: {self.base_temperature}°C")
        print(f"  Deadband range: [{self.deadband_min}, {self.deadband_max}]°C")
        print(f"  Airflow range: [{self.min_airflow}, {self.max_airflow}] m³/s")
        print("=" * 60)


# Global configuration instance
_hvac_config = None


def get_hvac_config(config_path: str = None) -> HVACConfig:
    """Get global HVAC configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        HVACConfig instance
    """
    global _hvac_config
    if _hvac_config is None:
        _hvac_config = HVACConfig(config_path)
    return _hvac_config


def reset_hvac_config():
    """Reset global HVAC configuration (useful for testing)."""
    global _hvac_config
    _hvac_config = None
