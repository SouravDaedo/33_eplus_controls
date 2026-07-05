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

        # Weather forecast: per-variable list of timesteps-ahead to forecast — each
        # variable can look ahead by a different amount (or not be forecast at all).
        wf_cfg = self.config['state_space']['weather_forecast']
        self.weather_forecast_offsets = {
            'oat': list(wf_cfg.get('outdoor_air_temperature', [])),
            'humidity': list(wf_cfg.get('relative_humidity', [])),
            'cloud_cover': list(wf_cfg.get('cloud_cover', [])),
        }
        forecast_count = sum(len(v) for v in self.weather_forecast_offsets.values())

        # Forecast error: Gaussian noise std (per timestep of lead time) added to
        # forecast values so the agent sees realistic uncertainty, not perfect foresight.
        noise_cfg = wf_cfg.get('noise', {})
        self.weather_forecast_noise = {
            'enabled': bool(noise_cfg.get('enabled', False)),
            'oat': float(noise_cfg.get('oat_std_per_step', 0.0)),
            'humidity': float(noise_cfg.get('humidity_std_per_step', 0.0)),
            'cloud_cover': float(noise_cfg.get('cloud_cover_std_per_step', 0.0)),
        }

        time_features_count = len(self.config['state_space']['time_features'])
        prev_actions_count = self.config['state_space']['previous_actions']['count']
        outdoor_co2_count = self.config['state_space'].get('outdoor_co2_ppm', {}).get('count', 0)
        zone_co2_count = self.config['state_space'].get('zone_co2_ppm', {}).get('count', 0)

        # Optional past weather (lagged history), disabled by default
        weather_history_cfg = self.config['state_space'].get('weather_history', {})
        self.weather_history_enabled = bool(weather_history_cfg.get('enabled', False))
        self.weather_history_horizon = weather_history_cfg.get('horizon', 0) if self.weather_history_enabled else 0
        weather_history_vars = weather_history_cfg.get('variables', current_weather_count)

        self.state_size = (zone_count +
                         current_weather_count +
                         forecast_count +
                         (self.weather_history_horizon * weather_history_vars) +
                         time_features_count +
                         prev_actions_count +
                         outdoor_co2_count +
                         zone_co2_count)
        
        # Action space dimensions
        self.action_size = len(self.config['action_space'])
        
        # Simulation parameters
        self.timesteps_per_hour = self.config['simulation']['timesteps_per_hour']
        ep = self.config['simulation'].get('episode_duration_hours')
        if isinstance(ep, list):
            self.episode_duration_min, self.episode_duration_max = ep[0], ep[1]
            self.episode_hours = (self.episode_duration_min + self.episode_duration_max) / 2.0
        else:
            self.episode_hours = self.config['simulation'].get('episode_hours', 24)
            self.episode_duration_min = self.episode_duration_max = self.episode_hours
        self.episode_timesteps = int(self.episode_hours * self.timesteps_per_hour)
        self.training_window = self.config['simulation'].get('training_window', {})
        
        # Control parameters
        self.base_temperature = self.config['control']['base_temperature']
        self.deadband_min = self.config['action_space']['deadband']['min']
        self.deadband_max = self.config['action_space']['deadband']['max']
        self.min_airflow = self.config['control']['min_airflow']
        self.max_airflow = self.config['control']['max_airflow']
        
        # Action bounds
        self.action_bounds = {
            'sp_offset': (
                self.config['action_space']['sp_offset']['min'],
                self.config['action_space']['sp_offset']['max']
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
        """Get total timesteps per episode (default/midpoint). Use set_episode_timesteps for variable length."""
        return self.episode_timesteps

    def get_episode_duration_range(self) -> Tuple[float, float]:
        """Get (min_hours, max_hours) for episode duration."""
        return (self.episode_duration_min, self.episode_duration_max)

    def get_training_window(self) -> Dict:
        """Get training window: start_month, start_day, start_hour, end_month, end_day, end_hour."""
        return self.training_window
    
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

    def is_normalization_enabled(self) -> bool:
        """Return True if min-max state normalization is enabled."""
        return self.config.get('normalization', {}).get('enabled', False)

    def get_state_normalization_bounds(self):
        """Return (mins, maxs) numpy arrays matching the state vector layout.

        Each element corresponds to one state feature in the same order as get_current_state().
        Used for min-max normalization: normalized = clip((x - min) / (max - min), 0, 1).
        """
        import numpy as np
        b = self.config.get('normalization', {}).get('bounds', {})

        def _b(key, default_min, default_max, count=1):
            lo, hi = b.get(key, [default_min, default_max])
            return [(lo, hi)] * count

        ss = self.config['state_space']
        zone_count  = ss['zone_temps']['count']
        n_time      = len(ss['time_features'])
        n_prev      = ss['previous_actions']['count']
        n_out_co2   = ss.get('outdoor_co2_ppm', {}).get('count', 0)
        n_zone_co2  = ss.get('zone_co2_ppm',    {}).get('count', 0)
        past_horizon = self.weather_history_horizon
        wf = self.weather_forecast_offsets

        def _weather_block(n_steps):
            # One (oat, humidity, sky_temp) triple per step, interleaved — matches the
            # per-timestep [oat, humidity, sky_temp] order get_current_state() builds.
            block = []
            for _ in range(n_steps):
                block += _b('oat', -20.0, 50.0, 1)
                block += _b('humidity', 0.0, 100.0, 1)
                block += _b('sky_temp', -30.0, 40.0, 1)
            return block

        pairs = (
            _b('zone_temp',     10.0,  40.0,  zone_count) +
            _weather_block(1) +           # current weather
            # Forecast weather: grouped per variable (oat offsets, then humidity
            # offsets, then cloud_cover offsets) — matches get_current_state() order,
            # since each variable can have a different number of forecast offsets.
            _b('oat',          -20.0,  50.0,  len(wf['oat'])) +
            _b('humidity',       0.0, 100.0,  len(wf['humidity'])) +
            _b('sky_temp',     -30.0,  40.0,  len(wf['cloud_cover'])) +
            _weather_block(past_horizon) + # past weather (lagged, interleaved per step)
            _b('hour_of_day',    0.0,   1.0,  1) +
            _b('day_of_week',    0.0,   1.0,  1) +
            _b('month_of_year',  0.0,   1.0,  1) +
            _b('prev_action',   -1.0,   1.0,  n_prev) +
            _b('outdoor_co2',  300.0, 1200.0, n_out_co2) +
            _b('zone_co2',     300.0, 2000.0, n_zone_co2)
        )

        mins = np.array([p[0] for p in pairs], dtype=np.float32)
        maxs = np.array([p[1] for p in pairs], dtype=np.float32)
        return mins, maxs
    
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
        wf = self.weather_forecast_offsets
        print(f"  Weather forecast offsets: oat={wf['oat']}, humidity={wf['humidity']}, cloud_cover={wf['cloud_cover']}")
        wn = self.weather_forecast_noise
        if wn['enabled']:
            print(f"  Forecast noise (std/step): oat={wn['oat']}, humidity={wn['humidity']}, cloud_cover={wn['cloud_cover']}")
        else:
            print(f"  Forecast noise: disabled (perfect forecast)")
        if self.weather_history_enabled:
            wh = self.config['state_space']['weather_history']
            print(f"  Weather history (past): {wh['horizon']} × {wh.get('variables', 3)}")
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
