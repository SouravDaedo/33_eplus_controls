# HVAC Configuration System

## Overview
The HVAC control system now uses a centralized configuration file to define:
- State space dimensions and components
- Action space bounds
- Simulation parameters (timesteps, episode duration)
- Control parameters
- Reward function weights

## Configuration Files

### 1. `config/hvac_config.yaml`
Main configuration file defining:
- **Simulation**: Timesteps per hour (4 = 15-min), episode duration
- **State Space**: Zone temps, weather, forecast, time features, previous actions
- **Action Space**: Heating offset, deadband, airflow multiplier bounds
- **Control**: Base temperature, airflow limits
- **Reward**: Weights for energy, comfort, setpoint penalties
- **Zones**: Zone names and groupings

### 2. `src/utils/hvac_config.py`
Utility module that:
- Loads configuration from YAML
- Computes state/action space dimensions
- Provides easy access to parameters
- Prints configuration summary

## Usage

### Mock Testing (without EnergyPlus)
```bash
python tests/test_rl_hvac_simple.py
```
Shows configuration summary and tests RL agent with simulated environment.

### Real EnergyPlus Simulation
```bash
python tests/rl_hvac_control.py --config config/hvac_config.yaml
```
Runs RL control with actual EnergyPlus simulation.

## Key Features

### Configurable Timesteps
- Default: 4 timesteps per hour (15-minute intervals)
- Easily change to 1 (60-min), 12 (5-min), etc.
- Automatically updates episode length and state calculations

### State Space
- **Zone temperatures**: 10 zones (configurable)
- **Current weather**: 3 variables (temp, humidity, cloud cover)
- **Weather forecast**: 6 timesteps × 3 variables
- **Time features**: Hour, day of week, month (normalized)
- **Previous actions**: Last 3 actions (heating offset, deadband, airflow)

### Action Space
- **Heating offset**: ±5°C from base temperature
- **Deadband**: 0.5-3.0°C between heating/cooling
- **Airflow multiplier**: 0.1-2.0× outdoor air flow

### Reward Function
- **Energy penalty**: Higher airflow and tighter deadband cost more
- **Comfort penalty**: Temperature deviations from setpoint
- **Setpoint penalty**: Extreme setpoint changes discouraged

## IDF Configuration
The EnergyPlus model (`MediumOffice_IAQ.idf`) is set to:
- `Timestep,4;` for 15-minute intervals
- CO2 tracking for IAQ monitoring
- VAV system for zone control

## Next Steps
1. Install `pyenergyplus-lbnl` for EnergyPlus API
2. Train RL agent with multiple episodes
3. Experiment with different zone groupings
4. Add weather forecast integration
5. Implement zone-specific control strategies
