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

### Training with Checkpointing
```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 4 \
  --training \
  --save-model \
  --save-every 50 \
  --live-plot \
  --loss-plot \
  --live-plot-hold
```
`--save-model` enables saving; `--save-every N` (default 20) writes a checkpoint every
N completed episodes to `<output>/checkpoints/rl_hvac_model_ep{N}.pth` (one file per
checkpoint, nothing overwritten), plus a final save to `<output>/rl_hvac_model.pth`
when the run ends. Without `--save-model`, nothing is saved at all. Note the interval
must be ≤ `--episodes` to actually fire mid-run — e.g. `--episodes 4 --save-every 50`
above will only produce the final save.

To resume training or evaluate a specific checkpoint, point `--model` at any saved
file:
```bash
python tests/rl_hvac_control.py --config config/hvac_config_24h.yaml \
  --model outputs/rl_hvac_24h/checkpoints/rl_hvac_model_ep50.pth
```

## Key Features

### Configurable Timesteps
- Default: 4 timesteps per hour (15-minute intervals)
- Easily change to 1 (60-min), 12 (5-min), etc.
- Automatically updates episode length and state calculations

### State Space
- **Zone temperatures**: 10 zones (configurable)
- **Current weather**: 3 variables (temp, humidity, cloud cover)
- **Weather forecast**: per-variable, forward-looking (see below)
- **Weather history** *(optional)*: lagged past weather (see below)
- **Time features**: Hour, day of week, month (normalized)
- **Previous actions**: Last 3 actions (heating offset, deadband, airflow)

### Weather Forecast (`state_space.weather_forecast`)
Forecast values are real forward-looking readings pulled from the EPW weather file
via EnergyPlus's `today_weather_*_at_time` / `tomorrow_weather_*_at_time` API — not a
repeat of past/current observations. Each variable specifies its own independent list
of timesteps-ahead to forecast:

```yaml
weather_forecast:
  outdoor_air_temperature: [1, 3, 5]   # look further ahead on temperature...
  relative_humidity: [1]               # ...but only 1 step ahead on humidity...
  cloud_cover: [1, 3]                  # ...and 2 offsets for cloud cover
```

**Forecast noise**: since EPW look-ahead is otherwise a *perfect* forecast (unrealistic
for a real controller), Gaussian noise can be added so the agent trains against
realistic forecast uncertainty. Std grows linearly with lead time — a forecast 5
timesteps out gets 5× the noise of one 1 timestep out:

```yaml
  noise:
    enabled: true
    oat_std_per_step: 0.3          # °C std per timestep of lead time
    humidity_std_per_step: 1.0     # % std per timestep of lead time
    cloud_cover_std_per_step: 1.0  # °C std per timestep of lead time
```
Set `noise.enabled: false` for a perfect forecast (e.g. as a baseline comparison run).

### Weather History (`state_space.weather_history`, optional)
Disabled by default. When enabled, adds lagged *observed* weather (1..N timesteps in
the past) as extra state features alongside the forecast:
```yaml
weather_history:
  enabled: false
  horizon: 6   # number of past timesteps to include
```

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
4. Implement zone-specific control strategies
