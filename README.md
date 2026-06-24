# EnergyPlus RL Controls

Reinforcement learning–based HVAC control using the EnergyPlus Python API. Includes a full SAC agent, real-time energy pricing, gas cost tracking, CO₂ monitoring, and programmable weather/CO₂ overrides — all driven from Python with no manual EnergyPlus installation.

**EnergyPlus version:** The control models in `energyplus/control_models/` (e.g. `MediumOffice_IAQ.idf`) are for **EnergyPlus 23.2**. Use that version of the EnergyPlus Python API for compatibility. Other versions may work if the IDF is upgraded or downgraded with the EnergyPlus IDFVersionUpdater/Transition tools.

---

## Quick Start

```bash
pip install -r requirements.txt
conda activate bem
cd tests
python rl_hvac_control.py --config ../config/hvac_config.yaml --episodes 1
```

Run with `run_rl_hvac.bat` from the project root for a one-click launch.

---

## Plotting simulation results

After running the RL HVAC simulation, results are written to a CSV (default: `outputs/rl_hvac_control/rl_hvac_log.csv`; override with `--output-dir`).

**To plot:**

1. Run the simulation at least once:
   ```bash
   python tests/rl_hvac_control.py --config config/hvac_config.yaml --episodes 1
   ```
2. Open the IAQ results notebook and run all cells:
   - **Jupyter:** Open `tests/plot_iaq_results.ipynb` (from project root or from `tests/`).
   - The notebook loads `outputs/rl_hvac_control/rl_hvac_log.csv` and plots:
     - CO₂ concentration (by zone and over time)
     - Zone temperatures and thermal comfort (PPD/PMV)
     - Reward components (energy cost, comfort, setpoint, demand)
     - Optional: choose a single episode via `EPISODE_TO_PLOT` (int, list, or `None` for all).

If you used a custom output directory, set `log_path` in the notebook’s “Load Data” cell to that path (e.g. `project_root / 'outputs' / 'my_run' / 'rl_hvac_log.csv'`).

---

## Directory Structure

```
eplus_controls/
├── config/                  # Simulation and RL configuration
├── energyplus/              # IDF building models (EnergyPlus 23.2)
├── src/                     # Core library code
│   ├── agents/              # RL agents (SAC, DQN)
│   ├── utils/               # Helpers (config, IDF, pricing, weather)
│   └── visualization/       # Plotting utilities
├── tests/                   # Runnable simulation scripts
├── weather/                 # EPW weather files by city/year
├── outputs/                 # Simulation results (auto-created)
├── sac_config/              # SAC hyperparameter configs
└── examples/                # Standalone usage examples
```

---

## Key Files

### Entry Points

| File | Purpose |
|------|---------|
| `run_rl_hvac.bat` | One-click launcher — activates conda env and runs the RL HVAC simulation |
| `tests/rl_hvac_control.py` | **Main simulation script** — EnergyPlus + SAC agent loop (see below) |
| `tests/test_rl_hvac_simple.py` | Mock environment test — runs the RL agent without EnergyPlus (fast unit test) |
| `tests/plot_iaq_results.ipynb` | **Plot RL HVAC results** — CO₂, zone temps, reward components, PPD/PMV (uses `outputs/rl_hvac_control/rl_hvac_log.csv`) |
| `plot_results.ipynb` | Plots external-controller simulation — temperatures, power, setpoints (uses `outputs/…/simulation_log.csv`) |
| `download_weather.py` | Download EPW weather files from Open-Meteo or PVGIS |
| `integrated_simulation.py` | Combines building + solar PV + battery + pricing in a single simulation |
| `battery_model.py` | Configurable battery energy storage model (charge/discharge state machine) |
| `solar_pv_model.py` | PVWatts-based solar power model driven by EPW weather |
| `energy_price_model.py` | Real-time, TOU, and dynamic electricity pricing models |

---

### `tests/rl_hvac_control.py` — Main RL Simulation

The core file. Runs EnergyPlus with the SAC agent controlling three actions per timestep:

| Action | Key | Range | Meaning |
|--------|-----|-------|---------|
| Setpoint offset | `sp_offset` | −5 to +5 °C | Shifts heating and cooling setpoints equally from `base_temperature` |
| Deadband | `deadband` | 2 to 10 °C | Gap between heating and cooling setpoints |
| Airflow multiplier | `airflow_multiplier` | 0.1 to 2.0 | Scales outdoor air infiltration rate |

**Reward components** (all configurable in `hvac_config.yaml`):

| Component | Description |
|-----------|-------------|
| `elec_cost` | Electricity cost: `kWh × price × energy_weight` |
| `gas_cost` | Gas cost: `kWh_thermal × gas_price × gas_weight` |
| `comfort` | Penalty when zone temps deviate from setpoint by more than `comfort_threshold` |
| `demand_penalty` | Penalty when electricity demand exceeds `demand_threshold` [kW] |
| `setpoint` | Small penalty for extreme or narrow setpoints |

**Key classes:**

| Class | Role |
|-------|------|
| `HVACEnvironment` | Wraps EnergyPlus state/action/reward. Holds handles, caches, overrides. |
| `RLHVACController` | Owns the SAC agent and EnergyPlus callbacks. Drives the episode loop. |

**Callbacks registered (in order):**

1. `callback_begin_zone_timestep_before_init_heat_balance` → `pre_timestep_callback` — applies CO₂ and weather overrides before zone calculations
2. `callback_after_predictor_after_hvac_managers` → `power_cache_callback` — caches electricity/gas demand rates before they reset
3. `callback_end_zone_timestep_after_zone_reporting` → `timestep_callback` — main RL loop (observe → act → reward → log)

**Programmatic overrides** (set anytime on `controller.env`):

```python
# Override outdoor weather each timestep instead of using EPW
controller.env.weather_override = {
    'dry_bulb':      28.5,   # °C
    'humidity':      60.0,   # %
    'wind_speed':    2.0,    # m/s
    'beam_solar':    700.0,  # W/m²
    'diffuse_solar': 90.0,   # W/m²
}

# Override outdoor CO₂ each timestep
controller.env.outdoor_co2_override = 500.0  # ppm
```

**CLI flags:**

```bash
python rl_hvac_control.py \
  --config  ../config/hvac_config.yaml \
  --epw     ../weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output  ../outputs/my_run \
  --episodes 5 \
  --training        # enable SAC learning
  --override-test   # inject dry_bulb=35°C, CO₂=650ppm to verify override pipeline
```

**Outputs** (written to `--output` directory, default `outputs/rl_hvac_control/`):

| File | Contents |
|------|---------|
| `rl_hvac_log.csv` | Per-timestep log: reward components, temperatures, power [kW], gas [kW], CO₂, actions |
| `rl_hvac_model.pth` | Trained SAC model (only with `--training`) |
| `eplusout.err` | EnergyPlus warnings and errors |
| `eplustbl.htm` | Annual energy summary table |

---

### `config/hvac_config.yaml` — All Tunable Parameters

| Section | Key settings |
|---------|-------------|
| `simulation` | `idf_path`, `timesteps_per_hour`, `training_window` (date/time range for RL), `episode_duration_hours`, `outdoor_co2_ppm` |
| `action_space` | `sp_offset` bounds, `deadband` bounds, `airflow_multiplier` bounds |
| `reward` | `energy_weight`, `gas_weight`, `gas_price_per_kwh`, `comfort_weight`, `demand_weight`, `demand_threshold` [kW], `energy_price_per_kwh`, TOU pricing config |
| `zones` | Zone names (15 zones) and group definitions (core, perimeter tiers) |
| `weather_files` | Named EPW paths (default, summer, chicago, etc.) |

---

### `src/agents/`

| File | Agent | Action space |
|------|-------|-------------|
| `sac_agent.py` | Soft Actor-Critic | Continuous (used for HVAC setpoint/airflow control) |
| `dqn_agent.py` | Deep Q-Network | Discrete |

SAC config (learning rate, batch size, replay buffer, etc.) lives in `sac_config/sac_config.yaml`.

---

### `src/utils/`

| File | Purpose |
|------|---------|
| `hvac_config.py` | Loads `hvac_config.yaml`; computes state/action sizes; exposes `get_action_bounds()` |
| `idf_modifier.py` | Generates custom IDFs: adjusts `RunPeriod`, injects `Output:Variable` lines for electricity and gas demand rates, sets outdoor CO₂ schedules |
| `energy_price.py` | `get_realtime_price(month, day, hour, config)` — returns $/kWh based on constant, TOU, or real-time config |
| `outdoor_co2_schedule.py` | Loads outdoor CO₂ CSV and builds an 8760-hour lookup for IDF `Schedule:File` injection |
| `weather_csv.py` | Downloads hourly weather from Open-Meteo (free, no key) and provides `WeatherLookup.get(month, day, hour)` → `weather_override` dict |

---

### `energyplus/control_models/`

| File | Description |
|------|-------------|
| `MediumOffice_IAQ.idf` | **Primary model** — DOE Medium Office with CO₂ tracking, Fanger comfort output, VAV system with gas reheat coils. Used by `rl_hvac_control.py`. |
| `MediumOffice_Control.idf` | Medium office configured for external setpoint actuation |

Gas heating note: the model uses `Coil:Heating:Fuel` (NaturalGas), so winter heating load appears in `gas[kW]`, not `elec[kW]`.

---

### `weather/`

| Directory | Contents |
|-----------|---------|
| `chicago/` | TMY3 EPW — default weather used by `rl_hvac_control.py` |
| `atlanta_2023/`, `atlanta_2024/` | Historical hourly EPW files |
| `atlanta_summer_2024/` | Summer-only EPW for cooling-season training |
| `chicago_2020/` | Historical Chicago data |

Download additional EPW files with:
```bash
python download_weather.py --lat 33.75 --lon -84.39 --start 2024-01-01 --end 2024-12-31
```

Or download as CSV for use with `weather_override`:
```python
from src.utils.weather_csv import download_weather_csv, WeatherLookup
download_weather_csv(lat=33.75, lon=-84.39, start="2024-06-01", end="2024-09-30",
                     output_path="data/weather_atlanta_2024.csv")
wx = WeatherLookup("data/weather_atlanta_2024.csv")
controller.env.weather_override = wx.get(month=7, day=15, hour=14)
```

---

### `outputs/rl_hvac_control/`

Auto-created on first run. Main files:

| File | Description |
|------|-------------|
| `rl_hvac_log.csv` | Full timestep log — load into `plot_results.ipynb` |
| `MediumOffice_IAQ_custom_*.idf` | Auto-generated IDF with adjusted run period and injected output variables. Delete this to force regeneration after config changes. |
| `eplusout.err` | Check here first if the simulation crashes |
| `eplustbl.htm` | Open in browser for annual energy summary |

---

## Environment Setup

```bash
conda create -n bem python=3.11
conda activate bem
pip install pyenergyplus-lbnl pandas numpy torch pyyaml
```

The `pyenergyplus-lbnl` package bundles EnergyPlus 23.2 — no separate installation needed.

Set `KMP_DUPLICATE_LIB_OK=TRUE` if running on Windows with PyTorch (already set in `run_rl_hvac.bat`).
