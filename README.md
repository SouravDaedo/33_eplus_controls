# EnergyPlus RL Controls

Reinforcement learning–based HVAC control using the EnergyPlus Python API. Includes a full SAC agent, real-time energy pricing, gas cost tracking, CO₂ monitoring, and programmable weather/CO₂ overrides — all driven from Python with no manual EnergyPlus installation.

## Version Information

Current tested setup, checked on 2026-06-24:

| Component | Version | Notes |
|-----------|---------|-------|
| `pyenergyplus-lbnl` package | `26.1.0` | Latest PyPI version at the time checked |
| EnergyPlus engine used in local simulation | `25.2.0-cf7368216c` | Printed by EnergyPlus when `tests/rl_hvac_control.py` starts |
| Control IDF model version | `23.2` | `energyplus/control_models/MediumOffice_IAQ.idf` declares `Version,23.2;` |
| Generated custom IDF version | `23.2` | Auto-generated under `outputs/rl_hvac_control/`; patched for compatibility with the installed engine |

The source control models in `energyplus/control_models/` are EnergyPlus 23.2 IDFs. When running with newer EnergyPlus engines, `src/utils/idf_modifier.py` applies small compatibility fixes during custom IDF generation. A fully upgraded 25.2 IDF would still require the official EnergyPlus IDFVersionUpdater/Transition tools.

To verify your local versions:

```bash
python -c "from pyenergyplus.api import EnergyPlusAPI; api = EnergyPlusAPI(); print(api.functional.ep_version())"
grep "Version," energyplus/control_models/MediumOffice_IAQ.idf
grep "Version," outputs/rl_hvac_control/MediumOffice_IAQ_custom_412_to_730.idf
```

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
| `generate_outdoor_co2.py` | Generate time-varying outdoor CO₂ CSV (seasonal, rush hour, pollution events) for the RL simulation |
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

**Example: fixed 24-hour simulation**

Use `config/hvac_config_24h.yaml` for a fixed-duration run with the upgraded EnergyPlus 25.2 model:

```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 1
```

Add `--live-plot` to show a running dashboard while the simulation logs timesteps:

```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 1 \
  --live-plot \
  --live-plot-hold
```

The HVAC live plot shows floor-average zone temperatures, effective heating/cooling setpoints, floor-average airflow, electricity/gas power, people/PPD, step reward, average zone CO₂, outside-air CO₂, and outdoor air temperature. Outdoor air temperature is grouped with the CO₂ panel on a second y-axis. A final snapshot is saved as `rl_hvac_live_plot.png` in the output directory.

Add `--loss-plot` during training to open a separate SAC loss dashboard:

```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 3 \
  --training \
  --loss-plot
```

The loss plot tracks actor loss, critic 1 loss, critic 2 loss, and alpha loss. It is controlled separately from `--live-plot`; use both flags if you want both dashboards. A final snapshot is saved as `sac_training_losses.png` in the output directory.

Live plot options:

| Flag | Use |
|------|-----|
| `--live-plot` | Opens the running matplotlib dashboard |
| `--loss-plot` | Opens a separate SAC loss dashboard; only active with `--training` |
| `--live-plot-hold` | Keeps the plot window open after the simulation finishes |
| `--live-plot-every N` | Refreshes the plot every `N` logged timesteps; useful for long runs |
| `--live-plot-scope current` | Default. Resets the plot when a new episode starts, so only the current episode is shown |
| `--live-plot-scope all` | Keeps all episodes on one continuous x-axis |

For multiple episodes with a separate plot per episode:

```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 3 \
  --live-plot \
  --live-plot-scope current \
  --live-plot-hold
```

For multiple episodes on one continuous plot:

```bash
python tests/rl_hvac_control.py \
  --config config/hvac_config_24h.yaml \
  --epw weather/chicago/TMY_lat41.88_lon-87.63.epw \
  --output outputs/rl_hvac_24h \
  --episodes 3 \
  --live-plot \
  --live-plot-scope all \
  --live-plot-hold
```

To choose the start day/hour and duration, edit these fields in `config/hvac_config_24h.yaml`:

```yaml
simulation:
  idf_path: "energyplus/control_models/MediumOffice_IAQ_25_2.idf"
  training_window:
    start_month: 6
    start_day: 6
    start_hour: 13
    end_month: 6
    end_day: 7
    end_hour: 13
  episode_duration_hours: [24, 24]
```

This example runs one 24-hour episode from June 6 at 13:00 through June 7 at 13:00. If the training window is wider than the duration, the controller samples the episode start randomly inside that window.

**Outputs** (written to `--output` directory, default `outputs/rl_hvac_control/`):

| File | Contents |
|------|---------|
| `rl_hvac_log.csv` | Per-timestep log: reward components, temperatures, power [kW], gas [kW], CO₂, actions |
| `rl_hvac_model.pth` | Trained SAC model (only with `--training`) |
| `rl_hvac_live_plot.png` | Final HVAC live plot snapshot (only with `--live-plot`) |
| `sac_training_losses.png` | Final SAC loss plot snapshot (only with `--training --loss-plot`) |
| `eplusout.err` | EnergyPlus warnings and errors |
| `eplustbl.htm` | Annual energy summary table |

---

### `config/hvac_config.yaml` — All Tunable Parameters

| Section | Key settings |
|---------|-------------|
| `simulation` | `idf_path`, `timesteps_per_hour`, `training_window` (date/time range for RL), `episode_duration_hours`, `outdoor_co2_ppm`, `outdoor_co2_csv_path`, `outdoor_co2_fallback_ppm` |
| `action_space` | `sp_offset` bounds, `deadband` bounds, `airflow_multiplier` bounds |
| `reward` | `energy_weight`, `gas_weight`, `gas_price_per_kwh`, `comfort_weight`, `demand_weight`, `demand_threshold` [kW], `energy_price_per_kwh`, TOU pricing config |
| `zones` | Zone names (15 zones) and group definitions (core, perimeter tiers) |
| `weather_files` | Named EPW paths (default, summer, chicago, etc.) |

---

### Outdoor CO₂ schedule (`generate_outdoor_co2.py`)

By default the simulation uses a **flat outdoor CO₂** value (`simulation.outdoor_co2_ppm: 400` in `hvac_config.yaml`). For realistic time-varying outdoor air, generate a CSV first, then point the config at it.

**Workflow:**

1. Generate the CSV (separate step — not auto-created at sim start).
2. Set `outdoor_co2_csv_path` in `config/hvac_config.yaml`.
3. Run `tests/rl_hvac_control.py` as usual.

**What the model includes:**

| Layer | Default | Notes |
|-------|---------|-------|
| Baseline | 420 ppm | `--baseline` |
| Seasonal (Keeling curve) | ±8 ppm | `--seasonal-amplitude` |
| Diurnal | ±3 ppm | Lowest ~14:00; `--diurnal-amplitude` |
| Urban offset | 0 ppm | `--urban-increment` |
| Weekday rush hour | +25–30 ppm peaks | Disable with `--no-rush-hour` |
| One-time events | See below | Custom via `--event` / `--events-file` |
| Noise | σ = 0.5 ppm | `--noise-std` |

**Built-in one-time events** (included unless `--no-events` or `--no-default-events`):

| Event | Start (month/day hour) | Duration | Peak +ppm | Scenario |
|-------|------------------------|----------|-----------|----------|
| `winter_inversion` | Jan 18, 00:00 | 48 h | +70 | Stagnant cold-air pool |
| `industrial_release` | Mar 10, 14:00 | 8 h | +50 | Factory/plant upset |
| `smog_inversion` | Apr 22, 00:00 | 36 h | +80 | Heat-inversion smog |
| `smoke_alert` | May 8, 06:00 | 18 h | +35 | Regional air-quality alert |
| `wildfire_smoke` | Jun 5, 06:00 | 72 h | +150 | Early-summer wildfire |
| `holiday_fireworks` | Jul 4, 22:00 | 4 h | +20 | Evening combustion spike |
| `pollution_spike` | Jul 14, 10:00 | 12 h | +60 | Industrial/traffic spike |
| `wildfire_smoke_2` | Aug 18, 08:00 | 60 h | +120 | Late-summer wildfire |
| `harvest_burn` | Oct 15, 07:00 | 12 h | +45 | Agricultural field burning |

**Traffic jam events** — 20 additional localized backups (`traffic_jam_01` … `traffic_jam_20`), roughly 1–2 per month (morning ~08:00 or evening ~17:00, 4–7 h, +30–44 ppm). Examples: Jan 9 morning (+32), Jun 24 evening (+42), Sep 12 morning (+35), Dec 18 evening (+44). Full list is in `TRAFFIC_JAM_EVENTS` in `generate_outdoor_co2.py`.

Each event is a single Gaussian pulse — it happens **once** on that calendar date in the generated year. Add your own with any label via `--event` or `--events-file`. An example JSON with seven custom events is in `data/co2_events_example.json`.

**Year vs event dates:** Events are specified by **month, day, and hour only** (no year field). The calendar year comes from `--year` when generating the CSV. The output CSV also stores `month, day, hour, ppm` — the simulation looks up outdoor CO₂ by `(month, day, hour)` at runtime, so the pattern applies to whatever year EnergyPlus is simulating.

**Generate a full year:**

```bash
python generate_outdoor_co2.py --year 2023
# -> data/outdoor_co2_2023.csv
```

**Generate only your training window** (match `training_window` in `hvac_config.yaml`):

```bash
python generate_outdoor_co2.py --year 2023 \
  --start-month 4 --start-day 12 \
  --end-month 7 --end-day 30
# -> data/outdoor_co2_2023_0412_to_0730.csv
```

All four range flags are required together; omit them for a full calendar year. Hours outside the CSV use `outdoor_co2_fallback_ppm` when the sim builds the 8760-hour EnergyPlus schedule.

**Custom events** — repeat `--event` (month/day/hour only; year from `--year`):

```bash
python generate_outdoor_co2.py --year 2023 \
  --start-month 4 --start-day 12 --end-month 7 --end-day 30 \
  --no-default-events \
  --event wildfire_1,6,5,6,72,150 \
  --event wildfire_2,7,20,8,48,200
```

Format: `label,month,day,hour,duration_hours,peak_ppm`

**Or load multiple events from JSON** (`--events-file`):

```json
[
  {"label": "wildfire_1", "month": 6, "day": 5, "hour": 6, "duration_hours": 72, "peak_increment": 150},
  {"label": "wildfire_2", "month": 7, "day": 20, "hour": 8, "duration_hours": 48, "peak_increment": 200}
]
```

```bash
python generate_outdoor_co2.py --year 2023 --events-file data/co2_events_example.json --no-default-events
```

Use `data/co2_events_example.json` as a starting template, or copy it to `data/my_co2_events.json` and edit.

**Event flags:**

| Flag | Effect |
|------|--------|
| *(none)* | All built-in one-time events (9 scenarios + 20 traffic jams) + weekday rush hour |
| `--no-events` | No one-time events |
| `--no-default-events` | Skip built-ins; use only `--event` / `--events-file` |
| `--event ...` | Add custom event (repeatable) |
| `--events-file path.json` | Load event list from JSON |

To keep built-ins **and** add your own, pass `--event` without `--no-default-events`.

**Wire into simulation** (`config/hvac_config.yaml`):

```yaml
simulation:
  outdoor_co2_csv_path: "data/outdoor_co2_2023_0412_to_0730.csv"
  outdoor_co2_fallback_ppm: 420
```

With `custom_period.enabled: true`, the custom IDF is patched to use a `Schedule:File` from this CSV. Each timestep, `HVACEnvironment` also sets outdoor CO₂ via the EnergyPlus actuator.

**Flat override at runtime** (bypasses CSV):

```python
controller.env.outdoor_co2_override = 500.0  # ppm
```

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

#### Building Model — `MediumOffice_IAQ.idf`

**Zone layout (18 zones total):**

| Floor | Zones | Notes |
|-------|-------|-------|
| Bottom | `Core_bottom`, `Perimeter_bot_ZN_1–4` | 5 occupied zones |
| Mid | `Core_mid`, `Perimeter_mid_ZN_1–4` | 5 occupied zones |
| Top | `Core_top`, `Perimeter_top_ZN_1–4` | 5 occupied zones |
| Plenum | `TopFloor_Plenum` (+ 2 others) | Unconditioned return plenums — excluded from RL control |

**HVAC system — 3 Packaged Rooftop Units (PACU_VAV):**

One packaged unit per floor: `PACU_VAV_bot`, `PACU_VAV_mid`, `PACU_VAV_top`. Each unit is a self-contained packaged air conditioning unit (not a central chilled-water AHU) with the following air-side components in sequence:

```
OA Mixing Box → CoilSystem:Cooling:DX → Coil:Heating:Fuel (gas) → Fan:VariableVolume
```

Each PACU distributes conditioned air to VAV terminal boxes at its 5 zones. Supply air temperature is managed by `SetpointManager:Warmest` (range 12.8–15.6 °C), which automatically raises the SAT to the warmest value that still satisfies the zone with the highest cooling load. This setpoint can be overridden via the EnergyPlus API using the actuator on `PACU_VAV_<floor> Supply Equipment Outlet Node`.

**Controllable actuators per zone (via EnergyPlus Python API):**

| Actuator | Component type | Control type | Key |
|----------|---------------|--------------|-----|
| Heating setpoint | `Zone Temperature Control` | `Heating Setpoint` | `<zone_name>` |
| Cooling setpoint | `Zone Temperature Control` | `Cooling Setpoint` | `<zone_name>` |
| Infiltration airflow | `Zone Infiltration` | `Air Exchange Flow Rate` | `<zone_name> Infiltration` |
| Supply air temp | `System Node Setpoint` | `Temperature Setpoint` | `PACU_VAV_<floor> Supply Equipment Outlet Node` |

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

The `pyenergyplus-lbnl` package bundles the EnergyPlus engine used by the Python API, so no separate desktop EnergyPlus installation is needed for this workflow. The installed package version and the EnergyPlus engine version are separate values; see "Version Information" above.

Set `KMP_DUPLICATE_LIB_OK=TRUE` if running on Windows with PyTorch (already set in `run_rl_hvac.bat`).
