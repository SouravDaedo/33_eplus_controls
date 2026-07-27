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
| AHU outdoor-air mass flow | `airflow_multiplier` | 0.0 to 6.0 kg/s | Commands the outdoor-air controller mass flow for each floor PACU |

**Reward components** (all configurable in `hvac_config.yaml`):

| Component | Description |
|-----------|-------------|
| `elec_cost` | Electricity cost: `kWh × RTP` ($) |
| `gas_cost` | Gas cost: `kWh_thermal × gas_price_per_kwh` ($) |
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
| `action_space` | `sp_offset` bounds, `deadband` bounds, `airflow_multiplier` bounds for AHU outdoor-air controller mass flow |
| `reward` | Dollar costs: RTP energy, optional gas; PPD→productivity and CO₂→productivity; `cost_normalization` (`absolute` \| `per_m2`); see **Reward function** below |

---

### Reward function (energy + productivity costs)

Paper-ready equations (LaTeX): [`docs/reward_formulation.md`](docs/reward_formulation.md).

The RL reward is the negative of a multi-term **cost**. With the current defaults, the main terms are all in **dollars** (or optionally **$/m²**) so energy, thermal comfort, and IAQ are on one scale:

\[
\text{reward} = -(\text{elec \$} + \text{gas \$} + \text{thermal productivity \$} + \text{IAQ productivity \$})
\]

Shared monetization of a fractional productivity loss (engineering cost-benefit step, not prescribed by the papers):

\[
\$ = \text{loss\_frac} \times \text{people} \times \texttt{labor\_cost\_per\_person\_hour} \times \Delta t
\]

People counts are used **only in the reward** (from EnergyPlus). They are not required in the policy observation (floor CO₂ is the transferable IAQ signal).

#### Scale discussion: productivity vs energy (the multi-objective problem)

Putting energy, thermal, and IAQ on a common `$` axis does **not** by itself make them equally important to the RL agent. Energy is priced by the market (RTP). Productivity is priced by **literature fractional losses × occupancy × wage**. Those two pipelines were never fitted to each other, so the default reward can be **strongly lopsided toward comfort/IAQ** whenever the building is occupied and conditions leave the reference band.

##### What the papers report (fractional productivity loss)

Representative **loss fractions** from the sources used above (not our default slopes—raw reported magnitudes):

| Source | Condition (summary) | Reported / cited loss |
|--------|---------------------|------------------------|
| **Lan, Wargocki & Lian (2011)** / REHVA summaries | Mild thermal-band effects (conservative Cat. III–style reading) | ~**0.5%** |
| **Kosonen & Tan (2004)** | PMV ≈ **+0.5** (warm edge of typical comfort), **thinking** tasks | ~**12%** |
| **Kosonen & Tan (2004)** | Same PMV ≈ **+0.5**, **typing** tasks | ~**26%** |
| **Seppänen, Fisk & Lei (2006)** | ~**1–3%** performance change per **+10 L/s·person** outdoor air (office work meta-analysis) | **1%, 2%, 3%** |
| **CO₂ / ventilation proxies** (Wargocki-type summaries used in engineering models) | Roughly **~1% per +100 ppm** CO₂ above a good-IAQ reference (task-dependent; uncertain) | e.g. **1–4%** for +100–400 ppm |

Take **medians of these reported fractions** (simple, transparent summary of the literature spread—not a meta-analysis):

| Domain | Values used | **Median loss_frac** |
|--------|-------------|----------------------|
| Thermal | 0.5%, 12%, 26% | **12%** |
| IAQ / ventilation | 1%, 2%, 3% | **2%** |
| CO₂ proxy | 1%, 2%, 4% | **2%** |

##### Scale by number of people in *this* building

DOE Medium Office here: `floor_area_m2 ≈ 4982`, IDF `Floor Area per Person ≈ 18.58 m²/person` → design occupancy

\[
N_{\text{design}} \approx 4982 / 18.58 \approx \mathbf{268\ people}
\]

(Actual timestep people follow the occupancy schedule; peaks approach this order of magnitude.)

With default monetization `labor_cost_per_person_hour = $40` and \(\Delta t = 0.25\) h (15 min):

\[
\$_{\text{step}} = \text{loss\_frac} \times N \times 40 \times 0.25 = \text{loss\_frac} \times N \times 10
\]

At **design occupancy**, paper-median losses become:

| Term | Median loss | Absolute `$` / 15‑min step | `$/m²` / step (`÷ 4982`) |
|------|-------------|----------------------------|---------------------------|
| Thermal (median **12%**) | 0.12 | **~$322** | ~$0.065 |
| IAQ vent / CO₂ proxy (median **2%**) | 0.02 | **~$54** | ~$0.011 |

##### Compare to median energy cost

For the training RTP file `data/rtp_prices_2023_0412_to_0730.csv`, **median price ≈ $0.101 / kWh**. Electric cost per step is `kW × 0.25 × RTP`:

| Electric load | `$` / step at median RTP | `$/m²` / step |
|---------------|--------------------------|---------------|
| 30 kW | ~$0.76 | ~$0.00015 |
| **50 kW** (illustrative mid load) | **~$1.27** | ~$0.00025 |
| 80 kW | ~$2.03 | ~$0.00041 |

**Median-to-median ratio** (design occupancy, wage $40/h, vs ~50 kW at median RTP):

| Productivity term | ≈ multiple of electric `$` |
|-------------------|----------------------------|
| Thermal (12%) | **~250×** |
| IAQ / CO₂ (2%) | **~40×** |

So: if you literally apply paper-scale fractional losses to **all occupants** at a full wage, **labor-productivity `$` dominates energy `$` by 1–2 orders of magnitude**. That is economically unsurprising (labor ≫ HVAC energy in offices) but it is a **problem for multi-objective RL**: the policy will almost always trade energy for tiny comfort/IAQ gains.

`cost_normalization: per_m2` does **not** fix this—it divides every `$` term by the same area, so **ratios stay the same**.

##### Implications for this control problem

1. **Literal “full wage × paper loss”** encodes a building-economics objective (maximize labor output), not a balanced HVAC tradeoff.
2. **Default config slopes** (`ppd_productivity_loss_per_percent = 0.5`, `co2_productivity_loss_per_100ppm = 0.01`) are **order-of-magnitude proxies** inspired by the same literature; they are **not** calibrated so that median productivity `$` ≈ median energy `$`.
3. For RL training where energy *and* comfort/IAQ should both matter, **predetermine a target mix** and scale productivity down, e.g.:
   - `comfort_weight` / `co2_weight` (direct), or
   - lower `labor_cost_per_person_hour`, or
   - softer loss slopes.
4. A transparent balancing rule: choose weights so that **at the paper-median loss and typical occupied people / typical kW**, productivity `$` and electric `$` are within a chosen factor (e.g. 1×–3×). Example (design `N`, 50 kW, median RTP):  
   `comfort_weight ≈ 1.27/322 ≈ 0.004`, `co2_weight ≈ 1.27/54 ≈ 0.024` to match medians one-to-one—then tune from live KPI plots.

##### Practical reading of the live cost panel

- Unoccupied (`people ≈ 0`): productivity `$` ≈ 0 → reward is energy-driven.
- Occupied + PPD/CO₂ above references: without down-weighting, thermal/IAQ traces will dwarf electric—**expected under full-wage monetization**, not a pricing bug.

#### Adaptive balancing (online PPD / CO₂ scales)

Instead of fixing `comfort_weight` / `co2_weight` by hand, you can let training data set **variable scales** that update as more timesteps arrive:

```yaml
reward:
  adaptive_balancing:
    enabled: true
    ema_alpha: 0.01
    min_samples: 100
    comfort_target_ratio: 1.0   # aim EMA(scaled comfort) ≈ 1× EMA(energy+gas)
    co2_target_ratio: 1.0
    weight_min: 0.0
    weight_max: 1.0             # only scale productivity *down*
    initial_scale: 1.0
```

Each step (before `/m²` normalization):

1. Compute raw thermal / IAQ `$` (literature × people × wage × Δt).
2. Update EMAs of `energy+gas`, `comfort_raw`, `co2_raw`.
3. After `min_samples`:
   - `adaptive_comfort_scale = clip(comfort_target_ratio × EMA_E / EMA_C, weight_min, weight_max)`
   - same for CO₂.
4. Final term: `raw × static_weight × adaptive_scale`.

Logged columns: `adaptive_comfort_scale`, `adaptive_co2_scale`, `adaptive_balancing_active`, `adaptive_n_samples`.

With `weight_max: 1.0`, productivity is never amplified above the static weights—only reduced when its running mean exceeds the energy mean. Set `comfort_target_ratio: 0.5` if you want thermal to contribute about half of energy on average.

#### Cost normalization

```yaml
reward:
  cost_normalization:
    mode: "absolute"       # or "per_m2"
    floor_area_m2: 4982.0  # required when mode is per_m2 (DOE Medium Office ≈ 53,628 ft²)
```

| `mode` | Meaning |
|--------|---------|
| `absolute` | Building-total `$` per timestep (default) |
| `per_m2` | Divide energy / gas / thermal / IAQ `$` by `floor_area_m2` → `$/m²` (better for transfer across building sizes) |

Demand and setpoint penalties default to weight `0` under RTP (no separate demand charge in the modeled tariff).

#### Thermal: PPD → productivity → $

Default model: `thermal_comfort_model: "ppd_productivity"`.

\[
\text{loss\_frac} = \mathrm{clip}\big((PPD - \texttt{ppd\_reference}) \times \texttt{ppd\_productivity\_loss\_per\_percent} / 100,\; 0,\; \texttt{ppd\_max\_productivity\_loss}\big)
\]

Defaults: reference **10% PPD**, slope **0.5% productivity per 1% PPD** above reference, cap **15%**.

| Source | What it says | How it is used here |
|--------|----------------|---------------------|
| **ASHRAE Standard 55** | Comfort is often designed around roughly **≤10% PPD** (typical criterion, not a productivity law). | Sets **`ppd_reference = 10`**: no productivity penalty until PPD exceeds that band. |
| **Kosonen & Tan** (PMV/PPD–productivity) | Link **thermal dissatisfaction (PPD)** to **work performance / productivity loss** for economic HVAC assessment. | Justifies using **PPD** (not raw °C error) as the discomfort metric. |
| **Lan, Wargocki & Lian (2011)**, *Energy and Buildings* | Quantify that **thermal discomfort reduces productivity**; support monetizing discomfort. | Supports turning discomfort into a **% productivity loss**, then `$`. We do **not** copy their exact lab curve; we use a simple linear slope above 10% PPD. |
| **Seppänen / Fisk** (temperature–performance) | Task performance varies with **temperature**; effects often on the order of **a few percent** over realistic ranges. | Background that thermal conditions affect performance. EnergyPlus **Fanger PPD** is used as the comfort proxy. |

The slope `ppd_productivity_loss_per_percent` is a **tunable linear proxy**, not a universal constant from one paper.

#### IAQ: CO₂ → productivity → $

Default model: `co2_model: "productivity"` (floor-average CO₂, occupant-weighted per floor).

\[
\text{loss\_frac} = \mathrm{clip}\big((CO_2 - \texttt{co2\_reference\_ppm}) / 100 \times \texttt{co2\_productivity\_loss\_per\_100ppm},\; 0,\; \texttt{co2\_max\_productivity\_loss}\big)
\]

Defaults: reference **800 ppm**, slope **~1% productivity per +100 ppm**, cap **15%**.

| Source | What it says | How it is used here |
|--------|----------------|---------------------|
| **Seppänen, Fisk & Lei (2006)**, *Indoor Air* | Higher **outdoor-air ventilation** is associated with better **office work performance** (often ~**1–3%** over practical L/s·person ranges). CO₂ is a common **ventilation/IAQ proxy**. | Supports: worse ventilation/IAQ → **small % performance loss**. **Floor CO₂** is the measurable signal. |
| **Wargocki et al. (2000)**, *Indoor Air* | Changing **outdoor air supply** affects perceived air quality, SBS symptoms, and **productivity** on simulated tasks (typically a **few percent**). | Same idea; we monetize IAQ via CO₂ rather than OA L/s·person in the reward. |
| **Wargocki et al.** CO₂–performance analyses | Performance changes are often summarized as a **fractional change per ~100 ppm CO₂** (task-dependent; sometimes ~**1%/100 ppm** in pooled fits, with uncertainty). | Motivates default `co2_productivity_loss_per_100ppm: 0.01` and a reference near **800 ppm**. |
| **Allen et al. (2016)** COGfx | Controlled exposures show **cognitive scores** can drop substantially as CO₂ rises / ventilation worsens (effects can be **larger** than mild office-task % changes). | Supports that elevated CO₂ is cognitively costly. Defaults stay **conservative** (capped 15%, ~1%/100 ppm) rather than using the largest COGfx magnitudes. |

Again, **800 ppm / 1%/100 ppm** is an **order-of-magnitude proxy**, not “the” Seppänen or Allen equation.

#### Related config keys

| Key | Role |
|-----|------|
| RTP `realtime_price`, `gas_price_per_kwh` | Electric/gas `$` = kWh × price (no extra weights) |
| `labor_cost_per_person_hour` | Wage used to monetize thermal/IAQ loss |
| `adaptive_balancing` | Online scales so EMA(PPD/CO₂ $) track EMA(energy $); see above |
| `thermal_comfort_model` | `ppd_productivity` \| `ppd` \| `temperature` |
| `co2_model` | `productivity` \| `threshold` |
| `cost_normalization` | `absolute` \| `per_m2` |

Full comments and defaults live in `config/hvac_config.yaml` under `reward`.

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
| `winter_inversion` | Jan 18, 00:00 | 48 h | +200 | Stagnant cold-air pool |
| `industrial_release` | Mar 10, 14:00 | 8 h | +140 | Factory/plant upset |
| `smog_inversion` | Apr 22, 00:00 | 36 h | +220 | Heat-inversion smog |
| `smoke_alert` | May 8, 06:00 | 18 h | +100 | Regional air-quality alert |
| `wildfire_smoke` | Jun 5, 06:00 | 72 h | +350 | Early-summer wildfire |
| `holiday_fireworks` | Jul 4, 22:00 | 4 h | +60 | Evening combustion spike |
| `pollution_spike` | Jul 14, 10:00 | 12 h | +170 | Industrial/traffic spike |
| `wildfire_smoke_2` | Aug 18, 08:00 | 60 h | +300 | Late-summer wildfire |
| `harvest_burn` | Oct 15, 07:00 | 12 h | +125 | Agricultural field burning |

**Traffic jam events** — 20 additional localized backups (`traffic_jam_01` … `traffic_jam_20`), roughly 1–2 per month (morning ~08:00 or evening ~17:00, 4–7 h, **+85–125 ppm**). Full list is in `TRAFFIC_JAM_EVENTS` in `generate_outdoor_co2.py`. Weekday rush hour peaks are **+85 ppm** (morning) and **+70 ppm** (evening).

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

**Controllable actuators via EnergyPlus Python API:**

| Actuator | Component type | Control type | Key |
|----------|---------------|--------------|-----|
| Heating setpoint | `Zone Temperature Control` | `Heating Setpoint` | `<zone_name>` |
| Cooling setpoint | `Zone Temperature Control` | `Cooling Setpoint` | `<zone_name>` |
| AHU outdoor-air mass flow | `Outdoor Air Controller` | `Air Mass Flow Rate` | `PACU_VAV_BOT_OA_CONTROLLER`, `PACU_VAV_MID_OA_CONTROLLER`, `PACU_VAV_TOP_OA_CONTROLLER` |
| Supply air temp | `System Node Setpoint` | `Temperature Setpoint` | `PACU_VAV_<floor> Supply Equipment Outlet Node` |

The same RL outdoor-air action is sent to all three PACU outdoor-air controllers in kg/s. The live airflow plots report the resulting `Air System Outdoor Air Mass Flow Rate` for `PACU_VAV_bot`, `PACU_VAV_mid`, and `PACU_VAV_top` (actual) plus the commanded OA overlay. To make commanded OA trackable in unoccupied hours, `create_custom_idf` forces `HVACOperationSchd` and `MinOA_MotorizedDamper_Sched` to always-on (stock fans/OA are otherwise off overnight). The economizer-scale upper bound comes from EnergyPlus autosizing: the OA controller maximums are `4.40`, `4.26`, and `4.94 m3/s`, so the largest controller is roughly `4.94 * 1.2 = 5.9 kg/s`.

**Why commanded OA often differs from actual OA**

The RL actuator request is accepted, but EnergyPlus still enforces:

\[
\dot{m}_{\text{OA, actual}} \le \dot{m}_{\text{mixed / supply}}
\]

From the EnergyPlus EMS Application Guide (*Outdoor Air Controller*): the actuated mass flow rate is not allowed to be greater than the current system mixed air flow rate. If the override exceeds mixed-air flow, OA is clipped to the mixed-air rate (limiting factor: mixed air) — the command is not ignored.

On these VAV PACUs that means:

1. Zone load falls → fan / supply airflow turns down.
2. The agent may still command 4–6 kg/s OA.
3. Actual OA is capped near the current supply mass flow (often ~1–2 kg/s in light-load periods).
4. When load (and supply) briefly rises, actual OA can spike toward the command.

That afternoon gap on the live plot is therefore **supply-flow limiting**, not the occupied/unoccupied schedule issue. The always-on schedule rewrite only fixes overnight “fans off → actual OA ≈ 0.”

Secondary caps: autosized maximum OA (~5–6 kg/s) and max OA fraction × supply flow (stock max-fraction schedules are 1.0 after EMS cleanup). The same command is applied to all three floors, so the three actual traces look similar.

How to read the plot:

| Observation | Interpretation |
|-------------|----------------|
| Commanded ≫ actual (afternoon) | Request above current supply flow; AHU is likely near **100% OA** of whatever supply exists |
| Actual tracks commanded | Supply flow ≥ command (or command is low) |
| Both ~0 overnight (before always-on patch) | Fans / OA availability off |

**Practical takeaway:** commanded OA is a request, not a guarantee of that mass flow. On VAV systems, a high OA command mainly raises the outdoor-air fraction up to 1.0 of the current supply flow.

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
