# EnergyPlus Python-Only Controls

A complete Python-only EnergyPlus simulation environment that requires **no manual EnergyPlus installation**. Everything runs through pip-installed packages.

## Features

- ✅ **No manual EnergyPlus install required** - Everything via `pip`
- ✅ **Generic simulation runner** - Run any IDF with any weather file
- ✅ **Weather data downloader** - Download from Open-Meteo (2024/2025) or PVGIS (TMY)
- ✅ **Automatic IDF sync** - Update IDF run periods to match weather data
- ✅ **Batch processing** - Run multiple building models automatically

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Simulation

```bash
python run_sim.py --idf energyplus/models/RefBldgMediumOfficeNew2004_Chicago.idf --epw weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw --output outputs/test
```

### 3. Download Weather Data

```bash
# Download 2024 summer data for Atlanta
python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30 --output weather/atlanta

# Download TMY data
python download_weather.py --lat 41.88 --lon -87.63 --tmy --output weather/chicago
```

## Model Versions

| Model | EnergyPlus Version | Compatible |
|-------|-------------------|------------|
| RefBldgMediumOfficeNew2004_Chicago | 23.2 | ✓ Match |
| RefBldgLargeHotelNew2004_Chicago | 23.2 | ✓ Match |
| RefBldgPrimarySchoolNew2004_Chicago | 23.2 | ✓ Match |
| ASHRAE901_OfficeMedium_STD2022_Denver | 22.1 | ⚠️ Older |
| ASHRAE901_OfficeLarge_STD2022_Denver | 25.2 | ❌ Too new |
| ASHRAE901_SchoolPrimary_STD2022_Denver | 22.1 | ⚠️ Older |

## Project Structure

```
33_eplus_controls/
├── energyplus/models/       # All IDF model files
│   ├── RefBldgMediumOfficeNew2004_Chicago.idf
│   ├── RefBldgLargeHotelNew2004_Chicago.idf
│   ├── RefBldgPrimarySchoolNew2004_Chicago.idf
│   ├── ASHRAE901_OfficeMedium_STD2022_Denver.idf
│   └── ...
├── weather/                 # Weather files (EPW)
├── outputs/                 # Simulation results
├── run_sim.py              # Generic simulation runner
├── download_weather.py     # Weather data downloader
├── check_engine.py         # Engine version checker
├── batch_runner.py         # Batch simulation runner
├── analyze_annual_results.py
├── requirements.txt
└── README.md
```

## Scripts

### `run_sim.py`
Generic EnergyPlus simulation runner.

```bash
# Basic usage
python run_sim.py --idf model.idf --epw weather.epw

# With custom output directory
python run_sim.py --idf model.idf --epw weather.epw --output results/
```

### `download_weather.py`
Download weather data from Open-Meteo (global, 2024/2025) or PVGIS (TMY).

```bash
# Download 2024 data (Open-Meteo)
python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30

# Download TMY (PVGIS)
python download_weather.py --lat 41.88 --lon -87.63 --tmy

# Download and update IDF run period
python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30 --update-idf model.idf

# Sync existing EPW to IDF
python download_weather.py --sync-epw weather.epw --update-idf model.idf
```

**Weather Data Sources:**
| Source | Coverage | Years | Best For |
|--------|----------|-------|----------|
| Open-Meteo | Global | 1940-present (5-day delay) | 2024/2025, specific dates |
| PVGIS | Europe/Africa/Med | Up to 2023 | TMY data |

### `batch_runner.py`
Run multiple building models in sequence.

```bash
python batch_runner.py
```

### `check_engine.py`
Check EnergyPlus engine version and model compatibility.

```bash
# Check engine version
python check_engine.py

# Check engine + model compatibility
python check_engine.py --check-models
```

**Example output:**
```
============================================================
ENERGYPLUS ENGINE VERSION CHECK
============================================================

Pip Package: pyenergyplus-lbnl 25.1.2

Engine Version: EnergyPlus 23.2.0-7636e6b3e9

⚠️  Note: Pip version (25.1.2) differs from engine version (23.2.0)
============================================================

MODEL COMPATIBILITY CHECK
============================================================

Engine: 23.2.0
------------------------------------------------------------
Model                                              Version  Status
------------------------------------------------------------
RefBldgMediumOfficeNew2004_Chicago.idf             23.2     ✓ Match
ASHRAE901_OfficeLarge_STD2022_Denver.idf           25.2     ❌ Too new
ASHRAE901_OfficeMedium_STD2022_Denver.idf          22.1     ⚠️ Older (may work)
------------------------------------------------------------
```

## Version Management

### Check Engine Version
```bash
python check_engine.py
```

### Check Pip Package
```bash
pip show pyenergyplus-lbnl
```

### Version Compatibility
EnergyPlus is strict about version compatibility:
- **✓ Match** - Model version equals engine version
- **⚠️ Older** - Older models usually work with newer engines
- **❌ Too new** - Newer models will NOT work with older engines

## Understanding Output Files

After running a simulation, check the `outputs/` directory:

| File | Description |
|------|-------------|
| `eplustbl.htm` | **Start here** - Summary report with annual energy use |
| `eplusout.csv` | Hourly/sub-hourly data (temperatures, loads, etc.) |
| `eplusout.err` | Error log - check if simulation had issues |
| `eplusout.eso` | Raw output (use for custom post-processing) |
| `eplusout.mtr` | Meter data (energy consumption by end-use) |

## Analyzing Results with Python

```python
import pandas as pd

# Load timeseries data
df = pd.read_csv('outputs/eplusout.csv')

# Display available columns
print(df.columns)

# Plot zone temperature
df['Zone Mean Air Temperature'].plot()
```

## Common Issues

### "Failed to download" error
The model may not exist for your EnergyPlus version. Try:
1. Check your version: `python check_version.py`
2. Use a different model name
3. Manually download from [EnergyPlus GitHub](https://github.com/NREL/EnergyPlus)

### Version mismatch error
Your `.idf` file version doesn't match your installed library. Either:
- Install the matching version: `pip install pyenergyplus==24.1.0`
- Or use transition tools to upgrade the model

### Simulation fails
Check `outputs/eplusout.err` for detailed error messages.

## Where to Find More Models

### Official EnergyPlus Examples
Browse the [testfiles directory](https://github.com/NREL/EnergyPlus/tree/develop/testfiles) on GitHub. Remember to switch to your version tag (e.g., `v24.1.0`).

### DOE Reference Buildings
Standard commercial building models:
- `RefBldgSmallOfficeNew2004_Chicago.idf`
- `RefBldgMediumOfficeNew2004_Chicago.idf`
- `RefBldgLargeOfficeNew2004_Chicago.idf`

### Weather Files
Download from [Climate.OneBuilding.org](https://climate.onebuilding.org/) for any location worldwide.

## Advanced Usage

### Running Custom Models
```python
from pyenergyplus.api import EnergyPlusAPI

api = EnergyPlusAPI()
state = api.state_manager.new_state()

api.runtime.run_energyplus(state, [
    '-d', 'my_outputs',
    '-w', 'path/to/weather.epw',
    'path/to/my_model.idf'
])

api.state_manager.delete_state(state)
```

### Modifying Models Programmatically
For editing `.idf` files with Python, install `eppy`:
```bash
pip install eppy
```

For modern `.epjson` files, use Python's built-in `json` library.

## Resources

- [EnergyPlus Documentation](https://energyplus.net/documentation)
- [EnergyPlus GitHub](https://github.com/NREL/EnergyPlus)
- [Python API Guide](https://energyplus.net/documentation)
- [Weather Data](https://climate.onebuilding.org/)

## License

This repository structure is provided as-is. EnergyPlus itself is licensed by the U.S. Department of Energy.
