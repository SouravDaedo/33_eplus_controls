# EnergyPlus Python-Only Controls

A complete Python-only EnergyPlus simulation environment that requires **no manual EnergyPlus installation**. Everything runs through pip-installed packages with automatic version-matched model downloads.

## Features

- ✅ **No manual EnergyPlus install required** - Everything via `pip`
- ✅ **Automatic version matching** - Downloads models that match your installed EnergyPlus version
- ✅ **Batch processing** - Run multiple building models automatically
- ✅ **Version checking** - Verify your installation and compatibility

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pyenergyplus` - The EnergyPlus simulation engine
- `requests` - For downloading model files
- `pandas` - For analyzing results (optional)

### 2. Run Your First Simulation

```bash
python run_simulation.py
```

This will:
1. Detect your installed EnergyPlus version
2. Download a simple test model (`1ZoneUncontrolled.idf`) from GitHub
3. Download matching weather data (Chicago)
4. Run the simulation
5. Generate results in the `outputs/` directory

### 3. Check Your Installation

```bash
python check_version.py
```

This displays:
- Installed EnergyPlus version
- Package location
- API status

## Project Structure

```
33_eplus_controls/
├── data/                    # Downloaded models and weather files
├── outputs/                 # Simulation results (single runs)
├── batch_outputs/           # Batch simulation results
├── run_simulation.py        # Main simulation runner
├── batch_runner.py          # Run multiple models
├── check_version.py         # Version checker
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Scripts

### `run_simulation.py`
Runs a single simulation with automatic version-matched file downloads.

**What it does:**
- Detects your EnergyPlus version
- Downloads `1ZoneUncontrolled.idf` (simple test model)
- Downloads Chicago weather file
- Runs simulation
- Lists output files

**Output files:**
- `outputs/eplustbl.htm` - Summary report (open in browser)
- `outputs/eplusout.csv` - Detailed timeseries data
- `outputs/eplusout.err` - Warnings and errors

### `batch_runner.py`
Runs multiple building models in sequence.

**What it does:**
- Downloads 3 different building models
- Runs each with the same weather file
- Saves results to separate directories
- Provides summary of successes/failures

**Models tested:**
1. `1ZoneUncontrolled.idf` - Simple box
2. `5ZoneAutoDXVAV.idf` - Commercial building with HVAC
3. `RefBldgSmallOfficeNew2004_Chicago.idf` - DOE reference building

```bash
python batch_runner.py
```

### `check_version.py`
Displays your EnergyPlus installation details.

```bash
python check_version.py
```

## Version Management

### Check Current Version
```bash
pip show pyenergyplus
```

### Install Specific Version
```bash
pip install pyenergyplus==24.1.0
```

### Upgrade to Latest
```bash
pip install --upgrade pyenergyplus
```

### Version Compatibility
The scripts automatically download models that match your installed version. EnergyPlus is strict about version compatibility - a v24.1 model may not work with v25.1 without transition.

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
