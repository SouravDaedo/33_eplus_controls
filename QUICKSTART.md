# Quick Start Guide

## 1. Install EnergyPlus (Python-Only Method)

```bash
pip install -r requirements.txt
```

That's it! No need to download the 500MB+ EnergyPlus installer.

## 2. Verify Installation

```bash
python check_version.py
```

Expected output:
```
EnergyPlus Python Library Version: 24.1.0
Package location: C:\...\site-packages\pyenergyplus\...
API available: Yes
```

## 3. Run Your First Simulation

```bash
python run_simulation.py
```

This will:
- Auto-detect your version
- Download a test model (1ZoneUncontrolled.idf)
- Download Chicago weather data
- Run the simulation (~10 seconds)
- Create `outputs/` folder with results

## 4. View Results

Open in your browser:
```
outputs/eplustbl.htm
```

This shows annual energy consumption, peak loads, and comfort metrics.

## 5. Run Multiple Models (Optional)

```bash
python batch_runner.py
```

Tests 3 different building types and saves results to `batch_outputs/`.

## What Models Should I Use?

### For Testing
- `1ZoneUncontrolled.idf` - Simplest (20m box, no HVAC)
- `5ZoneAutoDXVAV.idf` - Commercial building with HVAC

### For Real Projects
- DOE Reference Buildings: `RefBldgSmallOfficeNew2004_Chicago.idf`
- Custom models: Create in OpenStudio or IDF Editor

## Common First-Time Issues

### "Module not found: pyenergyplus"
```bash
pip install pyenergyplus
```

### "Failed to download model"
Your version might be too new. The scripts try to download from GitHub using your version tag. If the tag doesn't exist yet:
1. Check available versions: `pip install pyenergyplus==`
2. Install a stable version: `pip install pyenergyplus==24.1.0`

### "Simulation failed"
Check `outputs/eplusout.err` for the error message. Common causes:
- Version mismatch between model and engine
- Missing weather file
- Invalid model syntax

## Next Steps

1. **Analyze results**: Load `outputs/eplusout.csv` with pandas
2. **Modify models**: Use `eppy` library to edit IDF files
3. **Run parametric studies**: Loop through different design options
4. **Add controls**: Use the EnergyPlus Python Plugin API

## File Locations

After running simulations:
```
33_eplus_controls/
├── data/
│   ├── 1ZoneUncontrolled.idf          # Downloaded model
│   └── USA_IL_Chicago-OHare...epw     # Downloaded weather
├── outputs/
│   ├── eplustbl.htm                   # ← Open this first
│   ├── eplusout.csv                   # Timeseries data
│   └── eplusout.err                   # Error log
└── batch_outputs/
    ├── 1ZoneUncontrolled/
    ├── 5ZoneAutoDXVAV/
    └── RefBldgSmallOfficeNew2004_Chicago/
```

## Getting Help

- Check `README.md` for detailed documentation
- Review `outputs/eplusout.err` for simulation errors
- Visit [EnergyPlus Documentation](https://energyplus.net/documentation)
