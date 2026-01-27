# Model Management Examples

This guide shows how to download and upgrade EnergyPlus models to match your installed version.

## Quick Reference

```bash
# Run interactive examples
python example_model_management.py

# Or import and use functions directly
python
>>> from example_model_management import *
>>> example_1_download_version_matched_model()
```

## Understanding Version Compatibility

**Your Setup:**
- Package version: `25.1.2` (what pip reports)
- Engine version: `23.2.0` (what actually runs)
- Model version: Whatever is in the `.idf` file

**The Problem:** If these don't match, simulations may fail or produce incorrect results.

## Example 1: Download Version-Matched Model (RECOMMENDED)

The safest approach - download models that exactly match your engine version.

```python
from example_model_management import download_model_from_specific_version

# Download a model from v23.2.0 (matches your engine)
model_path = download_model_from_specific_version(
    model_name="1ZoneUncontrolled.idf",
    version_tag="v23.2.0",
    output_dir="data"
)
```

**When to use:** Always, if the model exists in your version.

## Example 2: Download and Upgrade

Download a newer model and adjust its version to match your engine.

```python
from example_model_management import (
    download_model_from_specific_version,
    upgrade_idf_version
)

# Download from latest
model_path = download_model_from_specific_version(
    model_name="5ZoneAutoDXVAV.idf",
    version_tag="develop",
    output_dir="data"
)

# Adjust version to match your engine
upgrade_idf_version(model_path, "23.2.0")
```

**When to use:** When you need a model that doesn't exist in your version, or want the latest features.

**Warning:** Only changes the version number. Complex models with new objects may still fail.

## Example 3: Batch Download Multiple Versions

Compare the same model across different EnergyPlus versions.

```python
from example_model_management import download_model_from_specific_version

versions = ["v23.1.0", "v23.2.0", "v24.1.0"]

for version in versions:
    download_model_from_specific_version(
        model_name="1ZoneUncontrolled.idf",
        version_tag=version,
        output_dir=f"data/{version}"
    )
```

**When to use:** Research, testing, or understanding version differences.

## Example 4: Complete Workflow

Production-ready example with all steps.

```python
from example_model_management import (
    download_model_from_specific_version,
    download_weather_file,
    upgrade_idf_version,
    get_idf_version,
    run_simulation
)

# 1. Download model (version-matched)
model_path = download_model_from_specific_version(
    model_name="1ZoneUncontrolled.idf",
    version_tag="v23.2.0",
    output_dir="data"
)

# 2. Download weather
weather_path = download_weather_file(
    weather_name="USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    output_dir="data"
)

# 3. Verify version compatibility
model_version = get_idf_version(model_path)
if model_version != "23.2.0":
    upgrade_idf_version(model_path, "23.2.0")

# 4. Run simulation
run_simulation(model_path, weather_path, "outputs")
```

## Common Scenarios

### Scenario A: "I want to run a specific model"

1. Check what version you need:
   ```bash
   python check_version.py
   ```

2. Download version-matched model:
   ```bash
   python example_model_management.py
   # Select option 1
   ```

### Scenario B: "My model is too new for my engine"

1. Downgrade the model version:
   ```python
   from example_model_management import upgrade_idf_version
   upgrade_idf_version("data/mymodel.idf", "23.2.0")
   ```

2. Or download an older version:
   ```python
   download_model_from_specific_version(
       model_name="mymodel.idf",
       version_tag="v23.2.0"
   )
   ```

### Scenario C: "My model is too old for my engine"

1. Upgrade the model version:
   ```python
   from example_model_management import upgrade_idf_version
   upgrade_idf_version("data/mymodel.idf", "23.2.0")
   ```

2. For complex upgrades, use the full IDFVersionUpdater tool (requires desktop EnergyPlus).

### Scenario D: "I want to test multiple models"

```python
from example_model_management import download_model_from_specific_version

models = [
    "1ZoneUncontrolled.idf",
    "5ZoneAutoDXVAV.idf",
    "RefBldgSmallOfficeNew2004_Chicago.idf"
]

for model in models:
    download_model_from_specific_version(
        model_name=model,
        version_tag="v23.2.0",
        output_dir="data"
    )
```

## Available Version Tags

Check GitHub for available tags:
- `v23.1.0`, `v23.2.0` - Your engine version range
- `v24.1.0`, `v24.2.0` - Newer versions
- `v25.1.0`, `v25.2.0` - Latest releases
- `develop` - Bleeding edge (may be unstable)

Browse models: https://github.com/NREL/EnergyPlus/tree/v23.2.0/testfiles

## Troubleshooting

### "Failed to download (Status 404)"
The model doesn't exist in that version. Try:
- Different version tag
- Check spelling of model name
- Browse GitHub to find available models

### "Simulation failed after upgrade"
The model uses features not available in your version:
- Download from your exact version instead
- Or upgrade your EnergyPlus engine

### "Version mismatch warning"
Your model version doesn't match your engine:
- Use `upgrade_idf_version()` to fix
- Or download version-matched model

## Best Practices

1. **Always match versions** - Download models from your engine's version tag
2. **Create backups** - The upgrade function does this automatically
3. **Test first** - Run simple models (like 1ZoneUncontrolled) before complex ones
4. **Check error logs** - Always review `outputs/eplusout.err` after simulation
5. **Use version control** - Keep your models in git to track changes

## Next Steps

- Run the interactive examples: `python example_model_management.py`
- Check your version: `python check_version.py`
- Browse available models: https://github.com/NREL/EnergyPlus/tree/v23.2.0/testfiles
- Read the main README: `README.md`
