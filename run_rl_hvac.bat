@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd tests
call conda activate bem
python -c "import sys; sys.path.insert(0, '.'); from pyenergyplus.api import EnergyPlusAPI; print('Import successful')"
python rl_hvac_control.py --config ../config/hvac_config.yaml
cd ..
