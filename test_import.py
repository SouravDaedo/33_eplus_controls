try:
    import pyenergyplus
    print("pyenergyplus found")
    from pyenergyplus.api import EnergyPlusAPI
    print("EnergyPlusAPI imported successfully")
except ImportError as e:
    print(f"Error: {e}")
