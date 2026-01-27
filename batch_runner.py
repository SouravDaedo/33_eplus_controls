import os
import sys
import requests
import importlib.metadata
from pyenergyplus.api import EnergyPlusAPI

DATA_DIR = "data"
OUTPUTS_BASE = "batch_outputs"


def get_energyplus_version():
    """Get the installed EnergyPlus version."""
    try:
        try:
            return importlib.metadata.version('pyenergyplus-lbnl')
        except importlib.metadata.PackageNotFoundError:
            return importlib.metadata.version('pyenergyplus')
    except Exception as e:
        print(f"Error detecting version: {e}")
        sys.exit(1)


def download_model(model_name, version):
    """Download a specific model file from GitHub."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = os.path.join(DATA_DIR, model_name)
    
    if os.path.exists(filepath):
        print(f"Model already exists: {model_name}")
        return filepath
    
    # Try multiple version tag formats
    version_tags = [
        f"v{version}",
        f"v{version.rsplit('.', 1)[0]}",
        "develop"
    ]
    
    print(f"Downloading {model_name}...")
    for v_tag in version_tags:
        url = f"https://raw.githubusercontent.com/NREL/EnergyPlus/{v_tag}/testfiles/{model_name}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded from {v_tag}: {model_name}")
                return filepath
        except Exception:
            continue
    
    print(f"Failed to download {model_name} from any version tag")
    return None


def download_weather(weather_name, version):
    """Download a weather file from GitHub."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = os.path.join(DATA_DIR, weather_name)
    
    if os.path.exists(filepath):
        print(f"Weather file already exists: {weather_name}")
        return filepath
    
    # Try multiple version tag formats
    version_tags = [
        f"v{version}",
        f"v{version.rsplit('.', 1)[0]}",
        "develop"
    ]
    
    print(f"Downloading {weather_name}...")
    for v_tag in version_tags:
        url = f"https://raw.githubusercontent.com/NREL/EnergyPlus/{v_tag}/weather/{weather_name}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded from {v_tag}: {weather_name}")
                return filepath
        except Exception:
            continue
    
    print(f"Failed to download {weather_name} from any version tag")
    return None


def run_single_simulation(model_path, weather_path, output_dir):
    """Run a single EnergyPlus simulation."""
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    
    os.makedirs(output_dir, exist_ok=True)
    
    args = [
        '-d', output_dir,
        '-w', weather_path,
        model_path
    ]
    
    try:
        print(f"Running simulation: {os.path.basename(model_path)}")
        api.runtime.run_energyplus(state, args)
        print(f"Completed: {os.path.basename(model_path)}")
        return True
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False
    finally:
        api.state_manager.delete_state(state)


def main():
    print("="*60)
    print("EnergyPlus Batch Runner")
    print("="*60 + "\n")
    
    version = get_energyplus_version()
    print(f"EnergyPlus version: {version}\n")
    
    models_to_run = [
        "1ZoneUncontrolled.idf",
        "5ZoneAutoDXVAV.idf",
        "RefBldgSmallOfficeNew2004_Chicago.idf"
    ]
    
    weather_file = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    
    print("Step 1: Downloading weather file...")
    weather_path = download_weather(weather_file, version)
    if not weather_path:
        print("Failed to download weather file. Exiting.")
        sys.exit(1)
    
    print("\nStep 2: Downloading and running models...")
    results = []
    
    for model_name in models_to_run:
        print(f"\n{'-'*60}")
        print(f"Processing: {model_name}")
        print('-'*60)
        
        model_path = download_model(model_name, version)
        if not model_path:
            print(f"Skipping {model_name} - download failed")
            results.append((model_name, False))
            continue
        
        output_dir = os.path.join(OUTPUTS_BASE, model_name.replace('.idf', ''))
        success = run_single_simulation(model_path, weather_path, output_dir)
        results.append((model_name, success))
    
    print("\n" + "="*60)
    print("Batch Run Summary")
    print("="*60)
    for model_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {model_name}: {status}")
    
    print("\nOutput directories:")
    if os.path.exists(OUTPUTS_BASE):
        for dirname in sorted(os.listdir(OUTPUTS_BASE)):
            dirpath = os.path.join(OUTPUTS_BASE, dirname)
            if os.path.isdir(dirpath):
                print(f"  - {dirpath}")
    
    print("="*60)


if __name__ == "__main__":
    main()
