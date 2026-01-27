import os
import sys
import requests
import importlib.metadata

DATA_DIR = "data"


def get_energyplus_version():
    """Get the installed EnergyPlus version."""
    try:
        try:
            return importlib.metadata.version('pyenergyplus-lbnl')
        except importlib.metadata.PackageNotFoundError:
            return importlib.metadata.version('pyenergyplus')
    except Exception as e:
        print(f"Error detecting version: {e}")
        return None


def download_model_from_version(model_name, version_tag):
    """Download a specific model from a specific GitHub version tag."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = os.path.join(DATA_DIR, model_name)
    
    if os.path.exists(filepath):
        overwrite = input(f"{model_name} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Skipping download.")
            return filepath
    
    url = f"https://raw.githubusercontent.com/NREL/EnergyPlus/{version_tag}/testfiles/{model_name}"
    
    print(f"Downloading {model_name} from {version_tag}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {filepath}")
            
            # Show version in file
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if 'VERSION' in line.upper():
                            print(f"Model version: {line.strip()}")
                            break
            except:
                pass
            
            return filepath
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            print(f"URL: {url}")
            return None
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def main():
    print("="*60)
    print("Download Specific Version Model")
    print("="*60 + "\n")
    
    installed_version = get_energyplus_version()
    if installed_version:
        print(f"Your installed pyenergyplus-lbnl: {installed_version}")
    
    print("\nNote: Your actual EnergyPlus engine is v23.2.0")
    print("(Check your last simulation output)\n")
    
    print("Available version tags to download from:")
    print("  - v23.2.0  (matches your engine)")
    print("  - v23.1.0")
    print("  - v24.1.0")
    print("  - v24.2.0")
    print("  - v25.1.0")
    print("  - develop  (latest, may be unstable)")
    print()
    
    version_tag = input("Enter version tag (e.g., v23.2.0): ").strip()
    if not version_tag:
        version_tag = "v23.2.0"
        print(f"Using default: {version_tag}")
    
    print("\nCommon model files:")
    print("  1. 1ZoneUncontrolled.idf")
    print("  2. 5ZoneAutoDXVAV.idf")
    print("  3. RefBldgSmallOfficeNew2004_Chicago.idf")
    print("  4. RefBldgMediumOfficeNew2004_Chicago.idf")
    print()
    
    model_name = input("Enter model filename (or custom): ").strip()
    if not model_name:
        model_name = "1ZoneUncontrolled.idf"
        print(f"Using default: {model_name}")
    
    if not model_name.endswith('.idf'):
        model_name += '.idf'
    
    print()
    filepath = download_model_from_version(model_name, version_tag)
    
    if filepath:
        print(f"\n✓ Model saved to: {filepath}")
        print(f"\nTo run this model:")
        print(f"  python run_simulation.py")
    else:
        print("\n✗ Download failed")
        print("\nTip: Check available models at:")
        print(f"https://github.com/NREL/EnergyPlus/tree/{version_tag}/testfiles")


if __name__ == "__main__":
    main()
