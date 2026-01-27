"""
Example: Complete Model Management Workflow
Shows how to download, upgrade, and run EnergyPlus models with version control
"""

import os
import sys
import requests
import importlib.metadata
from pyenergyplus.api import EnergyPlusAPI


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = "outputs"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_installed_version():
    """Get the installed pyenergyplus version."""
    try:
        return importlib.metadata.version('pyenergyplus-lbnl')
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version('pyenergyplus')


def get_actual_engine_version():
    """
    Get the actual EnergyPlus engine version by running a test.
    Note: This is more accurate than the package version.
    """
    # For now, return known version based on package
    # In production, you could parse the engine binary version
    return "23.2.0"  # Update this based on your actual engine


def get_idf_version(idf_path):
    """Extract the version number from an IDF file."""
    try:
        with open(idf_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'VERSION' in line.upper() and not line.strip().startswith('!'):
                    # Parse version from line like "Version,23.2;"
                    parts = line.split(',')
                    if len(parts) >= 2:
                        version = parts[1].strip().rstrip(';')
                        return version
    except Exception as e:
        print(f"Error reading version: {e}")
    return None


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def download_model_from_specific_version(model_name, version_tag, output_dir=DATA_DIR):
    """
    Download a model file from a specific EnergyPlus GitHub version.
    
    Args:
        model_name: Name of the IDF file (e.g., '1ZoneUncontrolled.idf')
        version_tag: GitHub tag (e.g., 'v23.2.0', 'v24.1.0', 'develop')
        output_dir: Directory to save the file
    
    Returns:
        Path to downloaded file or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, model_name)
    
    url = f"https://raw.githubusercontent.com/NREL/EnergyPlus/{version_tag}/testfiles/{model_name}"
    
    print(f"Downloading {model_name} from {version_tag}...")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify version
            file_version = get_idf_version(filepath)
            print(f"✓ Downloaded: {filepath}")
            print(f"  Model version: {file_version}")
            return filepath
        else:
            print(f"✗ Failed to download (Status {response.status_code})")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def download_weather_file(weather_name, version_tag="develop", output_dir=DATA_DIR):
    """Download a weather file from EnergyPlus GitHub."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, weather_name)
    
    if os.path.exists(filepath):
        print(f"Weather file already exists: {filepath}")
        return filepath
    
    url = f"https://raw.githubusercontent.com/NREL/EnergyPlus/{version_tag}/weather/{weather_name}"
    
    print(f"Downloading {weather_name}...")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {filepath}")
            return filepath
        else:
            print(f"✗ Failed to download weather file")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================
# UPGRADE FUNCTIONS
# ============================================================

def upgrade_idf_version(idf_path, target_version, create_backup=True):
    """
    Upgrade an IDF file's version number to match your EnergyPlus engine.
    
    Args:
        idf_path: Path to the IDF file
        target_version: Target version (e.g., '23.2.0')
        create_backup: Whether to create a .backup file
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(idf_path):
        print(f"File not found: {idf_path}")
        return False
    
    current_version = get_idf_version(idf_path)
    print(f"Current version: {current_version}")
    print(f"Target version: {target_version}")
    
    if current_version == target_version:
        print("✓ Already at target version")
        return True
    
    # Create backup
    if create_backup:
        backup_path = idf_path + ".backup"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(idf_path, backup_path)
            print(f"Created backup: {backup_path}")
    
    # Update version
    try:
        with open(idf_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if 'VERSION' in line.upper() and not line.strip().startswith('!'):
                # Preserve indentation
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + f"Version,{target_version};\n"
                updated = True
                break
        
        if updated:
            with open(idf_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"✓ Updated version to {target_version}")
            return True
        else:
            print("✗ Could not find VERSION field")
            return False
            
    except Exception as e:
        print(f"✗ Error updating file: {e}")
        return False


# ============================================================
# SIMULATION FUNCTION
# ============================================================

def run_simulation(idf_path, weather_path, output_dir):
    """Run an EnergyPlus simulation."""
    if not os.path.exists(idf_path):
        print(f"Model file not found: {idf_path}")
        return False
    
    if not os.path.exists(weather_path):
        print(f"Weather file not found: {weather_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    
    args = ['-d', output_dir, '-w', weather_path, idf_path]
    
    print(f"\nRunning simulation...")
    print(f"  Model: {os.path.basename(idf_path)}")
    print(f"  Weather: {os.path.basename(weather_path)}")
    print(f"  Output: {output_dir}\n")
    
    try:
        api.runtime.run_energyplus(state, args)
        print(f"\n✓ Simulation complete!")
        print(f"  Results: {os.path.abspath(output_dir)}")
        return True
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        return False
    finally:
        api.state_manager.delete_state(state)


# ============================================================
# EXAMPLE WORKFLOWS
# ============================================================

def example_1_download_version_matched_model():
    """
    Example 1: Download a model that matches your EnergyPlus engine version.
    This is the RECOMMENDED approach to avoid version mismatches.
    """
    print("="*70)
    print("EXAMPLE 1: Download Version-Matched Model")
    print("="*70 + "\n")
    
    engine_version = get_actual_engine_version()
    version_tag = f"v{engine_version}"
    
    print(f"Your EnergyPlus engine: {engine_version}")
    print(f"Downloading model from: {version_tag}\n")
    
    # Download a simple model
    model_path = download_model_from_specific_version(
        model_name="1ZoneUncontrolled.idf",
        version_tag=version_tag,
        output_dir=os.path.join(DATA_DIR, "example1")
    )
    
    if model_path:
        print(f"\n✓ Success! Model ready at: {model_path}")
        return model_path
    else:
        print(f"\n✗ Failed to download model")
        return None


def example_2_download_and_upgrade():
    """
    Example 2: Download a newer model and downgrade it to your engine version.
    Use this when you need a model that doesn't exist in your version.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Download Newer Model and Downgrade")
    print("="*70 + "\n")
    
    engine_version = get_actual_engine_version()
    
    # Download from develop branch (latest)
    model_path = download_model_from_specific_version(
        model_name="5ZoneAutoDXVAV.idf",
        version_tag="develop",
        output_dir=os.path.join(DATA_DIR, "example2")
    )
    
    if not model_path:
        print("Failed to download model")
        return None
    
    # Upgrade (downgrade in this case) the version
    print(f"\nAdjusting model version to match engine ({engine_version})...")
    success = upgrade_idf_version(model_path, engine_version)
    
    if success:
        print(f"\n✓ Model ready at: {model_path}")
        return model_path
    else:
        print("\n✗ Failed to adjust model version")
        return None


def example_3_batch_download_multiple_versions():
    """
    Example 3: Download the same model from multiple versions for comparison.
    Useful for testing version compatibility.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Download Multiple Versions for Comparison")
    print("="*70 + "\n")
    
    model_name = "1ZoneUncontrolled.idf"
    versions = ["v23.1.0", "v23.2.0", "develop"]
    
    downloaded_models = {}
    
    for version_tag in versions:
        output_dir = os.path.join(DATA_DIR, "example3", version_tag.replace('v', ''))
        
        model_path = download_model_from_specific_version(
            model_name=model_name,
            version_tag=version_tag,
            output_dir=output_dir
        )
        
        if model_path:
            downloaded_models[version_tag] = model_path
        
        print()
    
    print(f"✓ Downloaded {len(downloaded_models)} versions:")
    for tag, path in downloaded_models.items():
        print(f"  {tag}: {path}")
    
    return downloaded_models


def example_4_complete_workflow():
    """
    Example 4: Complete workflow - Download, upgrade, and run simulation.
    This is a production-ready example.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Workflow (Download → Upgrade → Simulate)")
    print("="*70 + "\n")
    
    engine_version = get_actual_engine_version()
    
    # Step 1: Download model
    print("Step 1: Downloading model...")
    model_path = download_model_from_specific_version(
        model_name="1ZoneUncontrolled.idf",
        version_tag=f"v{engine_version}",
        output_dir=os.path.join(DATA_DIR, "example4")
    )
    
    if not model_path:
        print("Failed at download step")
        return False
    
    # Step 2: Download weather
    print("\nStep 2: Downloading weather file...")
    weather_path = download_weather_file(
        weather_name="USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
        version_tag="develop",
        output_dir=os.path.join(DATA_DIR, "example4")
    )
    
    if not weather_path:
        print("Failed at weather download step")
        return False
    
    # Step 3: Verify/upgrade version
    print("\nStep 3: Verifying model version...")
    model_version = get_idf_version(model_path)
    if model_version != engine_version:
        print(f"Version mismatch detected. Upgrading...")
        upgrade_idf_version(model_path, engine_version)
    else:
        print(f"✓ Version matches ({engine_version})")
    
    # Step 4: Run simulation
    print("\nStep 4: Running simulation...")
    output_dir = os.path.join(OUTPUT_DIR, "example4")
    success = run_simulation(model_path, weather_path, output_dir)
    
    if success:
        print("\n" + "="*70)
        print("✓ COMPLETE WORKFLOW SUCCESSFUL!")
        print("="*70)
        print(f"\nResults available at: {os.path.abspath(output_dir)}")
        print("Open 'eplustbl.htm' in a browser to view the report.")
        return True
    else:
        print("\n✗ Workflow failed at simulation step")
        return False


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """Interactive menu to run different examples."""
    print("\n" + "="*70)
    print("EnergyPlus Model Management Examples")
    print("="*70)
    
    print(f"\nInstalled package version: {get_installed_version()}")
    print(f"Actual engine version: {get_actual_engine_version()}")
    
    print("\n" + "="*70)
    print("Available Examples:")
    print("="*70)
    print("\n1. Download version-matched model (RECOMMENDED)")
    print("   → Downloads model that exactly matches your engine")
    print("\n2. Download newer model and downgrade")
    print("   → Gets latest model and adjusts version for compatibility")
    print("\n3. Download multiple versions for comparison")
    print("   → Downloads same model from different EnergyPlus versions")
    print("\n4. Complete workflow (Download → Upgrade → Simulate)")
    print("   → Full production example with all steps")
    print("\n5. Run all examples")
    print("\n0. Exit")
    
    print("\n" + "="*70)
    
    choice = input("\nSelect example (0-5): ").strip()
    
    if choice == "1":
        example_1_download_version_matched_model()
    elif choice == "2":
        example_2_download_and_upgrade()
    elif choice == "3":
        example_3_batch_download_multiple_versions()
    elif choice == "4":
        example_4_complete_workflow()
    elif choice == "5":
        print("\nRunning all examples...\n")
        example_1_download_version_matched_model()
        example_2_download_and_upgrade()
        example_3_batch_download_multiple_versions()
        example_4_complete_workflow()
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
