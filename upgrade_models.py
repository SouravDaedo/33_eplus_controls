import os
import sys
import subprocess

DATA_DIR = "data"

def get_idf_version(idf_path):
    """Extract the version from an IDF file."""
    try:
        with open(idf_path, 'r') as f:
            for line in f:
                if line.strip().upper().startswith('VERSION'):
                    version_line = line.split(',')[0]
                    version = version_line.split(',')[-1].strip().rstrip(';')
                    if not version.upper().startswith('VERSION'):
                        for part in line.split(','):
                            if any(char.isdigit() for char in part):
                                version = part.strip().rstrip(';')
                                break
                    return version
    except Exception as e:
        print(f"Error reading version from {idf_path}: {e}")
    return None


def upgrade_idf(idf_path, target_version="23.2.0"):
    """Upgrade an IDF file to the target version."""
    if not os.path.exists(idf_path):
        print(f"File not found: {idf_path}")
        return False
    
    current_version = get_idf_version(idf_path)
    if current_version:
        print(f"Current model version: {current_version}")
    else:
        print("Could not detect model version")
    
    print(f"Target EnergyPlus version: {target_version}")
    
    # Create backup
    backup_path = idf_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(idf_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Manual version update (simple approach)
    print("\nUpdating version number in IDF file...")
    try:
        with open(idf_path, 'r') as f:
            content = f.read()
        
        # Find and replace version line
        lines = content.split('\n')
        updated = False
        for i, line in enumerate(lines):
            if line.strip().upper().startswith('VERSION'):
                lines[i] = f"  Version,{target_version};"
                updated = True
                break
        
        if updated:
            with open(idf_path, 'w') as f:
                f.write('\n'.join(lines))
            print(f"Updated version to {target_version}")
            return True
        else:
            print("Could not find VERSION field to update")
            return False
            
    except Exception as e:
        print(f"Error updating file: {e}")
        return False


def main():
    print("="*60)
    print("IDF Model Version Upgrader")
    print("="*60 + "\n")
    
    # Get target version (the version you're running)
    target_version = "23.2.0"  # Change this to match your EnergyPlus version
    
    print(f"Target EnergyPlus version: {target_version}\n")
    
    # Find all IDF files
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    idf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.idf')]
    
    if not idf_files:
        print(f"No IDF files found in {DATA_DIR}")
        return
    
    print(f"Found {len(idf_files)} IDF file(s):\n")
    
    for idf_file in idf_files:
        idf_path = os.path.join(DATA_DIR, idf_file)
        print(f"\n{'-'*60}")
        print(f"Processing: {idf_file}")
        print('-'*60)
        
        success = upgrade_idf(idf_path, target_version)
        
        if success:
            print(f"✓ Successfully updated {idf_file}")
        else:
            print(f"✗ Failed to update {idf_file}")
    
    print("\n" + "="*60)
    print("Upgrade complete!")
    print("="*60)
    print("\nNote: This script only updates the VERSION field.")
    print("For complex models with deprecated objects, you may need")
    print("the full EnergyPlus IDFVersionUpdater tool.")


if __name__ == "__main__":
    main()
