"""
Simple script to get the actual EnergyPlus engine version.
Uses multiple methods to detect the true engine version.
"""

import os
import sys
import importlib.metadata


def get_package_version():
    """Get the version from pip package metadata."""
    try:
        return importlib.metadata.version('pyenergyplus-lbnl')
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version('pyenergyplus')
        except:
            return None


def get_engine_version_from_idd():
    """
    Get version from the Energy+.idd file.
    The IDD (Input Data Dictionary) contains the engine version.
    """
    try:
        import pyenergyplus
        package_dir = os.path.dirname(pyenergyplus.__file__)
        
        # Common locations for IDD file
        possible_idd_paths = [
            os.path.join(package_dir, 'Energy+.idd'),
            os.path.join(package_dir, 'DataSets', 'Energy+.idd'),
            os.path.join(package_dir, '..', 'Energy+.idd'),
        ]
        
        for idd_path in possible_idd_paths:
            if os.path.exists(idd_path):
                print(f"Found IDD file: {idd_path}")
                
                with open(idd_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read first few lines - version is usually at the top
                    for i, line in enumerate(f):
                        if i > 50:  # Version should be in first 50 lines
                            break
                        
                        # Look for patterns like "!IDD_Version 23.2.0"
                        if 'IDD_Version' in line or 'IDD_BUILD' in line:
                            import re
                            version_match = re.search(r'(\d+\.\d+\.\d+)', line)
                            if version_match:
                                return version_match.group(1)
                
        return None
        
    except Exception as e:
        print(f"Error reading IDD: {e}")
        return None


def get_engine_version_from_energyplus_exe():
    """Try to get version from the EnergyPlus executable metadata."""
    try:
        import pyenergyplus
        package_dir = os.path.dirname(pyenergyplus.__file__)
        
        # Look for the energyplus binary
        import platform
        if platform.system() == 'Windows':
            exe_name = 'energyplus.exe'
        else:
            exe_name = 'energyplus'
        
        possible_exe_paths = [
            os.path.join(package_dir, exe_name),
            os.path.join(package_dir, 'bin', exe_name),
        ]
        
        for exe_path in possible_exe_paths:
            if os.path.exists(exe_path):
                print(f"Found EnergyPlus binary: {exe_path}")
                
                # On Windows, we could use version info from the exe
                # For now, just confirm it exists
                return None  # Would need platform-specific code to extract version
        
        return None
        
    except Exception as e:
        print(f"Error checking binary: {e}")
        return None


def main():
    """Display all version information."""
    print("="*70)
    print("EnergyPlus Version Information")
    print("="*70 + "\n")
    
    # Method 1: Package metadata
    package_version = get_package_version()
    print(f"Package metadata version: {package_version or 'Not found'}")
    
    # Method 2: IDD file
    print("\nChecking IDD file...")
    idd_version = get_engine_version_from_idd()
    if idd_version:
        print(f"IDD file version: {idd_version}")
    else:
        print("Could not detect version from IDD file")
    
    # Method 3: Binary check
    print("\nChecking EnergyPlus binary...")
    get_engine_version_from_energyplus_exe()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if package_version and idd_version:
        print(f"\nPackage reports: {package_version}")
        print(f"Engine (from IDD): {idd_version}")
        
        # Check for mismatch
        pkg_major_minor = '.'.join(package_version.split('.')[:2])
        idd_major_minor = '.'.join(idd_version.split('.')[:2])
        
        if pkg_major_minor != idd_major_minor:
            print("\n⚠ WARNING: Version mismatch detected!")
            print(f"  Package says {package_version} but engine is {idd_version}")
            print("\n  This means:")
            print("  - Download models from v{} for best compatibility".format(idd_version))
            print("  - Or upgrade your package to match the models you want")
        else:
            print(f"\n✓ Versions match (both {idd_major_minor}.x)")
    
    elif idd_version:
        print(f"\nEngine version: {idd_version}")
        print("Use this version when downloading models")
    
    elif package_version:
        print(f"\nPackage version: {package_version}")
        print("Warning: Could not verify actual engine version")
        print("Run a simulation to see the actual engine version in the output")
    
    else:
        print("\n✗ Could not detect any version information")
        print("\nTry running a simulation - the engine version will be printed at startup:")
        print("  python run_simulation.py")


if __name__ == "__main__":
    main()
