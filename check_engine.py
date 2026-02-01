"""
Check EnergyPlus engine version and model compatibility.

Usage:
    python check_engine.py
    python check_engine.py --check-models
"""

import os
import sys
import argparse
import tempfile
import importlib.metadata


def get_pip_package_version():
    """Get the installed pip package version."""
    try:
        return importlib.metadata.version('pyenergyplus-lbnl')
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version('pyenergyplus')
        except importlib.metadata.PackageNotFoundError:
            return None


def get_engine_version():
    """Get the actual EnergyPlus engine version by running it."""
    try:
        from pyenergyplus.api import EnergyPlusAPI
        
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        # Use a persistent temp directory
        tmpdir = os.path.join(tempfile.gettempdir(), 'eplus_version_check')
        os.makedirs(tmpdir, exist_ok=True)
        
        idf_path = os.path.join(tmpdir, 'version_check.idf')
        err_path = os.path.join(tmpdir, 'eplusout.err')
        
        with open(idf_path, 'w') as f:
            f.write('Version,99.9;')  # Invalid version to force error
        
        # Run (will print to console but that's ok)
        api.runtime.run_energyplus(state, ['-d', tmpdir, idf_path])
        
        # Read version from error file
        if os.path.exists(err_path):
            with open(err_path, 'r') as f:
                content = f.read()
                # Look for "Program Version,EnergyPlus, Version 23.2.0-xxx"
                for line in content.split('\n'):
                    if 'Program Version' in line and 'EnergyPlus' in line:
                        # Extract "23.2.0-7636e6b3e9"
                        parts = line.split('Version')[2].strip().split(',')[0].strip()
                        api.state_manager.delete_state(state)
                        return parts
        
        api.state_manager.delete_state(state)
            
    except Exception as e:
        return f"Error: {e}"
    
    return None


def get_idf_version(filepath):
    """Extract version from an IDF file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'Version,' in line and not line.strip().startswith('!'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        return parts[1].strip().rstrip(';')
    except:
        pass
    return None


def find_idf_files(directories):
    """Find all IDF files in given directories."""
    idf_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith('.idf'):
                        idf_files.append(os.path.join(root, f))
    return idf_files


def check_compatibility(model_version, engine_version):
    """Check if model version is compatible with engine version."""
    try:
        model_major = float('.'.join(model_version.split('.')[:2]))
        engine_major = float('.'.join(engine_version.split('-')[0].split('.')[:2]))
        
        if model_major > engine_major:
            return "❌ Too new"
        elif model_major == engine_major:
            return "✓ Match"
        else:
            return "⚠️ Older (may work)"
    except:
        return "? Unknown"


def main():
    parser = argparse.ArgumentParser(description='Check EnergyPlus engine version and model compatibility.')
    parser.add_argument('--check-models', action='store_true', help='Check compatibility of all IDF models')
    parser.add_argument('--model-dirs', nargs='+', default=['energyplus/models'],
                        help='Directories to search for IDF files')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENERGYPLUS ENGINE VERSION CHECK")
    print("=" * 60)
    
    # Pip package version
    pip_version = get_pip_package_version()
    print(f"\nPip Package: pyenergyplus-lbnl {pip_version or 'Not installed'}")
    
    # Actual engine version
    print("\nDetecting actual engine version...")
    engine_version = get_engine_version()
    print(f"Engine Version: EnergyPlus {engine_version or 'Unknown'}")
    
    if pip_version and engine_version:
        print(f"\n⚠️  Note: Pip version ({pip_version}) differs from engine version ({engine_version.split('-')[0]})")
    
    print("=" * 60)
    
    # Check models if requested
    if args.check_models:
        print("\nMODEL COMPATIBILITY CHECK")
        print("=" * 60)
        
        idf_files = find_idf_files(args.model_dirs)
        
        if not idf_files:
            print("No IDF files found.")
            return 0
        
        engine_ver = engine_version.split('-')[0] if engine_version else "23.2.0"
        
        print(f"\nEngine: {engine_ver}")
        print("-" * 60)
        print(f"{'Model':<50} {'Version':<8} {'Status'}")
        print("-" * 60)
        
        for idf_path in sorted(idf_files):
            model_version = get_idf_version(idf_path)
            filename = os.path.basename(idf_path)
            if len(filename) > 48:
                filename = filename[:45] + "..."
            
            status = check_compatibility(model_version, engine_ver) if model_version else "? No version"
            print(f"{filename:<50} {model_version or 'N/A':<8} {status}")
        
        print("-" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
