"""
Upgrade IDF files to match the installed EnergyPlus engine version.

This script uses local EnergyPlus transition tools (if available) or downloads
them from GitHub to upgrade older IDF files to newer versions.

Usage:
    python upgrade_idf.py model.idf                    # Upgrade single file
    python upgrade_idf.py --all                        # Upgrade all models in energyplus/models/
    python upgrade_idf.py model.idf --target 23.2     # Upgrade to specific version
    python upgrade_idf.py --check                      # Check which models need upgrading
"""

import os
import sys
import argparse
import subprocess
import tempfile
import zipfile
import shutil
import requests
from pathlib import Path


# Transition versions available (source -> target)
TRANSITIONS = [
    ("22.1", "22.2"),
    ("22.2", "23.1"),
    ("23.1", "23.2"),
    ("23.2", "24.1"),
    ("24.1", "24.2"),
    ("24.2", "25.1"),
    ("25.1", "25.2"),
]

# GitHub release URLs for transition tools
TRANSITION_BASE_URL = "https://github.com/NREL/EnergyPlus/releases/download"

# Common EnergyPlus installation paths
EPLUS_INSTALL_PATHS = [
    "C:\\EnergyPlusV{ver}",
    "C:\\Program Files\\EnergyPlusV{ver}",
    "C:\\Program Files (x86)\\EnergyPlusV{ver}",
    "/usr/local/EnergyPlus-{ver}",
    "/Applications/EnergyPlus-{ver}",
]


def get_engine_version():
    """Get the actual EnergyPlus engine version."""
    try:
        from pyenergyplus.api import EnergyPlusAPI
        
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        tmpdir = os.path.join(tempfile.gettempdir(), 'eplus_version_check')
        os.makedirs(tmpdir, exist_ok=True)
        
        idf_path = os.path.join(tmpdir, 'version_check.idf')
        err_path = os.path.join(tmpdir, 'eplusout.err')
        
        with open(idf_path, 'w') as f:
            f.write('Version,99.9;')
        
        # Suppress output
        import io
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            api.runtime.run_energyplus(state, ['-d', tmpdir, idf_path])
        
        if os.path.exists(err_path):
            with open(err_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Program Version' in line and 'EnergyPlus' in line:
                        parts = line.split('Version')[2].strip().split(',')[0].strip()
                        return parts.split('-')[0]  # Return just "23.2.0"
        
        api.state_manager.delete_state(state)
    except:
        pass
    return "23.2.0"  # Default


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


def get_transition_path(from_version, to_version):
    """Get the path needed to transition from one version to another."""
    from_major = '.'.join(from_version.split('.')[:2])
    to_major = '.'.join(to_version.split('.')[:2])
    
    if from_major == to_major:
        return []  # Same version, no transition needed
    
    path = []
    current = from_major
    
    for src, dst in TRANSITIONS:
        if current == src:
            path.append((src, dst))
            current = dst
            if current == to_major:
                break
    
    if current != to_major:
        return None  # No valid path found
    
    return path


def find_local_eplus_installations():
    """Find local EnergyPlus installations with transition tools."""
    installations = []
    
    if sys.platform == 'win32':
        # Search common Windows paths
        for drive in ['C:', 'D:']:
            import glob
            patterns = [
                f"{drive}\\EnergyPlusV*",
                f"{drive}\\Program Files\\EnergyPlusV*",
                f"{drive}\\Program Files (x86)\\EnergyPlusV*",
            ]
            for pattern in patterns:
                for path in glob.glob(pattern):
                    updater_path = os.path.join(path, 'PreProcess', 'IDFVersionUpdater')
                    if os.path.exists(updater_path):
                        installations.append(updater_path)
    else:
        # Linux/Mac
        import glob
        patterns = [
            "/usr/local/EnergyPlus-*",
            "/Applications/EnergyPlus-*",
            os.path.expanduser("~/EnergyPlus-*"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                updater_path = os.path.join(path, 'PreProcess', 'IDFVersionUpdater')
                if os.path.exists(updater_path):
                    installations.append(updater_path)
    
    return installations


def find_transition_tool(from_ver, to_ver, local_paths, cache_dir):
    """Find transition tool locally or download it."""
    from_parts = from_ver.replace('.', '-')
    to_parts = to_ver.replace('.', '-')
    
    if sys.platform == 'win32':
        exe_name = f"Transition-V{from_parts}-0-to-V{to_parts}-0.exe"
    else:
        exe_name = f"Transition-V{from_parts}-0-to-V{to_parts}-0"
    
    # First check local EnergyPlus installations
    for local_path in local_paths:
        exe_path = os.path.join(local_path, exe_name)
        if os.path.exists(exe_path):
            return exe_path, local_path
    
    # Check cache directory
    exe_path = os.path.join(cache_dir, exe_name)
    if os.path.exists(exe_path):
        return exe_path, cache_dir
    
    # Try to download from GitHub releases
    tag = f"v{to_ver}.0"
    
    print(f"      Downloading transition tool {from_ver} -> {to_ver}...")
    
    urls_to_try = [
        f"{TRANSITION_BASE_URL}/{tag}/{exe_name}",
        f"{TRANSITION_BASE_URL}/{tag}/PreProcess/{exe_name}",
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(exe_path, 'wb') as f:
                    f.write(response.content)
                if sys.platform != 'win32':
                    os.chmod(exe_path, 0o755)
                print(f"      ✓ Downloaded: {exe_name}")
                return exe_path, cache_dir
        except:
            continue
    
    return None, None


def run_transition(idf_path, from_ver, to_ver, transition_exe, working_dir):
    """Run a single transition on an IDF file."""
    if not os.path.exists(transition_exe):
        return False, "Transition tool not found"
    
    # Create backup
    backup_path = idf_path + f".v{from_ver}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(idf_path, backup_path)
    
    # Copy IDF to working directory (transition tools need IDD files in same dir)
    idf_name = os.path.basename(idf_path)
    temp_idf = os.path.join(working_dir, idf_name)
    shutil.copy2(idf_path, temp_idf)
    
    # Run transition from the working directory
    try:
        result = subprocess.run(
            [transition_exe, idf_name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=working_dir
        )
        
        # Copy back the upgraded file
        if os.path.exists(temp_idf):
            shutil.copy2(temp_idf, idf_path)
        
        # Check if successful by reading version
        new_version = get_idf_version(idf_path)
        if new_version and new_version.startswith(to_ver):
            # Clean up temp files
            for ext in ['.idfold', '.idfnew', '.VCperr']:
                temp_file = os.path.join(working_dir, idf_name.replace('.idf', ext))
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(temp_idf):
                os.remove(temp_idf)
            return True, f"Upgraded to {new_version}"
        else:
            return False, f"Version still {new_version}"
            
    except subprocess.TimeoutExpired:
        return False, "Transition timed out"
    except Exception as e:
        return False, str(e)


def upgrade_idf(idf_path, target_version, cache_dir, local_paths=None):
    """Upgrade an IDF file to the target version."""
    if local_paths is None:
        local_paths = find_local_eplus_installations()
    
    current_version = get_idf_version(idf_path)
    
    if not current_version:
        print(f"  ✗ Could not read version from {idf_path}")
        return False
    
    target_major = '.'.join(target_version.split('.')[:2])
    current_major = '.'.join(current_version.split('.')[:2])
    
    if current_major == target_major:
        print(f"  ✓ {os.path.basename(idf_path)} already at version {current_version}")
        return True
    
    if float(current_major) > float(target_major):
        print(f"  ✗ {os.path.basename(idf_path)} is v{current_version} (newer than target {target_version})")
        print(f"    Downgrading is not supported")
        return False
    
    # Get transition path
    path = get_transition_path(current_version, target_version)
    
    if path is None:
        print(f"  ✗ No transition path from {current_version} to {target_version}")
        return False
    
    print(f"  Upgrading {os.path.basename(idf_path)}: {current_version} -> {target_version}")
    print(f"    Path: {' -> '.join([p[0] for p in path] + [path[-1][1]])}")
    
    # Run each transition
    for from_ver, to_ver in path:
        exe, working_dir = find_transition_tool(from_ver, to_ver, local_paths, cache_dir)
        if not exe:
            print(f"    ✗ Could not find transition tool for {from_ver} -> {to_ver}")
            print(f"      Install EnergyPlus or download from: https://github.com/NREL/EnergyPlus/releases")
            return False
        
        success, msg = run_transition(idf_path, from_ver, to_ver, exe, working_dir)
        if not success:
            print(f"    ✗ Failed at {from_ver} -> {to_ver}: {msg}")
            return False
        print(f"    ✓ {from_ver} -> {to_ver}")
    
    final_version = get_idf_version(idf_path)
    print(f"  ✓ Upgraded to {final_version}")
    return True


def test_idf(idf_path, weather_file=None):
    """Test an IDF file by running a simulation."""
    try:
        from pyenergyplus.api import EnergyPlusAPI
        
        # Find weather file if not provided
        if not weather_file:
            weather_dirs = ['weather', 'weather/chicago', 'weather/atlanta']
            for wdir in weather_dirs:
                if os.path.exists(wdir):
                    for f in os.listdir(wdir):
                        if f.endswith('.epw'):
                            weather_file = os.path.join(wdir, f)
                            break
                if weather_file:
                    break
        
        if not weather_file or not os.path.exists(weather_file):
            return False, "No weather file found"
        
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        # Create temp output directory
        output_dir = os.path.join(tempfile.gettempdir(), 'eplus_test_' + os.path.basename(idf_path).replace('.idf', ''))
        os.makedirs(output_dir, exist_ok=True)
        
        # Run simulation (don't use -r flag to avoid ReadVarsESO error)
        api.runtime.run_energyplus(
            state,
            ['-w', weather_file, '-d', output_dir, idf_path]
        )
        
        api.state_manager.delete_state(state)
        
        # Check error file for success
        err_file = os.path.join(output_dir, 'eplusout.err')
        if os.path.exists(err_file):
            with open(err_file, 'r') as f:
                content = f.read()
                if '** Fatal **' in content:
                    return False, "Fatal error in simulation"
                if 'EnergyPlus Completed Successfully' in content:
                    return True, "Simulation completed successfully"
        
        # Check if output files exist as backup indicator
        eso_file = os.path.join(output_dir, 'eplusout.eso')
        if os.path.exists(eso_file):
            return True, "Simulation completed (output files created)"
        
        return False, "Simulation failed"
        
    except Exception as e:
        return False, str(e)


def find_idf_files(directory):
    """Find all IDF files in a directory."""
    idf_files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith('.idf'):
                idf_files.append(os.path.join(directory, f))
    return idf_files


def main():
    parser = argparse.ArgumentParser(
        description='Upgrade IDF files to match the EnergyPlus engine version.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upgrade_idf.py model.idf                    # Upgrade single file
  python upgrade_idf.py model.idf --test             # Upgrade and test
  python upgrade_idf.py model.idf --test --move-to energyplus/models  # Upgrade, test, move on success
  python upgrade_idf.py model.idf --test --move-to energyplus/models --clean-backups  # Also remove backups
  python upgrade_idf.py model.idf --test --move-to energyplus/models --clean-source   # Remove source file after move
  python upgrade_idf.py --all                        # Upgrade all models
  python upgrade_idf.py --check                      # Check which need upgrading
  python upgrade_idf.py --list-backups               # List backup files
  python upgrade_idf.py --delete-backups             # Delete all backup files

Note: Uses local EnergyPlus installations for transition tools if available.
      Downgrading (newer -> older) is NOT supported.
        """
    )
    
    parser.add_argument('idf_file', nargs='?', help='IDF file to upgrade')
    parser.add_argument('--all', action='store_true', help='Upgrade all IDF files in energyplus/models/')
    parser.add_argument('--check', action='store_true', help='Check which models need upgrading')
    parser.add_argument('--list-backups', action='store_true', help='List backup files in model directory')
    parser.add_argument('--delete-backups', action='store_true', help='Delete all backup files in model directory')
    parser.add_argument('--test', action='store_true', help='Test upgraded file with a simulation')
    parser.add_argument('--move-to', type=str, help='Move file to this directory after successful test')
    parser.add_argument('--clean-backups', action='store_true', help='Remove backup files after successful test')
    parser.add_argument('--clean-source', action='store_true', help='Remove source IDF file after successful move')
    parser.add_argument('--weather', type=str, help='Weather file for testing')
    parser.add_argument('--target', type=str, help='Target version (default: engine version)')
    parser.add_argument('--model-dir', default='energyplus/models', help='Directory containing IDF files')
    parser.add_argument('--cache-dir', default=None, help='Directory to cache transition tools')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IDF VERSION UPGRADER")
    print("=" * 60)
    
    # Find local EnergyPlus installations
    print("\nSearching for local EnergyPlus installations...")
    local_paths = find_local_eplus_installations()
    if local_paths:
        print(f"  ✓ Found {len(local_paths)} installation(s) with transition tools:")
        for p in local_paths:
            print(f"    - {p}")
    else:
        print("  ⚠️  No local installations found (will try to download tools)")
    
    # Get target version
    if args.target:
        target_version = args.target
    else:
        print("\nDetecting engine version...")
        target_version = get_engine_version()
    
    print(f"Target version: {target_version}")
    
    # Setup cache directory for transition tools
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), 'eplus_transitions')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Handle backup file operations
    if args.list_backups or args.delete_backups:
        print("\nBACKUP FILES")
        print("=" * 60)
        
        backups = []
        if os.path.exists(args.model_dir):
            for f in os.listdir(args.model_dir):
                if '.backup' in f:
                    backups.append(os.path.join(args.model_dir, f))
        
        if not backups:
            print(f"No backup files found in {args.model_dir}/")
            return 0
        
        print(f"Found {len(backups)} backup file(s):\n")
        
        for backup_path in sorted(backups):
            backup_name = os.path.basename(backup_path)
            size_kb = os.path.getsize(backup_path) / 1024
            
            # Check if upgraded IDF exists
            parts = backup_name.split('.idf.')
            if len(parts) >= 2:
                idf_name = parts[0] + '.idf'
                idf_path = os.path.join(args.model_dir, idf_name)
                if os.path.exists(idf_path):
                    current_ver = get_idf_version(idf_path)
                    status = f"✓ Upgraded to v{current_ver}"
                else:
                    status = "⚠️  IDF not found"
            else:
                status = ""
            
            print(f"  {backup_name} ({size_kb:.1f} KB) {status}")
            
            if args.delete_backups:
                os.remove(backup_path)
                print(f"    ✓ Deleted")
        
        print("\n" + "=" * 60)
        if args.delete_backups:
            print(f"Deleted {len(backups)} backup file(s)")
        else:
            print("Run with --delete-backups to remove these files")
        return 0
    
    # Get list of files to process
    if args.all:
        idf_files = find_idf_files(args.model_dir)
    elif args.idf_file:
        idf_files = [args.idf_file]
    elif args.check:
        idf_files = find_idf_files(args.model_dir)
    else:
        parser.print_help()
        return 1
    
    if not idf_files:
        print("No IDF files found.")
        return 1
    
    print("=" * 60)
    
    # Check mode
    if args.check:
        print("MODEL VERSION CHECK")
        print("=" * 60)
        target_major = '.'.join(target_version.split('.')[:2])
        
        needs_upgrade = []
        too_new = []
        ok = []
        
        for idf_path in sorted(idf_files):
            version = get_idf_version(idf_path)
            filename = os.path.basename(idf_path)
            
            if not version:
                print(f"  ? {filename}: Unknown version")
                continue
            
            version_major = '.'.join(version.split('.')[:2])
            
            if float(version_major) < float(target_major):
                needs_upgrade.append((filename, version))
                print(f"  ⬆️  {filename}: {version} -> needs upgrade")
            elif float(version_major) > float(target_major):
                too_new.append((filename, version))
                print(f"  ❌ {filename}: {version} -> too new (cannot downgrade)")
            else:
                ok.append((filename, version))
                print(f"  ✓ {filename}: {version} -> OK")
        
        print("=" * 60)
        print(f"Summary: {len(ok)} OK, {len(needs_upgrade)} need upgrade, {len(too_new)} too new")
        return 0
    
    # Upgrade mode
    print("UPGRADING IDF FILES")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    test_passed = 0
    test_failed = 0
    moved_count = 0
    
    for idf_path in idf_files:
        if upgrade_idf(idf_path, target_version, cache_dir, local_paths):
            success_count += 1
            
            # Test if requested
            if args.test:
                print(f"\n  🧪 Testing {os.path.basename(idf_path)}...")
                test_success, test_msg = test_idf(idf_path, args.weather)
                
                if test_success:
                    print(f"    ✓ {test_msg}")
                    test_passed += 1
                    
                    # Move if requested and test passed
                    if args.move_to:
                        os.makedirs(args.move_to, exist_ok=True)
                        dest_path = os.path.join(args.move_to, os.path.basename(idf_path))
                        idf_dir = os.path.dirname(idf_path) or '.'
                        idf_name = os.path.basename(idf_path)
                        shutil.move(idf_path, dest_path)
                        print(f"    ✓ Moved to {args.move_to}/")
                        moved_count += 1
                        
                        # Remove backup/intermediate files if --clean-backups or --clean-source
                        if args.clean_backups or args.clean_source:
                            for f in os.listdir(idf_dir):
                                if f.startswith(idf_name) and '.backup' in f:
                                    backup_path = os.path.join(idf_dir, f)
                                    os.remove(backup_path)
                                    print(f"    ✓ Removed: {f}")
                else:
                    print(f"    ✗ {test_msg}")
                    test_failed += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Upgrades: {success_count} succeeded, {fail_count} failed")
    if args.test:
        print(f"  Tests:    {test_passed} passed, {test_failed} failed")
    if args.move_to:
        print(f"  Moved:    {moved_count} files to {args.move_to}/")
    
    return 0 if fail_count == 0 and test_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
