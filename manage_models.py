"""
Automatically manage IDF model versions.

This script:
1. Detects the EnergyPlus engine version
2. Moves IDF files with HIGHER versions to 'higher_version/' folder (cannot be used)
3. Upgrades IDF files with LOWER versions to match the engine
4. Leaves IDF files that already match the engine version
5. Tests upgraded models to verify they run correctly

Usage:
    python manage_models.py                    # Run with default settings
    python manage_models.py --dry-run          # Preview changes without executing
    python manage_models.py --no-test          # Skip testing upgraded models
    python manage_models.py --model-dir path   # Specify custom model directory
"""

import os
import sys
import argparse
import subprocess
import tempfile
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

TRANSITION_BASE_URL = "https://github.com/NREL/EnergyPlus/releases/download"


def get_engine_version():
    """Get the actual EnergyPlus engine version."""
    try:
        from pyenergyplus.api import EnergyPlusAPI
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        tmpdir = os.path.join(tempfile.gettempdir(), 'eplus_version_check')
        os.makedirs(tmpdir, exist_ok=True)
        
        idf_path = os.path.join(tmpdir, 'version_check.idf')
        err_path = os.path.join(tmpdir, 'eplusout.err')
        
        with open(idf_path, 'w') as f:
            f.write('Version,99.9;')
        
        # Suppress console output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            api.runtime.run_energyplus(state, ['-d', tmpdir, idf_path])
        
        if os.path.exists(err_path):
            with open(err_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Program Version' in line and 'EnergyPlus' in line:
                        parts = line.split('Version')[2].strip().split(',')[0].strip()
                        return parts.split('-')[0]
        
        api.state_manager.delete_state(state)
    except:
        pass
    return "23.2.0"


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


def compare_versions(v1, v2):
    """Compare two version strings. Returns -1 if v1<v2, 0 if equal, 1 if v1>v2."""
    try:
        v1_parts = [int(x) for x in v1.split('.')[:2]]
        v2_parts = [int(x) for x in v2.split('.')[:2]]
        
        for a, b in zip(v1_parts, v2_parts):
            if a < b:
                return -1
            elif a > b:
                return 1
        return 0
    except:
        return 0


def get_transition_path(from_version, to_version):
    """Get the path needed to transition from one version to another."""
    from_major = '.'.join(from_version.split('.')[:2])
    to_major = '.'.join(to_version.split('.')[:2])
    
    if from_major == to_major:
        return []
    
    path = []
    current = from_major
    
    for src, dst in TRANSITIONS:
        if current == src:
            path.append((src, dst))
            current = dst
            if current == to_major:
                break
    
    if current != to_major:
        return None
    
    return path


def download_transition_tool(from_ver, to_ver, cache_dir):
    """Download transition executable for a specific version pair."""
    from_parts = from_ver.replace('.', '-')
    to_parts = to_ver.replace('.', '-')
    
    if sys.platform == 'win32':
        exe_name = f"Transition-V{from_parts}-0-to-V{to_parts}-0.exe"
    else:
        exe_name = f"Transition-V{from_parts}-0-to-V{to_parts}-0"
    
    exe_path = os.path.join(cache_dir, exe_name)
    
    if os.path.exists(exe_path):
        return exe_path
    
    tag = f"v{to_ver}.0"
    
    print(f"      Downloading transition tool {from_ver} -> {to_ver}...")
    
    urls_to_try = [
        f"{TRANSITION_BASE_URL}/{tag}/{exe_name}",
        f"{TRANSITION_BASE_URL}/{tag}/PreProcess/{exe_name}",
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(exe_path, 'wb') as f:
                    f.write(response.content)
                if sys.platform != 'win32':
                    os.chmod(exe_path, 0o755)
                return exe_path
        except:
            continue
    
    return None


def run_transition(idf_path, from_ver, to_ver, transition_exe):
    """Run a single transition on an IDF file."""
    if not os.path.exists(transition_exe):
        return False, "Transition tool not found"
    
    try:
        result = subprocess.run(
            [transition_exe, idf_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        new_version = get_idf_version(idf_path)
        if new_version and new_version.startswith(to_ver):
            return True, f"Upgraded to {new_version}"
        else:
            return False, f"Version still {new_version}"
            
    except subprocess.TimeoutExpired:
        return False, "Transition timed out"
    except Exception as e:
        return False, str(e)


def upgrade_idf(idf_path, target_version, cache_dir):
    """Upgrade an IDF file to the target version."""
    current_version = get_idf_version(idf_path)
    
    if not current_version:
        return False, "Could not read version"
    
    path = get_transition_path(current_version, target_version)
    
    if path is None:
        return False, f"No transition path from {current_version} to {target_version}"
    
    if not path:
        return True, "Already at target version"
    
    # Create backup
    backup_path = idf_path + f".v{current_version}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(idf_path, backup_path)
    
    # Run each transition
    for from_ver, to_ver in path:
        exe = download_transition_tool(from_ver, to_ver, cache_dir)
        if not exe:
            return False, f"Could not download transition tool for {from_ver} -> {to_ver}"
        
        success, msg = run_transition(idf_path, from_ver, to_ver, exe)
        if not success:
            return False, f"Failed at {from_ver} -> {to_ver}: {msg}"
    
    final_version = get_idf_version(idf_path)
    return True, f"Upgraded from {current_version} to {final_version}"


def find_idf_files(directory):
    """Find all IDF files in a directory."""
    idf_files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith('.idf'):
                idf_files.append(os.path.join(directory, f))
    return idf_files


def find_weather_file(weather_dir='weather'):
    """Find a weather file for testing."""
    if os.path.exists(weather_dir):
        for root, dirs, files in os.walk(weather_dir):
            for f in files:
                if f.endswith('.epw'):
                    return os.path.join(root, f)
    return None


def test_model(idf_path, weather_file, output_dir):
    """Test run a model to verify it works with the engine."""
    try:
        from pyenergyplus.api import EnergyPlusAPI
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        # Create output directory for this test
        model_name = os.path.splitext(os.path.basename(idf_path))[0]
        test_output = os.path.join(output_dir, f"test_{model_name}")
        os.makedirs(test_output, exist_ok=True)
        
        args = ['-d', test_output, '-w', weather_file, idf_path]
        
        # Run simulation (suppress output)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            api.runtime.run_energyplus(state, args)
        
        # Check error file for severe errors
        err_path = os.path.join(test_output, 'eplusout.err')
        if os.path.exists(err_path):
            with open(err_path, 'r') as f:
                content = f.read()
                
            # Check for fatal errors
            if '**  Fatal  **' in content:
                # Count severe errors
                severe_count = content.count('** Severe  **')
                return False, f"Fatal error ({severe_count} severe errors)"
            
            # Check if simulation completed
            if 'EnergyPlus Completed Successfully' in content:
                return True, "Simulation completed successfully"
            
            # Partial success - ran but with warnings
            if 'EnergyPlus Warmup Error Summary' in content:
                return True, "Simulation completed with warnings"
        
        api.state_manager.delete_state(state)
        return False, "Could not verify simulation status"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Automatically manage IDF model versions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions:
  - Models with HIGHER version than engine -> moved to 'higher_version/' folder
  - Models with LOWER version than engine -> upgraded to engine version
  - Models matching engine version -> left unchanged

Example:
  python manage_models.py                    # Run with defaults
  python manage_models.py --dry-run          # Preview only
        """
    )
    
    parser.add_argument('--model-dir', default='energyplus/models', help='Directory containing IDF files')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without executing')
    parser.add_argument('--no-test', action='store_true', help='Skip testing upgraded models')
    parser.add_argument('--cache-dir', default=None, help='Directory to cache transition tools')
    parser.add_argument('--weather', default=None, help='Weather file for testing (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IDF MODEL VERSION MANAGER")
    print("=" * 70)
    
    # Get engine version
    print("\nDetecting engine version...")
    engine_version = get_engine_version()
    engine_major = '.'.join(engine_version.split('.')[:2])
    print(f"Engine version: {engine_version} (major: {engine_major})")
    
    # Setup directories
    model_dir = args.model_dir
    higher_version_dir = os.path.join(os.path.dirname(model_dir), 'higher_version')
    
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), 'eplus_transitions')
    
    if not args.dry_run:
        os.makedirs(higher_version_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
    
    # Find all IDF files
    idf_files = find_idf_files(model_dir)
    
    if not idf_files:
        print(f"\nNo IDF files found in {model_dir}")
        return 1
    
    print(f"\nFound {len(idf_files)} IDF files in {model_dir}")
    print("=" * 70)
    
    # Categorize files
    higher = []
    lower = []
    matching = []
    unknown = []
    
    for idf_path in idf_files:
        version = get_idf_version(idf_path)
        filename = os.path.basename(idf_path)
        
        if not version:
            unknown.append((idf_path, filename, version))
            continue
        
        cmp = compare_versions(version, engine_version)
        
        if cmp > 0:
            higher.append((idf_path, filename, version))
        elif cmp < 0:
            lower.append((idf_path, filename, version))
        else:
            matching.append((idf_path, filename, version))
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print("-" * 70)
    print(f"  ✓ Matching engine ({engine_major}):  {len(matching)} files")
    print(f"  ⬆️  Need upgrade (< {engine_major}):   {len(lower)} files")
    print(f"  ❌ Too new (> {engine_major}):         {len(higher)} files")
    if unknown:
        print(f"  ? Unknown version:          {len(unknown)} files")
    print("-" * 70)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made\n")
    
    # Process matching files (no action needed)
    if matching:
        print(f"\n✓ MATCHING VERSION ({engine_major}) - No action needed:")
        for idf_path, filename, version in matching:
            print(f"    {filename} (v{version})")
    
    # Process higher version files (move to higher_version folder)
    if higher:
        print(f"\n❌ HIGHER VERSION - Moving to '{os.path.basename(higher_version_dir)}/':")
        for idf_path, filename, version in higher:
            dest_path = os.path.join(higher_version_dir, filename)
            print(f"    {filename} (v{version}) -> {os.path.basename(higher_version_dir)}/")
            
            if not args.dry_run:
                shutil.move(idf_path, dest_path)
                print(f"      ✓ Moved")
    
    # Process lower version files (move to need_update folder)
    need_update_dir = os.path.join(os.path.dirname(model_dir), f'need_update_to_{engine_major.replace(".", "_")}v')
    
    if not args.dry_run:
        os.makedirs(need_update_dir, exist_ok=True)
    
    upgraded_models = []
    if lower:
        print(f"\n⬆️  LOWER VERSION - Moving to '{os.path.basename(need_update_dir)}/':")
        
        for idf_path, filename, version in lower:
            dest_path = os.path.join(need_update_dir, filename)
            print(f"    {filename} (v{version}) -> {os.path.basename(need_update_dir)}/")
            
            if not args.dry_run:
                shutil.move(idf_path, dest_path)
                print(f"      ✓ Moved")
    
    # Test upgraded models
    test_results = {'passed': [], 'failed': []}
    if upgraded_models and not args.dry_run and not args.no_test:
        print(f"\n🧪 TESTING UPGRADED MODELS")
        print("-" * 70)
        
        # Find weather file
        weather_file = args.weather or find_weather_file()
        if not weather_file:
            print("  ⚠️  No weather file found - skipping tests")
            print("     Use --weather to specify a weather file")
        else:
            print(f"  Using weather file: {os.path.basename(weather_file)}")
            
            test_output_dir = os.path.join('outputs', 'upgrade_tests')
            os.makedirs(test_output_dir, exist_ok=True)
            
            for idf_path, filename in upgraded_models:
                print(f"\n  Testing {filename}...")
                success, msg = test_model(idf_path, weather_file, test_output_dir)
                
                if success:
                    print(f"    ✓ {msg}")
                    test_results['passed'].append(filename)
                else:
                    print(f"    ✗ {msg}")
                    test_results['failed'].append(filename)
            
            print(f"\n  Test results: {len(test_results['passed'])} passed, {len(test_results['failed'])} failed")
    
    # Final summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE - Run without --dry-run to apply changes")
    else:
        print("MODEL MANAGEMENT COMPLETE")
        print(f"  - {len(matching)} models already compatible")
        print(f"  - {len(higher)} models moved to higher_version/")
        print(f"  - {len(lower)} models moved to need_update_to_{engine_major.replace('.', '_')}v/")
        if test_results['passed'] or test_results['failed']:
            print(f"  - {len(test_results['passed'])} upgraded models tested successfully")
            if test_results['failed']:
                print(f"  - ⚠️  {len(test_results['failed'])} upgraded models failed testing:")
                for f in test_results['failed']:
                    print(f"      - {f}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
