"""
Detect the actual EnergyPlus engine version by running a test simulation.
This is more accurate than relying on package metadata.
"""

import os
import tempfile
import re
from pyenergyplus.api import EnergyPlusAPI


def detect_engine_version_from_api():
    """
    Detect the actual EnergyPlus engine version by running a minimal test.
    Returns the version string (e.g., '23.2.0') or None if detection fails.
    """
    
    # Create a minimal IDF file for testing
    # Note: We intentionally use a generic version that should work across versions
    # The engine will report its actual version regardless
    minimal_idf = """
Building,
  Simple Building,         !- Name
  0.0,                     !- North Axis {deg}
  Suburbs,                 !- Terrain
  0.04,                    !- Loads Convergence Tolerance Value
  0.4,                     !- Temperature Convergence Tolerance Value {deltaC}
  FullInteriorAndExterior, !- Solar Distribution
  25,                      !- Maximum Number of Warmup Days
  6;                       !- Minimum Number of Warmup Days

GlobalGeometryRules,
  UpperLeftCorner,         !- Starting Vertex Position
  CounterClockWise,        !- Vertex Entry Direction
  Relative;                !- Coordinate System

SimulationControl,
  No,                      !- Do Zone Sizing Calculation
  No,                      !- Do System Sizing Calculation
  No,                      !- Do Plant Sizing Calculation
  No,                      !- Run Simulation for Sizing Periods
  No;                      !- Run Simulation for Weather File Run Periods
"""
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.idf', delete=False) as f:
            idf_path = f.name
            f.write(minimal_idf)
        
        output_dir = tempfile.mkdtemp()
        
        # Run EnergyPlus and capture output
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        
        # Redirect output to capture version info
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # This will fail but we just need the version output
                api.runtime.run_energyplus(state, ['-d', output_dir, idf_path])
        except:
            pass  # Expected to fail with minimal IDF
        
        api.state_manager.delete_state(state)
        
        # Parse version from output
        output = stdout_capture.getvalue() + stderr_capture.getvalue()
        
        # Look for version pattern: "Version 23.2.0-xxxxx"
        version_match = re.search(r'Version\s+(\d+\.\d+\.\d+)', output)
        if version_match:
            version = version_match.group(1)
            return version
        
        # Alternative: Check error file
        err_file = os.path.join(output_dir, 'eplusout.err')
        if os.path.exists(err_file):
            with open(err_file, 'r') as f:
                err_content = f.read()
                version_match = re.search(r'Version\s+(\d+\.\d+\.\d+)', err_content)
                if version_match:
                    return version_match.group(1)
        
        return None
        
    except Exception as e:
        print(f"Error detecting version: {e}")
        return None
    
    finally:
        # Cleanup
        try:
            if os.path.exists(idf_path):
                os.unlink(idf_path)
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
        except:
            pass


def detect_version_from_binary():
    """
    Try to detect version by finding the EnergyPlus binary location.
    This is faster but less reliable.
    """
    try:
        import pyenergyplus
        package_path = os.path.dirname(pyenergyplus.__file__)
        
        # Look for version info in package files
        possible_paths = [
            os.path.join(package_path, 'Energy+.idd'),
            os.path.join(package_path, 'energyplus'),
            os.path.join(package_path, 'energyplus.exe'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found EnergyPlus file: {path}")
                
                # Try to read version from IDD file
                if path.endswith('.idd'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_lines = f.read(1000)
                        version_match = re.search(r'(\d+\.\d+\.\d+)', first_lines)
                        if version_match:
                            return version_match.group(1)
        
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Test version detection methods."""
    print("="*60)
    print("EnergyPlus Engine Version Detector")
    print("="*60 + "\n")
    
    # Method 1: Package metadata
    try:
        import importlib.metadata
        package_version = importlib.metadata.version('pyenergyplus-lbnl')
        print(f"Package metadata version: {package_version}")
    except:
        print("Package metadata version: Not available")
    
    # Method 2: Binary location
    print("\nMethod 1: Checking binary location...")
    binary_version = detect_version_from_binary()
    if binary_version:
        print(f"Detected from binary: {binary_version}")
    else:
        print("Could not detect from binary")
    
    # Method 3: Run test simulation (most accurate)
    print("\nMethod 2: Running test simulation (most accurate)...")
    print("This may take a few seconds...")
    engine_version = detect_engine_version_from_api()
    
    if engine_version:
        print(f"\n✓ Actual engine version: {engine_version}")
    else:
        print("\n✗ Could not detect engine version")
        print("\nFallback: Check your last simulation output for a line like:")
        print("  'EnergyPlus, Version 23.2.0-...'")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Package reports: {package_version if 'package_version' in locals() else 'Unknown'}")
    print(f"Actual engine: {engine_version if engine_version else 'Unknown'}")
    
    if engine_version and 'package_version' in locals() and engine_version != package_version.split('.')[0:3]:
        print("\n⚠ WARNING: Version mismatch detected!")
        print(f"  Your package says {package_version} but engine is {engine_version}")


if __name__ == "__main__":
    main()
