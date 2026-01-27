"""
Generic EnergyPlus simulation runner.
Specify IDF model and weather file via command-line arguments.

Usage:
    python run_sim.py --idf path/to/model.idf --epw path/to/weather.epw
    python run_sim.py --idf path/to/model.idf --epw path/to/weather.epw --output results/
"""

import os
import sys
import argparse
from pyenergyplus.api import EnergyPlusAPI


def run_simulation(idf_path, epw_path, output_dir):
    """Run EnergyPlus simulation with specified files."""
    
    # Validate inputs
    if not os.path.exists(idf_path):
        print(f"Error: IDF file not found: {idf_path}")
        return False
    
    if not os.path.exists(epw_path):
        print(f"Error: Weather file not found: {epw_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize API
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    
    # Build arguments
    args = ['-d', output_dir, '-w', epw_path, idf_path]
    
    print("\n" + "=" * 70)
    print("ENERGYPLUS SIMULATION")
    print("=" * 70)
    print(f"IDF Model:    {os.path.abspath(idf_path)}")
    print(f"Weather File: {os.path.abspath(epw_path)}")
    print(f"Output Dir:   {os.path.abspath(output_dir)}")
    print("=" * 70 + "\n")
    
    try:
        api.runtime.run_energyplus(state, args)
        
        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        
        # List key output files
        key_files = ['eplustbl.htm', 'eplusout.csv', 'eplusout.err', 'eplusout.eso']
        print("\nKey output files:")
        for filename in key_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✓ {filename} ({size:,} bytes)")
        
        # Check for errors
        err_file = os.path.join(output_dir, 'eplusout.err')
        if os.path.exists(err_file):
            with open(err_file, 'r') as f:
                content = f.read()
                if 'Severe' in content:
                    severe_count = content.count('** Severe  **')
                    print(f"\n⚠ Warning: {severe_count} severe error(s) found in eplusout.err")
                if 'Fatal' in content:
                    print("✗ Fatal error occurred - check eplusout.err")
                    return False
        
        print(f"\nFull results: {os.path.abspath(output_dir)}")
        return True
        
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        return False
        
    finally:
        api.state_manager.delete_state(state)


def main():
    parser = argparse.ArgumentParser(
        description='Run EnergyPlus simulation with specified IDF and weather files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sim.py --idf data/model.idf --epw data/weather.epw
  python run_sim.py -i data/model.idf -w data/weather.epw -o results/
        """
    )
    
    parser.add_argument(
        '--idf', '-i',
        required=True,
        help='Path to the IDF model file'
    )
    
    parser.add_argument(
        '--epw', '-w',
        required=True,
        help='Path to the EPW weather file'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    args = parser.parse_args()
    
    success = run_simulation(args.idf, args.epw, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
