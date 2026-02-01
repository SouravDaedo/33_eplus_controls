"""
External Rule-Based Controller

This script demonstrates how to use the EnergyPlus step environment
with control logic that lives completely outside the simulation.

The controller receives observations at each timestep and decides
what actions to take based on its own rules.

Usage:
    python external_controller.py
    python external_controller.py --max-steps 1000
"""

import os
import sys
import csv
import shutil
import argparse
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple, List


class RuleBasedController:
    """
    External rule-based controller.
    
    This class contains ALL the control logic, completely separate
    from the EnergyPlus simulation. It receives observations and
    returns actions.
    """
    
    def __init__(self):
        # Control parameters - tune these as needed
        self.power_threshold_high = 150000  # W
        self.power_threshold_low = 50000    # W
        self.oat_hot = 30.0                 # °C
        self.oat_cold = 5.0                 # °C
        
        # Base setpoints
        self.cooling_base = 24.0  # °C
        self.heating_base = 21.0  # °C
        
        # Adjustment limits
        self.max_cooling_adj = 2.0  # °C
        self.max_heating_adj = 2.0  # °C
        
        # State tracking
        self.previous_power = 0.0
        self.step_count = 0
        
    def compute_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute control action based on current observations.
        
        This is where YOUR control logic goes!
        
        Args:
            obs: Dictionary containing:
                - 'outdoor_temp': float
                - 'total_power': float
                - 'zone_temps': dict of zone temperatures
                - 'timestep': int
                
        Returns:
            action: Dictionary containing setpoints
        """
        self.step_count += 1
        
        # Extract observations
        oat = obs.get('outdoor_temp', 20.0)
        power = obs.get('total_power', 0.0)
        
        # Use previous timestep power for control decisions
        # (current power reflects previous setpoints)
        control_power = self.previous_power
        
        # Initialize adjustments
        cooling_adj = 0.0
        heating_adj = 0.0
        
        # ============================================
        # RULE-BASED CONTROL LOGIC - MODIFY AS NEEDED
        # ============================================
        
        if control_power > self.power_threshold_high:
            # High power consumption - need to reduce
            power_factor = min((control_power - self.power_threshold_high) / 100000, 1.0)
            
            if oat > self.oat_hot:
                # Hot outside + high power -> raise cooling setpoint (less cooling)
                cooling_adj = power_factor * self.max_cooling_adj
                
            elif oat < self.oat_cold:
                # Cold outside + high power -> lower heating setpoint (less heating)
                heating_adj = -power_factor * self.max_heating_adj
                
            else:
                # Mild weather - moderate adjustments to both
                cooling_adj = power_factor * self.max_cooling_adj * 0.5
                heating_adj = -power_factor * self.max_heating_adj * 0.5
                
        elif control_power < self.power_threshold_low:
            # Low power - can improve comfort
            if oat > self.oat_hot:
                # Hot outside, low power -> can cool more aggressively
                cooling_adj = -0.5
            elif oat < self.oat_cold:
                # Cold outside, low power -> can heat more
                heating_adj = 0.5
        
        # ============================================
        # END CONTROL LOGIC
        # ============================================
        
        # Calculate final setpoints
        cooling_sp = self.cooling_base + cooling_adj
        heating_sp = self.heating_base + heating_adj
        
        # Ensure deadband (cooling must be > heating)
        if cooling_sp - heating_sp < 2.0:
            cooling_sp = heating_sp + 2.0
        
        # Update state for next timestep
        self.previous_power = power
        
        # Return action
        return {
            'cooling_setpoint': cooling_sp,
            'heating_setpoint': heating_sp
        }
    
    def get_stats(self) -> str:
        """Return controller statistics."""
        return f"Controller ran for {self.step_count} steps"


def prepare_idf(
    idf_path: str,
    output_dir: str,
    start_month: int = None,
    start_day: int = None,
    end_month: int = None,
    end_day: int = None,
    timestep: int = None
) -> str:
    """
    Prepare IDF file with modified RunPeriod and Timestep using eppy.
    
    Args:
        idf_path: Path to source IDF file
        output_dir: Directory to save modified IDF
        start_month: Start month (1-12)
        start_day: Start day of month
        end_month: End month (1-12)
        end_day: End day of month
        timestep: Timesteps per hour (e.g., 4=15min, 6=10min, 12=5min)
        
    Returns:
        Path to modified IDF file
    """
    from eppy.modeleditor import IDF
    
    # Find IDD file (needed by eppy)
    try:
        from pyenergyplus.api import EnergyPlusAPI
        api = EnergyPlusAPI()
        # Get EnergyPlus install directory from the API
        import pyenergyplus
        eplus_dir = os.path.dirname(os.path.dirname(pyenergyplus.__file__))
        # Try common locations
        idd_paths = [
            os.path.join(eplus_dir, 'Energy+.idd'),
            'C:/EnergyPlusV23-2-0/Energy+.idd',
            'C:/EnergyPlusV24-1-0/Energy+.idd',
            'C:/EnergyPlusV24-2-0/Energy+.idd',
            'C:/EnergyPlusV25-1-0/Energy+.idd',
            'C:/EnergyPlusV25-2-0/Energy+.idd',
            'C:/EnergyPlusV22-1-0/Energy+.idd',
        ]
        idd_file = None
        for path in idd_paths:
            if os.path.exists(path):
                idd_file = path
                break
        if not idd_file:
            print("Warning: Could not find Energy+.idd, using IDF without modification")
            return idf_path
    except Exception as e:
        print(f"Warning: Could not locate IDD file: {e}")
        return idf_path
    
    # Set IDD file for eppy
    IDF.setiddname(idd_file)
    
    # Load IDF
    idf = IDF(idf_path)
    
    modified = False
    
    # Modify Timestep
    if timestep is not None:
        timesteps = idf.idfobjects['TIMESTEP']
        if timesteps:
            timesteps[0].Number_of_Timesteps_per_Hour = timestep
            print(f"  Timestep set to {timestep} per hour ({60//timestep} min intervals)")
            modified = True
    
    # Modify RunPeriod
    if any([start_month, start_day, end_month, end_day]):
        runperiods = idf.idfobjects['RUNPERIOD']
        if runperiods:
            rp = runperiods[0]
            if start_month is not None:
                rp.Begin_Month = start_month
            if start_day is not None:
                rp.Begin_Day_of_Month = start_day
            if end_month is not None:
                rp.End_Month = end_month
            if end_day is not None:
                rp.End_Day_of_Month = end_day
            print(f"  RunPeriod: {rp.Begin_Month}/{rp.Begin_Day_of_Month} to {rp.End_Month}/{rp.End_Day_of_Month}")
            modified = True
    
    if not modified:
        return idf_path
    
    # Save modified IDF
    os.makedirs(output_dir, exist_ok=True)
    modified_idf_path = os.path.join(output_dir, 'model_modified.idf')
    idf.saveas(modified_idf_path)
    print(f"  Saved modified IDF to: {modified_idf_path}")
    
    return modified_idf_path


def run_controlled_simulation(
    idf_path: str,
    epw_path: str,
    output_dir: str,
    max_steps: int = None,
    log_interval: int = 100,
    save_csv: bool = True,
    start_month: int = None,
    start_day: int = None,
    end_month: int = None,
    end_day: int = None,
    timestep: int = None
):
    """
    Run simulation with external controller.
    
    Args:
        idf_path: Path to IDF file
        epw_path: Path to weather file
        output_dir: Output directory
        max_steps: Maximum steps to run (None = full simulation)
        log_interval: How often to print status
        save_csv: Whether to save observations to CSV
        start_month: Start month (1-12)
        start_day: Start day of month
        end_month: End month (1-12)
        end_day: End day of month
        timestep: Timesteps per hour
    """
    # Import the environment
    from eplus_env import EnergyPlusEnv
    
    print("=" * 60)
    print("EXTERNAL CONTROLLER SIMULATION")
    print("=" * 60)
    print(f"\nIDF: {idf_path}")
    print(f"EPW: {epw_path}")
    print(f"Output: {output_dir}")
    if max_steps:
        print(f"Max steps: {max_steps}")
    print()
    
    # Prepare IDF with modifications if needed
    if any([start_month, start_day, end_month, end_day, timestep]):
        print("\nModifying IDF settings:")
        idf_path = prepare_idf(
            idf_path, output_dir,
            start_month, start_day, end_month, end_day, timestep
        )
    
    # Create environment and controller
    env = EnergyPlusEnv(idf_path, epw_path, output_dir)
    controller = RuleBasedController()
    
    # Track metrics
    total_power = 0.0
    min_power = float('inf')
    max_power = 0.0
    cooling_sps = []
    heating_sps = []
    
    # Data logging
    log_data: List[Dict] = []
    
    try:
        # Reset environment to start simulation
        print("Starting simulation...")
        obs = env.reset()
        step = 0
        
        while not env.done:
            # Check max steps
            if max_steps and step >= max_steps:
                print(f"\nReached max steps ({max_steps})")
                break
            
            # =============================================
            # CONTROLLER DECIDES ACTION BASED ON OBSERVATIONS
            # =============================================
            action = controller.compute_action(obs)
            
            # =============================================
            # ENVIRONMENT EXECUTES ACTION AND RETURNS NEW STATE
            # =============================================
            obs, reward, done, info = env.step(action)
            
            step += 1
            
            # Track metrics
            power = obs.get('total_power', 0)
            total_power += power
            min_power = min(min_power, power)
            max_power = max(max_power, power)
            cooling_sps.append(action['cooling_setpoint'])
            heating_sps.append(action['heating_setpoint'])
            
            # Log observation data
            if save_csv:
                # Build datetime from simulation
                year = obs.get('year', 2007)
                month = obs.get('month', 1)
                day = obs.get('day', 1)
                hour = obs.get('hour', 0)
                minute = obs.get('minute', 0)
                # Handle EnergyPlus returning minute=60 at end of hour
                if minute >= 60:
                    minute = 0
                    hour += 1
                if hour >= 24:
                    hour = 0
                    day += 1
                # Create proper ISO timestamp
                try:
                    timestamp = datetime(year, month, day, hour, minute, 0)
                except ValueError:
                    # Fallback for edge cases (month overflow, etc.)
                    timestamp = datetime(year, month, 1, hour, minute, 0)
                
                row = {
                    'timestamp': timestamp.isoformat(),
                    'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestep': step,
                    'outdoor_temp_C': obs.get('outdoor_temp', 0),
                    'total_power_W': power,
                    'cooling_setpoint_C': action['cooling_setpoint'],
                    'heating_setpoint_C': action['heating_setpoint'],
                }
                # Add zone temperatures
                for zone, temp in obs.get('zone_temps', {}).items():
                    row[f'zone_temp_{zone}_C'] = temp
                log_data.append(row)
            
            # Log progress
            if step % log_interval == 0:
                oat = obs.get('outdoor_temp', 0)
                print(f"Step {step:6d}: OAT={oat:5.1f}°C, "
                      f"Power={power/1000:8.1f}kW, "
                      f"Cool={action['cooling_setpoint']:.1f}°C, "
                      f"Heat={action['heating_setpoint']:.1f}°C")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"\nTotal steps: {step}")
        print(f"\nPower Statistics:")
        print(f"  Min:  {min_power/1000:,.1f} kW")
        print(f"  Max:  {max_power/1000:,.1f} kW")
        print(f"  Avg:  {total_power/step/1000:,.1f} kW")
        print(f"\nSetpoint Ranges:")
        print(f"  Cooling: {min(cooling_sps):.1f} - {max(cooling_sps):.1f} °C")
        print(f"  Heating: {min(heating_sps):.1f} - {max(heating_sps):.1f} °C")
        print(f"\n{controller.get_stats()}")
        
        # Save CSV
        if save_csv and log_data:
            csv_path = os.path.join(output_dir, 'simulation_log.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
                writer.writeheader()
                writer.writerows(log_data)
            print(f"\nSaved {len(log_data)} timesteps to: {csv_path}")
        
    finally:
        env.close()
    
    return step


def main():
    parser = argparse.ArgumentParser(
        description='Run EnergyPlus with external rule-based controller.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python external_controller.py                              # Full year simulation
  python external_controller.py --max-steps 1000             # Run 1000 steps only
  python external_controller.py --start-date 7/1 --end-date 7/31  # July only
  python external_controller.py --timestep 4                 # 15-minute intervals
  python external_controller.py --timestep 12 --start-date 1/1 --end-date 1/7  # First week, 5-min
        """
    )
    
    parser.add_argument('--idf', type=str,
                        default='energyplus/control_models/MediumOffice_Control.idf',
                        help='Path to IDF file')
    parser.add_argument('--epw', type=str,
                        default='weather/chicago/TMY_lat41.88_lon-87.63.epw',
                        help='Path to weather file')
    parser.add_argument('--output', type=str,
                        default='outputs/external_control',
                        help='Output directory')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum simulation steps (default: full simulation)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Steps between log messages')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date as M/D (e.g., 7/1 for July 1)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date as M/D (e.g., 7/31 for July 31)')
    parser.add_argument('--timestep', type=int, default=None,
                        help='Timesteps per hour (4=15min, 6=10min, 12=5min)')
    
    args = parser.parse_args()
    
    # Parse dates
    start_month, start_day = None, None
    end_month, end_day = None, None
    if args.start_date:
        parts = args.start_date.split('/')
        start_month, start_day = int(parts[0]), int(parts[1])
    if args.end_date:
        parts = args.end_date.split('/')
        end_month, end_day = int(parts[0]), int(parts[1])
    
    # Validate paths
    if not os.path.exists(args.idf):
        print(f"Error: IDF not found: {args.idf}")
        return 1
    if not os.path.exists(args.epw):
        print(f"Error: EPW not found: {args.epw}")
        return 1
    
    # Run simulation
    run_controlled_simulation(
        args.idf,
        args.epw,
        args.output,
        args.max_steps,
        args.log_interval,
        save_csv=True,
        start_month=start_month,
        start_day=start_day,
        end_month=end_month,
        end_day=end_day,
        timestep=args.timestep
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
