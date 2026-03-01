"""
IAQ Control Simulation for EnergyPlus

This script runs an EnergyPlus simulation with real-time control of:
- Heating and cooling setpoints
- Zone occupancy (number of people)
- Outdoor air flow rate

It tracks CO2 PPM at each timestep to monitor indoor air quality.

Usage:
    python test_iaq_control_sim.py
    python test_iaq_control_sim.py --idf path/to/model.idf --epw path/to/weather.epw
    python test_iaq_control_sim.py --no-control  # Run baseline without control
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import HVAC configuration
from src.utils.hvac_config import get_hvac_config


class IAQController:
    """
    Controller that manages:
    - Heating/cooling setpoints
    - Zone occupancy levels
    - Outdoor air flow rates
    
    And tracks CO2 PPM for indoor air quality monitoring.
    """
    
    def __init__(self, api, state, config_path=None):
        self.api = api
        self.state = state
        
        # Load HVAC configuration
        self.hvac_config = get_hvac_config(config_path)
        
        # Control parameters from config
        self.base_temp = self.hvac_config.base_temperature
        self.temp_range = 2.0   # Temperature range for setpoints (°C)
        self.min_occupancy = 0.0  # Minimum people per zone
        self.max_occupancy = 10.0  # Maximum people per zone
        self.min_oa_multiplier = self.hvac_config.min_airflow * 100  # Convert to percentage
        self.max_oa_multiplier = self.hvac_config.max_airflow * 100  # Convert to percentages
        self.oa_multiplier = 1.0  # Can be adjusted based on CO2 levels
        
        # CO2 thresholds for control
        self.co2_target = 800  # ppm - target CO2 level
        self.co2_high = 1000   # ppm - increase OA if above this
        self.co2_low = 600     # ppm - can reduce OA if below this
        
        # State variables
        self.current_hour = 0
        self.current_occupancy_fraction = 0.0
        
        # Handles (initialized during warmup)
        self.handles_initialized = False
        self.oat_handle = None
        self.zone_handles = {}  # zone_name -> dict of handles
        
        # Logging
        self.log_data = []
        self.timestep_count = 0
        
    def _create_occupancy_schedule(self):
        """Create hourly occupancy schedule (fraction of design occupancy)."""
        # Typical office occupancy pattern
        schedule = {
            0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
            6: 0.1, 7: 0.3, 8: 0.8, 9: 1.0, 10: 1.0, 11: 1.0,
            12: 0.8, 13: 0.9, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.5,
            18: 0.2, 19: 0.1, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0
        }
        return schedule
        
    def initialize_handles(self):
        """Initialize EnergyPlus data exchange handles."""
        if self.handles_initialized:
            return True
            
        exchange = self.api.exchange
        
        # Outdoor air temperature
        self.oat_handle = exchange.get_variable_handle(
            self.state,
            "Site Outdoor Air Drybulb Temperature",
            "Environment"
        )
        
        # Hour of day for schedule
        self.hour_handle = exchange.get_variable_handle(
            self.state,
            "Site Simple Factor Model Ground Temperature",  # Placeholder - will use API time
            "Environment"
        )
        
        # Zone names for the medium office model
        zone_names = [
            "Core_bottom", "Core_mid", "Core_top",
            "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
            "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
            "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4"
        ]
        
        for zone_name in zone_names:
            zone_handles = {}
            
            # Thermostat setpoint actuators
            cooling_handle = exchange.get_actuator_handle(
                self.state,
                "Zone Temperature Control",
                "Cooling Setpoint",
                zone_name
            )
            heating_handle = exchange.get_actuator_handle(
                self.state,
                "Zone Temperature Control",
                "Heating Setpoint",
                zone_name
            )
            
            # People/Occupancy actuator (number of people)
            people_handle = exchange.get_actuator_handle(
                self.state,
                "People",
                "Number of People",
                f"{zone_name} People"  # Typical naming convention
            )
            
            # Outdoor air actuator (for zone-level OA control)
            # This depends on the HVAC system configuration
            oa_handle = exchange.get_actuator_handle(
                self.state,
                "Zone Infiltration",
                "Air Exchange Flow Rate",
                f"{zone_name} Infiltration"
            )
            
            # CO2 concentration variable
            co2_handle = exchange.get_variable_handle(
                self.state,
                "Zone Air CO2 Concentration",
                zone_name
            )
            
            # Zone temperature variable
            temp_handle = exchange.get_variable_handle(
                self.state,
                "Zone Mean Air Temperature",
                zone_name
            )
            
            # Store handles
            zone_handles['cooling'] = cooling_handle
            zone_handles['heating'] = heating_handle
            zone_handles['people'] = people_handle
            zone_handles['oa'] = oa_handle
            zone_handles['co2'] = co2_handle
            zone_handles['temp'] = temp_handle
            
            # Only add zone if we have at least thermostat handles
            if cooling_handle > 0 and heating_handle > 0:
                self.zone_handles[zone_name] = zone_handles
        
        # Check if handles are valid
        if self.oat_handle <= 0:
            print("Warning: Could not get outdoor air temperature handle")
            
        if not self.zone_handles:
            print("Warning: Could not get any zone actuator handles")
            return False
            
        print(f"Initialized handles for {len(self.zone_handles)} zones")
        
        # Report which handles were found
        sample_zone = list(self.zone_handles.keys())[0]
        sample_handles = self.zone_handles[sample_zone]
        print(f"  Thermostat handles: {'✓' if sample_handles['cooling'] > 0 else '✗'}")
        print(f"  People handles: {'✓' if sample_handles['people'] > 0 else '✗'}")
        print(f"  OA handles: {'✓' if sample_handles['oa'] > 0 else '✗'}")
        print(f"  CO2 handles: {'✓' if sample_handles['co2'] > 0 else '✗'}")
        
        self.handles_initialized = True
        return True
        
    def get_current_hour(self):
        """Get current simulation hour."""
        return self.api.exchange.hour(self.state)
        
    def compute_control_actions(self, hour, avg_co2):
        """
        Compute control actions based on time of day and CO2 levels.
        
        Returns:
            tuple: (cooling_sp, heating_sp, occupancy_fraction, oa_multiplier)
        """
        # Get occupancy from schedule
        occupancy_fraction = self.occupancy_schedule.get(hour, 0.0)
        
        # Adjust outdoor air based on CO2 levels
        if avg_co2 > self.co2_high:
            # High CO2 - increase outdoor air
            oa_multiplier = min(2.0, 1.0 + (avg_co2 - self.co2_high) / 500)
        elif avg_co2 < self.co2_low:
            # Low CO2 - can reduce outdoor air to save energy
            oa_multiplier = max(0.5, avg_co2 / self.co2_low)
        else:
            # Normal range
            oa_multiplier = 1.0
            
        # Setpoints - could be adjusted based on occupancy
        if occupancy_fraction < 0.1:
            # Unoccupied - widen deadband
            cooling_sp = self.cooling_setpoint_base + 2.0
            heating_sp = self.heating_setpoint_base - 2.0
        else:
            # Occupied - normal setpoints
            cooling_sp = self.cooling_setpoint_base
            heating_sp = self.heating_setpoint_base
            
        return cooling_sp, heating_sp, occupancy_fraction, oa_multiplier
        
    def timestep_callback(self, state):
        """Called at each timestep to apply control actions and log data."""
        self.timestep_count += 1
        
        # Skip if in warmup
        if self.api.exchange.warmup_flag(state):
            return
            
        # Initialize handles on first real timestep
        if not self.handles_initialized:
            if not self.initialize_handles():
                return
        
        exchange = self.api.exchange
        
        # Get current time
        hour = self.get_current_hour()
        
        # Get outdoor air temperature
        oat = exchange.get_variable_value(state, self.oat_handle) if self.oat_handle > 0 else 0.0
        
        # Collect CO2 readings from all zones
        co2_readings = []
        temp_readings = []
        
        for zone_name, handles in self.zone_handles.items():
            if handles['co2'] > 0:
                co2 = exchange.get_variable_value(state, handles['co2'])
                if co2 > 0:  # Valid reading
                    co2_readings.append(co2)
            if handles['temp'] > 0:
                temp = exchange.get_variable_value(state, handles['temp'])
                temp_readings.append(temp)
        
        # Calculate average CO2 (use 400 ppm as default if no readings)
        avg_co2 = sum(co2_readings) / len(co2_readings) if co2_readings else 400.0
        avg_temp = sum(temp_readings) / len(temp_readings) if temp_readings else 22.0
        
        # Compute control actions
        cooling_sp, heating_sp, occupancy_frac, oa_mult = self.compute_control_actions(hour, avg_co2)
        
        # Apply control actions to all zones
        for zone_name, handles in self.zone_handles.items():
            # Set thermostat setpoints
            if handles['cooling'] > 0:
                exchange.set_actuator_value(state, handles['cooling'], cooling_sp)
            if handles['heating'] > 0:
                exchange.set_actuator_value(state, handles['heating'], heating_sp)
                
            # Set occupancy (if handle available)
            # Note: This requires the IDF to have EMS-controllable people objects
            if handles['people'] > 0:
                # Assuming design occupancy of ~10 people per zone for medium office
                design_people = 10.0
                exchange.set_actuator_value(state, handles['people'], design_people * occupancy_frac)
                
            # Set outdoor air flow (if handle available)
            if handles['oa'] > 0:
                base_oa_flow = 0.01  # m³/s base flow
                exchange.set_actuator_value(state, handles['oa'], base_oa_flow * oa_mult)
        
        # Log data every timestep
        self.log_data.append({
            'timestep': self.timestep_count,
            'hour': hour,
            'oat': oat,
            'avg_co2_ppm': avg_co2,
            'max_co2_ppm': max(co2_readings) if co2_readings else 0,
            'min_co2_ppm': min(co2_readings) if co2_readings else 0,
            'avg_temp': avg_temp,
            'cooling_sp': cooling_sp,
            'heating_sp': heating_sp,
            'occupancy_fraction': occupancy_frac,
            'oa_multiplier': oa_mult
        })
        
        # Print periodic updates
        if self.timestep_count % 24 == 0:  # Every ~6 hours for 15-min timesteps
            print(f"  Step {self.timestep_count:5d} | Hour {hour:2d} | "
                  f"CO2: {avg_co2:6.0f} ppm | Temp: {avg_temp:5.1f}°C | "
                  f"Occ: {occupancy_frac:.0%} | OA mult: {oa_mult:.2f}")
        
        # Store current state
        self.current_hour = hour
        self.current_occupancy_fraction = occupancy_frac
        
    def get_summary(self):
        """Return summary of control actions and IAQ metrics."""
        if not self.log_data:
            return "No control data logged"
            
        co2_values = [d['avg_co2_ppm'] for d in self.log_data if d['avg_co2_ppm'] > 0]
        temps = [d['avg_temp'] for d in self.log_data]
        oa_mults = [d['oa_multiplier'] for d in self.log_data]
        
        # Count time above CO2 thresholds
        time_above_800 = sum(1 for co2 in co2_values if co2 > 800)
        time_above_1000 = sum(1 for co2 in co2_values if co2 > 1000)
        
        summary = f"""
IAQ Control Summary:
  Total timesteps: {self.timestep_count}
  Zones controlled: {len(self.zone_handles)}
  
  CO2 Concentration (ppm):
    Min: {min(co2_values):,.0f}
    Max: {max(co2_values):,.0f}
    Avg: {sum(co2_values)/len(co2_values):,.0f}
    Time > 800 ppm: {time_above_800} timesteps ({100*time_above_800/len(co2_values):.1f}%)
    Time > 1000 ppm: {time_above_1000} timesteps ({100*time_above_1000/len(co2_values):.1f}%)
    
  Zone Temperature (°C):
    Min: {min(temps):.1f}
    Max: {max(temps):.1f}
    Avg: {sum(temps)/len(temps):.1f}
    
  Outdoor Air Multiplier:
    Min: {min(oa_mults):.2f}
    Max: {max(oa_mults):.2f}
    Avg: {sum(oa_mults)/len(oa_mults):.2f}
"""
        return summary
        
    def save_log_to_csv(self, filepath):
        """Save logged data to CSV file."""
        if not self.log_data:
            print("No data to save")
            return
            
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
            
        print(f"Saved {len(self.log_data)} timesteps to {filepath}")


def run_simulation(idf_path, epw_path, output_dir, enable_control=True, config_path=None):
    """Run EnergyPlus simulation with IAQ control."""
    from pyenergyplus.api import EnergyPlusAPI
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize API
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    
    controller = None
    if enable_control:
        controller = IAQController(api, state, config_path)
        
        # Register callback for each timestep
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state,
            controller.timestep_callback
        )
        print("IAQ Control enabled - will control setpoints, occupancy, and outdoor air")
        print("Tracking CO2 PPM at each timestep")
    else:
        print("Control disabled - running baseline simulation")
    
    # Run simulation
    print(f"\nRunning simulation...")
    print(f"  IDF: {idf_path}")
    print(f"  EPW: {epw_path}")
    print(f"  Output: {output_dir}")
    print()
    
    args = [
        '-w', epw_path,
        '-d', output_dir,
        idf_path
    ]
    
    exit_code = api.runtime.run_energyplus(state, args)
    
    # Clean up
    api.state_manager.delete_state(state)
    
    # Print results
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("SIMULATION COMPLETED SUCCESSFULLY")
    else:
        print(f"SIMULATION FAILED (exit code: {exit_code})")
    print("=" * 60)
    
    if controller:
        print(controller.get_summary())
        
        # Save log data
        log_path = os.path.join(output_dir, 'iaq_control_log.csv')
        controller.save_log_to_csv(log_path)
        
    # Check output files
    err_file = os.path.join(output_dir, 'eplusout.err')
    if os.path.exists(err_file):
        with open(err_file, 'r') as f:
            content = f.read()
            severe_count = content.count('** Severe  **')
            warning_count = content.count('** Warning **')
            print(f"\nError file summary:")
            print(f"  Warnings: {warning_count}")
            print(f"  Severe errors: {severe_count}")
    
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description='Run EnergyPlus simulation with IAQ control (occupancy, outdoor air, CO2 tracking).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_iaq_control_sim.py                    # Run with default model and control
  python test_iaq_control_sim.py --no-control       # Run baseline without control
  python test_iaq_control_sim.py --idf model.idf --epw weather.epw
        """
    )
    
    parser.add_argument('--idf', type=str, 
                        default='energyplus/control_models/MediumOffice_IAQ.idf',
                        help='Path to IDF file')
    parser.add_argument('--config', type=str, 
                        default='config/hvac_config.yaml',
                        help='Path to HVAC configuration file')
    parser.add_argument('--epw', type=str,
                        default='weather/chicago/TMY_lat41.88_lon-87.63.epw',
                        help='Path to weather file')
    parser.add_argument('--output', type=str,
                        default='outputs/iaq_control_test',
                        help='Output directory')
    parser.add_argument('--no-control', action='store_true',
                        help='Disable control, run baseline simulation')
    
    args = parser.parse_args()
    
    # Handle relative paths from project root
    if not os.path.isabs(args.idf):
        args.idf = str(project_root / args.idf)
    if not os.path.isabs(args.config):
        args.config = str(project_root / args.config)
    if not os.path.isabs(args.epw):
        args.epw = str(project_root / args.epw)
    if not os.path.isabs(args.output):
        args.output = str(project_root / args.output)
    
    # Validate inputs
    if not os.path.exists(args.idf):
        print(f"Error: IDF file not found: {args.idf}")
        return 1
        
    if not os.path.exists(args.epw):
        print(f"Error: Weather file not found: {args.epw}")
        return 1

    # Run simulation
    exit_code = run_simulation(args.idf, args.epw, args.output, not args.no_control, args.config)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
