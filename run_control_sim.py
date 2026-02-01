"""
Rule-Based Control Simulation for EnergyPlus

This script runs an EnergyPlus simulation with real-time control callbacks.
At each timestep, it modifies zone thermostat setpoints based on:
- Total building power consumption from the previous timestep
- Current outdoor air temperature

Usage:
    python run_control_sim.py
    python run_control_sim.py --idf path/to/model.idf --epw path/to/weather.epw
    python run_control_sim.py --no-control  # Run baseline without control
"""

import os
import sys
import argparse
import shutil
from datetime import datetime


class RuleBasedController:
    """
    Rule-based controller that adjusts zone setpoints based on:
    - Previous timestep total power consumption
    - Current outdoor air temperature
    """
    
    def __init__(self, api, state):
        self.api = api
        self.state = state
        
        # Control parameters
        self.power_threshold_high = 150000  # W - reduce cooling if above
        self.power_threshold_low = 50000    # W - can be more aggressive
        self.oat_hot_threshold = 30         # °C - hot outdoor conditions
        self.oat_cold_threshold = 5         # °C - cold outdoor conditions
        
        # Setpoint adjustments (°C)
        self.cooling_setpoint_base = 24.0
        self.heating_setpoint_base = 21.0
        self.max_cooling_adjustment = 2.0   # Max increase in cooling setpoint
        self.max_heating_adjustment = 2.0   # Max decrease in heating setpoint
        
        # State variables
        self.previous_power = 0.0
        self.current_cooling_setpoint = self.cooling_setpoint_base
        self.current_heating_setpoint = self.heating_setpoint_base
        
        # Handles (initialized during warmup)
        self.handles_initialized = False
        self.oat_handle = None
        self.power_handle = None
        self.zone_handles = {}  # zone_name -> (cooling_actuator, heating_actuator)
        
        # Logging
        self.log_data = []
        self.timestep_count = 0
        
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
        
        # Facility total electric demand (W)
        self.power_handle = exchange.get_meter_handle(
            self.state,
            "Electricity:Facility"
        )
        
        # Get zone names and create actuator handles for setpoints
        # We'll use EMS actuators to override thermostat setpoints
        zone_names = [
            "Core_bottom", "Core_mid", "Core_top",
            "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
            "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
            "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4"
        ]
        
        for zone_name in zone_names:
            # Actuator for zone thermostat cooling setpoint
            cooling_handle = exchange.get_actuator_handle(
                self.state,
                "Zone Temperature Control",
                "Cooling Setpoint",
                zone_name
            )
            
            # Actuator for zone thermostat heating setpoint
            heating_handle = exchange.get_actuator_handle(
                self.state,
                "Zone Temperature Control",
                "Heating Setpoint",
                zone_name
            )
            
            if cooling_handle > 0 and heating_handle > 0:
                self.zone_handles[zone_name] = (cooling_handle, heating_handle)
        
        # Check if handles are valid
        if self.oat_handle <= 0:
            print("Warning: Could not get outdoor air temperature handle")
            return False
            
        if self.power_handle <= 0:
            print("Warning: Could not get power meter handle")
            return False
            
        if not self.zone_handles:
            print("Warning: Could not get any zone actuator handles")
            return False
            
        print(f"Initialized handles for {len(self.zone_handles)} zones")
        self.handles_initialized = True
        return True
        
    def compute_setpoints(self, oat, power):
        """
        Compute new setpoints based on outdoor temperature and power consumption.
        
        Rule logic:
        1. If power is high AND outdoor temp is hot -> raise cooling setpoint (reduce cooling)
        2. If power is high AND outdoor temp is cold -> lower heating setpoint (reduce heating)
        3. If power is low -> can be more aggressive with conditioning
        4. Outdoor temp affects the aggressiveness of adjustments
        """
        cooling_adjustment = 0.0
        heating_adjustment = 0.0
        
        # Power-based adjustment
        if power > self.power_threshold_high:
            # High power - need to reduce consumption
            power_factor = min((power - self.power_threshold_high) / 100000, 1.0)
            
            if oat > self.oat_hot_threshold:
                # Hot outside, high power -> raise cooling setpoint
                cooling_adjustment = power_factor * self.max_cooling_adjustment
            elif oat < self.oat_cold_threshold:
                # Cold outside, high power -> lower heating setpoint
                heating_adjustment = -power_factor * self.max_heating_adjustment
            else:
                # Mild weather - smaller adjustments
                cooling_adjustment = power_factor * self.max_cooling_adjustment * 0.5
                heating_adjustment = -power_factor * self.max_heating_adjustment * 0.5
                
        elif power < self.power_threshold_low:
            # Low power - can be more aggressive with comfort
            if oat > self.oat_hot_threshold:
                # Hot outside, low power -> can cool more
                cooling_adjustment = -0.5
            elif oat < self.oat_cold_threshold:
                # Cold outside, low power -> can heat more
                heating_adjustment = 0.5
        
        # Apply adjustments
        new_cooling = self.cooling_setpoint_base + cooling_adjustment
        new_heating = self.heating_setpoint_base + heating_adjustment
        
        # Ensure deadband (cooling > heating)
        if new_cooling - new_heating < 2.0:
            new_cooling = new_heating + 2.0
            
        return new_cooling, new_heating
        
    def timestep_callback(self, state):
        """Called at each timestep to apply control actions."""
        self.timestep_count += 1
        
        # Skip if in warmup
        if self.api.exchange.warmup_flag(state):
            return
            
        # Initialize handles on first real timestep
        if not self.handles_initialized:
            if not self.initialize_handles():
                return
        
        exchange = self.api.exchange
        
        # Get current values
        oat = exchange.get_variable_value(state, self.oat_handle)
        power = exchange.get_meter_value(state, self.power_handle)
        
        # Compute new setpoints based on previous power and current OAT
        new_cooling, new_heating = self.compute_setpoints(oat, self.previous_power)
        
        # Apply setpoints to all zones
        for zone_name, (cooling_handle, heating_handle) in self.zone_handles.items():
            exchange.set_actuator_value(state, cooling_handle, new_cooling)
            exchange.set_actuator_value(state, heating_handle, new_heating)
        
        # Log data periodically (every 4 timesteps = hourly for 15-min timesteps)
        if self.timestep_count % 4 == 0:
            self.log_data.append({
                'timestep': self.timestep_count,
                'oat': oat,
                'power': power,
                'cooling_sp': new_cooling,
                'heating_sp': new_heating
            })
            
        # Update previous power for next timestep
        self.previous_power = power
        self.current_cooling_setpoint = new_cooling
        self.current_heating_setpoint = new_heating
        
    def get_summary(self):
        """Return summary of control actions."""
        if not self.log_data:
            return "No control data logged"
            
        powers = [d['power'] for d in self.log_data]
        cooling_sps = [d['cooling_sp'] for d in self.log_data]
        heating_sps = [d['heating_sp'] for d in self.log_data]
        
        return f"""
Control Summary:
  Total timesteps: {self.timestep_count}
  Zones controlled: {len(self.zone_handles)}
  
  Power (W):
    Min: {min(powers):,.0f}
    Max: {max(powers):,.0f}
    Avg: {sum(powers)/len(powers):,.0f}
    
  Cooling Setpoint (°C):
    Min: {min(cooling_sps):.1f}
    Max: {max(cooling_sps):.1f}
    
  Heating Setpoint (°C):
    Min: {min(heating_sps):.1f}
    Max: {max(heating_sps):.1f}
"""


def run_simulation(idf_path, epw_path, output_dir, enable_control=True):
    """Run EnergyPlus simulation with optional control."""
    from pyenergyplus.api import EnergyPlusAPI
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize API
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    
    controller = None
    if enable_control:
        controller = RuleBasedController(api, state)
        
        # Register callback for each timestep
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state,
            controller.timestep_callback
        )
        print("Control enabled - will adjust setpoints based on power and OAT")
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
        description='Run EnergyPlus simulation with rule-based control.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_control_sim.py                    # Run with default model and control
  python run_control_sim.py --no-control       # Run baseline without control
  python run_control_sim.py --idf model.idf --epw weather.epw
        """
    )
    
    parser.add_argument('--idf', type=str, 
                        default='energyplus/control_models/MediumOffice_Control.idf',
                        help='Path to IDF file')
    parser.add_argument('--epw', type=str,
                        default='weather/chicago/TMY_lat41.88_lon-87.63.epw',
                        help='Path to weather file')
    parser.add_argument('--output', type=str,
                        default='outputs/control_test',
                        help='Output directory')
    parser.add_argument('--no-control', action='store_true',
                        help='Run baseline simulation without control')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.idf):
        print(f"Error: IDF file not found: {args.idf}")
        return 1
        
    if not os.path.exists(args.epw):
        print(f"Error: Weather file not found: {args.epw}")
        return 1
    
    # Run simulation
    return run_simulation(
        args.idf,
        args.epw,
        args.output,
        enable_control=not args.no_control
    )


if __name__ == "__main__":
    sys.exit(main())
