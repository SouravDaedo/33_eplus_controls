"""
Integrated Building Energy Simulation

Combines live EnergyPlus simulation with:
- Solar PV model
- Battery storage model  
- Real-time electricity pricing

The EnergyPlus simulation runs step-by-step, providing real building load
at each timestep. The controller optimizes battery charging/discharging
based on PV production and electricity prices.

Usage:
    python integrated_simulation.py
    python integrated_simulation.py --idf models/your_model.idf --epw weather/your_weather.epw
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from eplus_env import EnergyPlusEnv
from battery_model import create_battery, BatteryAction, BatteryModel
from solar_pv_model import create_pv_system, SolarPVModel
from energy_price_model import (
    EnergyPriceModel, PricingConfig, PricingType, TOUPeriod,
    create_tou_pricing, create_rtp_pricing
)


class IntegratedController:
    """
    Integrated controller for building + PV + battery optimization.
    
    Receives building load from EnergyPlus at each timestep and decides:
    1. HVAC setpoints (demand response)
    2. Battery charge/discharge actions
    3. Grid import/export
    """
    
    def __init__(
        self,
        pv_system: Optional[SolarPVModel] = None,
        battery: Optional[BatteryModel] = None,
        price_model: Optional[EnergyPriceModel] = None
    ):
        self.pv = pv_system
        self.battery = battery
        self.price_model = price_model
        
        # HVAC control parameters
        self.cooling_base = 24.0
        self.heating_base = 21.0
        self.max_adjustment = 2.0
        
        # Tracking
        self.step_count = 0
        self.results: List[Dict] = []
        
        # Cumulative costs
        self.cumulative_import_cost = 0.0
        self.cumulative_export_credit = 0.0
    
    def step(
        self,
        obs: Dict[str, Any],
        simulation_datetime: datetime
    ) -> Dict[str, Any]:
        """
        Process one simulation timestep.
        
        Args:
            obs: EnergyPlus observations (outdoor_temp, total_power, zone_temps)
            simulation_datetime: Current simulation datetime
            
        Returns:
            Dictionary with HVAC actions and step results
        """
        self.step_count += 1
        
        # Extract building load from EnergyPlus (convert W to kW)
        building_load_kw = obs.get('total_power', 0) / 1000.0
        outdoor_temp = obs.get('outdoor_temp', 20.0)
        
        # Get PV production for this timestep
        pv_power_kw = 0.0
        if self.pv is not None:
            try:
                pv_state = self.pv.get_power_at_timestep(simulation_datetime)
                pv_power_kw = pv_state.ac_power_kw
            except Exception:
                pv_power_kw = 0.0
        
        # Get current electricity price
        current_price = 0.12  # Default
        is_peak = False
        if self.price_model is not None:
            price_state = self.price_model.get_price(simulation_datetime)
            current_price = price_state.price_per_kwh
            is_peak = price_state.is_peak
        
        # Calculate net load (building - PV)
        net_load_kw = building_load_kw - pv_power_kw
        
        # Battery control strategy
        battery_action = BatteryAction.IDLE
        battery_power = 0.0
        battery_soc = 0.5
        
        if self.battery is not None:
            battery_soc = self.battery.get_soc()
            
            if net_load_kw < 0:
                # Excess PV - charge battery
                battery_action = BatteryAction.CHARGE_FROM_PV
                battery_power = min(abs(net_load_kw), self.battery.get_available_charge_power())
            elif is_peak and battery_soc > 0.2:
                # Peak hours - discharge to reduce grid import
                battery_action = BatteryAction.DISCHARGE_TO_LOAD
                battery_power = min(net_load_kw, self.battery.get_available_discharge_power())
            elif not is_peak and battery_soc < 0.8:
                # Off-peak - charge from grid (pre-charge for peak)
                hour = simulation_datetime.hour
                if hour < 14:  # Only charge before peak
                    battery_action = BatteryAction.CHARGE_FROM_GRID
                    battery_power = min(30, self.battery.get_available_charge_power())
            
            # Execute battery action
            batt_result = self.battery.step(
                action=battery_action,
                power_kw=battery_power,
                pv_available_kw=max(0, -net_load_kw),
                load_demand_kw=max(0, net_load_kw)
            )
            battery_soc = batt_result.soc_after
            battery_power = batt_result.power_actual_kw
        
        # Calculate grid interaction
        if battery_action == BatteryAction.DISCHARGE_TO_LOAD:
            grid_import_kw = max(0, net_load_kw - battery_power)
            grid_export_kw = 0
        elif battery_action == BatteryAction.CHARGE_FROM_GRID:
            grid_import_kw = max(0, net_load_kw) + battery_power
            grid_export_kw = 0
        elif battery_action == BatteryAction.CHARGE_FROM_PV:
            grid_import_kw = max(0, net_load_kw)
            grid_export_kw = max(0, -net_load_kw - battery_power)
        else:
            grid_import_kw = max(0, net_load_kw)
            grid_export_kw = max(0, -net_load_kw)
        
        # Calculate energy cost (assuming 1-hour timestep, adjust as needed)
        timestep_hours = 1.0  # Adjust based on EnergyPlus timestep
        import_cost = grid_import_kw * timestep_hours * current_price
        export_credit = grid_export_kw * timestep_hours * (current_price * 0.4)  # Feed-in at 40% of retail
        
        self.cumulative_import_cost += import_cost
        self.cumulative_export_credit += export_credit
        
        # HVAC demand response (optional)
        cooling_sp = self.cooling_base
        heating_sp = self.heating_base
        
        if is_peak and building_load_kw > 100:
            # High load during peak - raise cooling setpoint
            cooling_sp = self.cooling_base + 1.0
        
        # Store results
        result = {
            'timestamp': simulation_datetime,
            'hour': simulation_datetime.hour,
            'building_load_kw': building_load_kw,
            'pv_power_kw': pv_power_kw,
            'net_load_kw': net_load_kw,
            'battery_action': battery_action.name,
            'battery_power_kw': battery_power,
            'battery_soc': battery_soc,
            'grid_import_kw': grid_import_kw,
            'grid_export_kw': grid_export_kw,
            'price_per_kwh': current_price,
            'is_peak': is_peak,
            'import_cost': import_cost,
            'export_credit': export_credit,
            'cumulative_cost': self.cumulative_import_cost - self.cumulative_export_credit,
            'outdoor_temp': outdoor_temp
        }
        self.results.append(result)
        
        # Return HVAC actions for EnergyPlus
        return {
            'cooling_setpoint': cooling_sp,
            'heating_setpoint': heating_sp,
            'result': result
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get simulation results as DataFrame."""
        return pd.DataFrame(self.results)
    
    def get_summary(self) -> Dict:
        """Get simulation summary."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        return {
            'total_steps': len(self.results),
            'total_building_load_kwh': df['building_load_kw'].sum(),
            'total_pv_generation_kwh': df['pv_power_kw'].sum(),
            'total_grid_import_kwh': df['grid_import_kw'].sum(),
            'total_grid_export_kwh': df['grid_export_kw'].sum(),
            'total_import_cost': self.cumulative_import_cost,
            'total_export_credit': self.cumulative_export_credit,
            'net_cost': self.cumulative_import_cost - self.cumulative_export_credit,
            'peak_building_load_kw': df['building_load_kw'].max(),
            'peak_pv_power_kw': df['pv_power_kw'].max(),
        }


def run_integrated_simulation(
    idf_path: str,
    epw_path: str,
    output_dir: str = "outputs/integrated_sim",
    pv_capacity_kw: float = 100.0,
    battery_capacity_kwh: float = 200.0,
    battery_power_kw: float = 50.0,
    max_steps: Optional[int] = None,
    log_interval: int = 12,
    save_results: bool = True
):
    """
    Run integrated simulation with live EnergyPlus + PV + Battery + Pricing.
    
    Args:
        idf_path: Path to EnergyPlus IDF model
        epw_path: Path to weather file (EPW)
        output_dir: Output directory
        pv_capacity_kw: PV system capacity in kW
        battery_capacity_kwh: Battery capacity in kWh
        battery_power_kw: Battery max power in kW
        max_steps: Maximum simulation steps (None = full simulation)
        log_interval: Print status every N steps
        save_results: Save results to CSV
    """
    print("=" * 70)
    print("INTEGRATED BUILDING ENERGY SIMULATION")
    print("EnergyPlus + Solar PV + Battery + Real-Time Pricing")
    print("=" * 70)
    
    # Validate paths
    if not os.path.exists(idf_path):
        print(f"ERROR: IDF file not found: {idf_path}")
        return
    if not os.path.exists(epw_path):
        print(f"ERROR: EPW file not found: {epw_path}")
        return
    
    print(f"\nConfiguration:")
    print(f"  IDF Model: {idf_path}")
    print(f"  Weather: {epw_path}")
    print(f"  PV System: {pv_capacity_kw} kW")
    print(f"  Battery: {battery_capacity_kwh} kWh / {battery_power_kw} kW")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print("\nInitializing models...")
    
    # PV System
    pv = create_pv_system(capacity_kw=pv_capacity_kw)
    try:
        pv.load_weather(epw_path)
        print(f"  PV model loaded with weather from {epw_path}")
    except Exception as e:
        print(f"  Warning: Could not load weather for PV: {e}")
        pv = None
    
    # Battery
    battery = create_battery(
        capacity_kwh=battery_capacity_kwh,
        max_power_kw=battery_power_kw,
        efficiency=0.90,
        timestep_minutes=60  # Adjust based on EnergyPlus timestep
    )
    print(f"  Battery model initialized: {battery_capacity_kwh} kWh")
    
    # Pricing model (TOU with peak 2-8 PM)
    price_config = PricingConfig(
        pricing_type=PricingType.TOU,
        tou_periods=[
            TOUPeriod("off-peak", 0, 8, 0.06, [0,1,2,3,4,5,6]),
            TOUPeriod("mid-peak", 8, 14, 0.12, [0,1,2,3,4,5,6]),
            TOUPeriod("peak", 14, 20, 0.28, [0,1,2,3,4,5,6]),
            TOUPeriod("mid-peak", 20, 24, 0.12, [0,1,2,3,4,5,6]),
        ],
        feed_in_tariff=0.05
    )
    price_model = EnergyPriceModel(price_config)
    print(f"  Pricing model: TOU (peak $0.28, off-peak $0.06)")
    
    # Create integrated controller
    controller = IntegratedController(
        pv_system=pv,
        battery=battery,
        price_model=price_model
    )
    
    # Create EnergyPlus environment
    print("\nStarting EnergyPlus simulation...")
    env = EnergyPlusEnv(idf_path, epw_path, output_dir)
    
    try:
        # Reset environment to start simulation
        obs = env.reset()
        step = 0
        
        print("\nSimulation running...")
        print("-" * 70)
        print(f"{'Step':>6} {'Time':<12} {'Bldg kW':>8} {'PV kW':>8} "
              f"{'Batt':>8} {'SOC':>6} {'Grid':>8} {'Price':>7} {'Cost':>8}")
        print("-" * 70)
        
        while not env.done:
            # Check max steps
            if max_steps and step >= max_steps:
                print(f"\nReached max steps ({max_steps})")
                break
            
            # Build simulation datetime from EnergyPlus observations
            year = obs.get('year', 2007)
            month = obs.get('month', 1)
            day = obs.get('day', 1)
            hour = obs.get('hour', 0)
            minute = obs.get('minute', 0)
            
            # Handle EnergyPlus edge cases
            if minute >= 60:
                minute = 0
                hour += 1
            if hour >= 24:
                hour = 0
            
            try:
                sim_datetime = datetime(year, month, day, hour, minute)
            except ValueError:
                sim_datetime = datetime(year, month, 1, hour, minute)
            
            # Controller processes this timestep
            action = controller.step(obs, sim_datetime)
            result = action['result']
            
            # Log progress
            if step % log_interval == 0:
                print(f"{step:>6} {sim_datetime.strftime('%m/%d %H:%M'):<12} "
                      f"{result['building_load_kw']:>8.1f} {result['pv_power_kw']:>8.1f} "
                      f"{result['battery_action'][:8]:>8} {result['battery_soc']:>5.1%} "
                      f"{result['grid_import_kw']-result['grid_export_kw']:>+8.1f} "
                      f"${result['price_per_kwh']:>5.2f} ${result['cumulative_cost']:>7.2f}")
            
            # Send HVAC actions to EnergyPlus and get next observation
            obs, reward, done, info = env.step({
                'cooling_setpoint': action['cooling_setpoint'],
                'heating_setpoint': action['heating_setpoint']
            })
            
            step += 1
        
        # Simulation complete
        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        
        # Print summary
        summary = controller.get_summary()
        print(f"\nResults Summary:")
        print(f"  Total Steps: {summary.get('total_steps', 0)}")
        print(f"  Building Load: {summary.get('total_building_load_kwh', 0):,.0f} kWh")
        print(f"  PV Generation: {summary.get('total_pv_generation_kwh', 0):,.0f} kWh")
        print(f"  Grid Import: {summary.get('total_grid_import_kwh', 0):,.0f} kWh")
        print(f"  Grid Export: {summary.get('total_grid_export_kwh', 0):,.0f} kWh")
        print(f"  Import Cost: ${summary.get('total_import_cost', 0):,.2f}")
        print(f"  Export Credit: ${summary.get('total_export_credit', 0):,.2f}")
        print(f"  Net Cost: ${summary.get('net_cost', 0):,.2f}")
        print(f"  Peak Building Load: {summary.get('peak_building_load_kw', 0):,.1f} kW")
        print(f"  Peak PV Power: {summary.get('peak_pv_power_kw', 0):,.1f} kW")
        
        # Save results
        if save_results:
            results_df = controller.get_results_dataframe()
            csv_path = os.path.join(output_dir, 'integrated_results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
        
        return controller.get_results_dataframe()
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Integrated Building Energy Simulation')
    parser.add_argument('--idf', type=str, default='models/RefBldgMediumOfficeNew2004_Chicago.idf',
                       help='Path to IDF model file')
    parser.add_argument('--epw', type=str, default='weather/chicago/TMY_lat41.88_lon-87.63.epw',
                       help='Path to EPW weather file')
    parser.add_argument('--output', type=str, default='outputs/integrated_sim',
                       help='Output directory')
    parser.add_argument('--pv-kw', type=float, default=100.0,
                       help='PV system capacity in kW')
    parser.add_argument('--battery-kwh', type=float, default=200.0,
                       help='Battery capacity in kWh')
    parser.add_argument('--battery-kw', type=float, default=50.0,
                       help='Battery max power in kW')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum simulation steps')
    parser.add_argument('--log-interval', type=int, default=12,
                       help='Log every N steps')
    
    args = parser.parse_args()
    
    run_integrated_simulation(
        idf_path=args.idf,
        epw_path=args.epw,
        output_dir=args.output,
        pv_capacity_kw=args.pv_kw,
        battery_capacity_kwh=args.battery_kwh,
        battery_power_kw=args.battery_kw,
        max_steps=args.max_steps,
        log_interval=args.log_interval
    )


if __name__ == "__main__":
    main()
