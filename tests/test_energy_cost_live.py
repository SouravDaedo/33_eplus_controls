"""
Test script with LIVE PLOTTING for Energy Price Model

Shows real-time updating charts during simulation:
- Electricity price
- PV generation vs Building load
- Battery SOC
- Cumulative cost

Run with: python test_energy_cost_live.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

from energy_price_model import (
    EnergyPriceModel,
    PricingConfig,
    PricingType,
    create_tou_pricing,
    create_rtp_pricing,
)
from battery_model import create_battery, BatteryAction
from solar_pv_model import create_pv_system


class LiveSimulationPlotter:
    """
    Live plotting for building energy simulation.
    
    Updates charts in real-time as simulation progresses.
    """
    
    def __init__(self, num_hours: int = 24, update_delay: float = 0.3):
        """
        Initialize the live plotter.
        
        Args:
            num_hours: Total simulation hours
            update_delay: Delay between updates (seconds) for visualization
        """
        self.num_hours = num_hours
        self.update_delay = update_delay
        
        # Data storage
        self.hours = []
        self.prices = []
        self.pv_power = []
        self.load = []
        self.battery_soc = []
        self.grid_import = []
        self.grid_export = []
        self.cumulative_cost = []
        self.cumulative_credit = []
        
        # Setup figure
        self._setup_figure()
    
    def _setup_figure(self):
        """Create the figure and subplots."""
        plt.ion()  # Interactive mode
        
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Live Building Energy Simulation', fontsize=14, fontweight='bold')
        
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.35, wspace=0.25)
        
        # Subplot 1: Electricity Price
        self.ax_price = self.fig.add_subplot(gs[0, 0])
        self.ax_price.set_title('Electricity Price')
        self.ax_price.set_xlabel('Hour')
        self.ax_price.set_ylabel('Price ($/kWh)')
        self.ax_price.set_xlim(0, self.num_hours)
        self.ax_price.set_ylim(0, 0.35)
        self.ax_price.grid(True, alpha=0.3)
        self.line_price, = self.ax_price.plot([], [], 'b-', linewidth=2, label='Price')
        self.ax_price.axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='High price threshold')
        self.ax_price.legend(loc='upper right', fontsize=8)
        
        # Subplot 2: PV vs Load
        self.ax_power = self.fig.add_subplot(gs[0, 1])
        self.ax_power.set_title('PV Generation vs Building Load')
        self.ax_power.set_xlabel('Hour')
        self.ax_power.set_ylabel('Power (kW)')
        self.ax_power.set_xlim(0, self.num_hours)
        self.ax_power.set_ylim(0, 100)
        self.ax_power.grid(True, alpha=0.3)
        self.line_pv, = self.ax_power.plot([], [], 'g-', linewidth=2, label='PV')
        self.line_load, = self.ax_power.plot([], [], 'r-', linewidth=2, label='Load')
        self.ax_power.legend(loc='upper right', fontsize=8)
        self.ax_power.fill_between([], [], alpha=0.3)
        
        # Subplot 3: Battery SOC
        self.ax_soc = self.fig.add_subplot(gs[1, 0])
        self.ax_soc.set_title('Battery State of Charge')
        self.ax_soc.set_xlabel('Hour')
        self.ax_soc.set_ylabel('SOC (%)')
        self.ax_soc.set_xlim(0, self.num_hours)
        self.ax_soc.set_ylim(0, 100)
        self.ax_soc.grid(True, alpha=0.3)
        self.line_soc, = self.ax_soc.plot([], [], 'purple', linewidth=2)
        self.ax_soc.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Min SOC')
        self.ax_soc.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Max SOC')
        self.ax_soc.legend(loc='upper right', fontsize=8)
        
        # Subplot 4: Grid Power
        self.ax_grid = self.fig.add_subplot(gs[1, 1])
        self.ax_grid.set_title('Grid Power Flow')
        self.ax_grid.set_xlabel('Hour')
        self.ax_grid.set_ylabel('Power (kW)')
        self.ax_grid.set_xlim(0, self.num_hours)
        self.ax_grid.set_ylim(-50, 80)
        self.ax_grid.grid(True, alpha=0.3)
        self.ax_grid.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.line_grid_import, = self.ax_grid.plot([], [], 'r-', linewidth=2, label='Import')
        self.line_grid_export, = self.ax_grid.plot([], [], 'g-', linewidth=2, label='Export')
        self.ax_grid.legend(loc='upper right', fontsize=8)
        
        # Subplot 5: Cumulative Cost (spans bottom row)
        self.ax_cost = self.fig.add_subplot(gs[2, :])
        self.ax_cost.set_title('Cumulative Energy Cost')
        self.ax_cost.set_xlabel('Hour')
        self.ax_cost.set_ylabel('Cost ($)')
        self.ax_cost.set_xlim(0, self.num_hours)
        self.ax_cost.set_ylim(0, 200)
        self.ax_cost.grid(True, alpha=0.3)
        self.line_cost, = self.ax_cost.plot([], [], 'b-', linewidth=2, label='Import Cost')
        self.line_credit, = self.ax_cost.plot([], [], 'g-', linewidth=2, label='Export Credit')
        self.line_net, = self.ax_cost.plot([], [], 'k-', linewidth=3, label='Net Cost')
        self.ax_cost.legend(loc='upper left', fontsize=8)
        
        # Text annotation for current values
        self.text_current = self.fig.text(
            0.02, 0.02, '', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.fig.canvas.draw()
        plt.pause(0.1)
    
    def update(self, hour: int, price: float, pv: float, load: float, 
               soc: float, grid_imp: float, grid_exp: float, 
               cost: float, credit: float, action: str):
        """
        Update the plots with new data.
        
        Args:
            hour: Current simulation hour
            price: Current electricity price ($/kWh)
            pv: PV power output (kW)
            load: Building load (kW)
            soc: Battery SOC (0-1)
            grid_imp: Grid import power (kW)
            grid_exp: Grid export power (kW)
            cost: Cumulative import cost ($)
            credit: Cumulative export credit ($)
            action: Current battery action
        """
        # Append data
        self.hours.append(hour)
        self.prices.append(price)
        self.pv_power.append(pv)
        self.load.append(load)
        self.battery_soc.append(soc * 100)
        self.grid_import.append(grid_imp)
        self.grid_export.append(-grid_exp)  # Negative for export
        self.cumulative_cost.append(cost)
        self.cumulative_credit.append(credit)
        
        # Update lines
        self.line_price.set_data(self.hours, self.prices)
        self.line_pv.set_data(self.hours, self.pv_power)
        self.line_load.set_data(self.hours, self.load)
        self.line_soc.set_data(self.hours, self.battery_soc)
        self.line_grid_import.set_data(self.hours, self.grid_import)
        self.line_grid_export.set_data(self.hours, self.grid_export)
        self.line_cost.set_data(self.hours, self.cumulative_cost)
        self.line_credit.set_data(self.hours, self.cumulative_credit)
        
        # Net cost line
        net_costs = [c - cr for c, cr in zip(self.cumulative_cost, self.cumulative_credit)]
        self.line_net.set_data(self.hours, net_costs)
        
        # Auto-scale y-axes
        if self.prices:
            self.ax_price.set_ylim(0, max(0.35, max(self.prices) * 1.1))
        if self.pv_power or self.load:
            max_power = max(max(self.pv_power, default=0), max(self.load, default=0))
            self.ax_power.set_ylim(0, max(100, max_power * 1.1))
        if self.cumulative_cost:
            max_cost = max(max(self.cumulative_cost), max(net_costs, default=0))
            self.ax_cost.set_ylim(0, max(50, max_cost * 1.1))
        
        # Update current values text
        net_cost = cost - credit
        self.text_current.set_text(
            f'Hour: {hour:02d}:00 | Price: ${price:.3f}/kWh | '
            f'PV: {pv:.1f}kW | Load: {load:.1f}kW | '
            f'SOC: {soc*100:.1f}% | Action: {action}\n'
            f'Grid: {grid_imp-grid_exp:+.1f}kW | '
            f'Cost: ${cost:.2f} | Credit: ${credit:.2f} | Net: ${net_cost:.2f}'
        )
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Delay for visualization
        time.sleep(self.update_delay)
    
    def finish(self):
        """Finalize the plot."""
        plt.ioff()
        self.fig.suptitle('Building Energy Simulation - Complete', fontsize=14, fontweight='bold')
        self.fig.canvas.draw()
    
    def show(self):
        """Keep the plot window open."""
        plt.show()


def run_live_simulation():
    """Run the simulation with live plotting."""
    print("=" * 60)
    print("LIVE ENERGY SIMULATION")
    print("=" * 60)
    
    # Find weather file
    weather_file = None
    for wf in ["weather/chicago/TMY_lat41.88_lon-87.63.epw",
               "weather/atlanta_2023/hourly_lat33.75_lon-84.39_2023-2023.epw"]:
        if Path(wf).exists():
            weather_file = wf
            break
    
    if not weather_file:
        print("No weather file found!")
        return
    
    # Create models
    pv = create_pv_system(capacity_kw=75)
    pv.load_weather(weather_file)
    weather_year = pv.weather_data.index[0].year
    
    battery = create_battery(
        capacity_kwh=200,
        max_power_kw=50,
        efficiency=0.90,
        timestep_minutes=60
    )
    
    price_model = create_rtp_pricing(
        base_price=0.12,
        volatility=0.4,
        feed_in_tariff=0.05
    )
    
    print(f"\nSystem Configuration:")
    print(f"  PV: 75 kW")
    print(f"  Battery: 200 kWh, 50 kW")
    print(f"  Pricing: Real-Time Pricing")
    print(f"  Weather: {weather_file}")
    
    # Create live plotter
    plotter = LiveSimulationPlotter(num_hours=24, update_delay=0.3)
    
    # Simulation
    start = datetime(weather_year, 7, 15, 0, 0)
    cumulative_cost = 0
    cumulative_credit = 0
    
    print(f"\nStarting simulation for {start.date()}...")
    print("Watch the live plot update!")
    
    for hour in range(24):
        ts = start + timedelta(hours=hour)
        
        # Variable building load
        if 8 <= hour <= 18:
            building_load = 60 + 20 * (1 - abs(hour - 13) / 5)
        else:
            building_load = 30
        
        # Get PV production
        pv_state = pv.get_power_at_timestep(ts)
        pv_power = pv_state.ac_power_kw
        
        # Get current price
        price_state = price_model.get_price(ts)
        current_price = price_state.price_per_kwh
        
        # Control strategy
        net_load = building_load - pv_power
        
        if net_load > 0:
            if current_price > 0.15 and battery.get_soc() > 0.2:
                action = BatteryAction.DISCHARGE_TO_LOAD
                power = min(net_load, battery.get_available_discharge_power())
            else:
                action = BatteryAction.IDLE
                power = 0
        else:
            if battery.get_soc() < 0.9:
                action = BatteryAction.CHARGE_FROM_PV
                power = min(abs(net_load), battery.get_available_charge_power())
            else:
                action = BatteryAction.IDLE
                power = 0
        
        # Execute battery action
        batt_result = battery.step(
            action=action,
            power_kw=power,
            pv_available_kw=max(0, -net_load),
            load_demand_kw=max(0, net_load)
        )
        
        # Calculate grid interaction
        if action == BatteryAction.DISCHARGE_TO_LOAD:
            grid_import = max(0, net_load - batt_result.power_actual_kw)
            grid_export = 0
        elif action == BatteryAction.CHARGE_FROM_PV:
            grid_import = max(0, net_load)
            grid_export = max(0, -net_load - batt_result.power_actual_kw)
        else:
            grid_import = max(0, net_load)
            grid_export = max(0, -net_load)
        
        # Calculate cost
        cost_result = price_model.calculate_cost(
            timestamp=ts,
            energy_consumed_kwh=grid_import,
            energy_exported_kwh=grid_export
        )
        
        cumulative_cost += cost_result.import_cost
        cumulative_credit += cost_result.export_credit
        
        # Update live plot
        plotter.update(
            hour=hour,
            price=current_price,
            pv=pv_power,
            load=building_load,
            soc=batt_result.soc_after,
            grid_imp=grid_import,
            grid_exp=grid_export,
            cost=cumulative_cost,
            credit=cumulative_credit,
            action=action.name
        )
    
    # Finalize
    plotter.finish()
    
    # Print summary
    summary = price_model.get_cost_summary()
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total Import Cost: ${summary['total_import_cost']:.2f}")
    print(f"Total Export Credit: ${summary['total_export_credit']:.2f}")
    print(f"Net Energy Cost: ${summary['net_energy_cost']:.2f}")
    print(f"Battery Final SOC: {battery.get_soc():.1%}")
    
    print("\nClose the plot window to exit.")
    plotter.show()


if __name__ == "__main__":
    run_live_simulation()
