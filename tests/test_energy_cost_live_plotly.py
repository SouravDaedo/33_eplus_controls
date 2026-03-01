"""
Test script with LIVE PLOTTING using Plotly (faster than matplotlib)

Shows real-time updating charts during simulation:
- Electricity price
- PV generation vs Building load
- Battery SOC
- Cumulative cost

Run with: python test_energy_cost_live_plotly.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import webbrowser
import threading
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from energy_price_model import (
    create_tou_pricing,
    create_rtp_pricing,
)
from battery_model import create_battery, BatteryAction
from solar_pv_model import create_pv_system


def run_simulation_with_plotly(num_hours: int = 24):
    """Run the simulation and create an animated Plotly chart.
    
    Args:
        num_hours: Number of hours to simulate (default 24, can be multi-day)
    """
    print("=" * 60)
    print("ENERGY SIMULATION WITH PLOTLY VISUALIZATION")
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
    
    # Use TOU pricing for clearer peak/off-peak behavior
    # Note: July 15, 2006 was a Saturday, so we need weekday simulation
    # Let's use a Monday instead (July 17, 2006)
    from energy_price_model import PricingConfig, PricingType, TOUPeriod, EnergyPriceModel
    
    config = PricingConfig(
        pricing_type=PricingType.TOU,
        tou_periods=[
            TOUPeriod("off-peak", 0, 8, 0.06, [0,1,2,3,4,5,6]),   # Night
            TOUPeriod("mid-peak", 8, 14, 0.12, [0,1,2,3,4,5,6]),  # Morning
            TOUPeriod("peak", 14, 20, 0.28, [0,1,2,3,4,5,6]),     # Afternoon peak (all days)
            TOUPeriod("mid-peak", 20, 24, 0.12, [0,1,2,3,4,5,6]), # Evening
        ],
        feed_in_tariff=0.05
    )
    price_model = EnergyPriceModel(config)
    
    print(f"\nSystem Configuration:")
    print(f"  PV: 75 kW")
    print(f"  Battery: 200 kWh, 50 kW")
    print(f"  Pricing: TOU (peak: $0.28, off-peak: $0.08)")
    print(f"  Weather: {weather_file}")
    
    # Run simulation and collect all data
    start = datetime(weather_year, 7, 15, 0, 0)
    end = start + timedelta(hours=num_hours)
    
    results = []
    cumulative_cost = 0
    cumulative_credit = 0
    
    num_days = (num_hours + 23) // 24
    print(f"\nRunning simulation from {start} to {end} ({num_hours} hours, {num_days} day(s))...")
    
    for hour in range(num_hours):
        ts = start + timedelta(hours=hour)
        
        # Variable building load (office pattern based on hour of day)
        hour_of_day = ts.hour
        if 8 <= hour_of_day <= 18:
            building_load = 60 + 20 * (1 - abs(hour_of_day - 13) / 5)
        else:
            building_load = 30
        
        # Get PV production
        pv_state = pv.get_power_at_timestep(ts)
        pv_power = pv_state.ac_power_kw
        
        # Get current price
        price_state = price_model.get_price(ts)
        current_price = price_state.price_per_kwh
        is_peak = price_state.is_peak
        
        # Improved control strategy
        net_load = building_load - pv_power
        soc = battery.get_soc()
        
        # Strategy:
        # 1. During off-peak (cheap): charge battery from grid if SOC < 80%
        # 2. During peak (expensive): discharge battery if SOC > 20%
        # 3. Always use excess PV to charge battery
        
        if net_load < 0:
            # Excess PV - charge battery
            action = BatteryAction.CHARGE_FROM_PV
            power = min(abs(net_load), battery.get_available_charge_power())
        elif is_peak and soc > 0.2:
            # Peak hours - discharge to reduce grid import
            action = BatteryAction.DISCHARGE_TO_LOAD
            power = min(net_load, battery.get_available_discharge_power())
        elif not is_peak and soc < 0.8 and hour_of_day < 14:
            # Off-peak before peak - charge from grid
            action = BatteryAction.CHARGE_FROM_GRID
            power = min(30, battery.get_available_charge_power())  # Charge at 30 kW
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
        elif action == BatteryAction.CHARGE_FROM_GRID:
            grid_import = max(0, net_load) + batt_result.power_actual_kw
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
        
        results.append({
            'hour': hour,
            'timestamp': ts,
            'price': current_price,
            'is_peak': is_peak,
            'pv': pv_power,
            'load': building_load,
            'net_load': net_load,
            'soc': batt_result.soc_after * 100,
            'batt_power': batt_result.power_actual_kw,
            'action': action.name,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'hourly_cost': cost_result.net_cost,
            'cumulative_cost': cumulative_cost,
            'cumulative_credit': cumulative_credit,
            'net_cost': cumulative_cost - cumulative_credit
        })
        
        print(f"  Hour {hour:02d}: Price=${current_price:.2f}, PV={pv_power:.1f}kW, "
              f"SOC={batt_result.soc_after*100:.1f}%, Action={action.name}")
    
    df = pd.DataFrame(results)
    
    # Create animated Plotly figure
    print("\nCreating animated visualization...")
    create_animated_plot(df)
    
    # Print summary
    summary = price_model.get_cost_summary()
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total Import Cost: ${summary['total_import_cost']:.2f}")
    print(f"Total Export Credit: ${summary['total_export_credit']:.2f}")
    print(f"Net Energy Cost: ${summary['net_energy_cost']:.2f}")
    print(f"Battery Final SOC: {battery.get_soc():.1%}")


def create_animated_plot(df: pd.DataFrame):
    """Create an animated Plotly visualization."""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Electricity Price ($/kWh)', 
            'PV vs Load (kW)',
            'Battery SOC (%)', 
            'Grid Power (kW)',
            'Cumulative Cost ($)',
            'Hourly Summary'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Create frames for animation
    frames = []
    
    for i in range(1, len(df) + 1):
        frame_data = df.iloc[:i]
        
        frame = go.Frame(
            data=[
                # Price
                go.Scatter(x=frame_data['timestamp'], y=frame_data['price'],
                          mode='lines+markers', name='Price',
                          line=dict(color='blue', width=2)),
                
                # PV
                go.Scatter(x=frame_data['timestamp'], y=frame_data['pv'],
                          mode='lines+markers', name='PV',
                          line=dict(color='green', width=2),
                          fill='tozeroy', fillcolor='rgba(0,255,0,0.2)'),
                
                # Load
                go.Scatter(x=frame_data['timestamp'], y=frame_data['load'],
                          mode='lines+markers', name='Load',
                          line=dict(color='red', width=2)),
                
                # SOC
                go.Scatter(x=frame_data['timestamp'], y=frame_data['soc'],
                          mode='lines+markers', name='SOC',
                          line=dict(color='purple', width=3),
                          fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'),
                
                # Grid Import
                go.Scatter(x=frame_data['timestamp'], y=frame_data['grid_import'],
                          mode='lines+markers', name='Import',
                          line=dict(color='red', width=2)),
                
                # Grid Export (negative)
                go.Scatter(x=frame_data['timestamp'], y=-frame_data['grid_export'],
                          mode='lines+markers', name='Export',
                          line=dict(color='green', width=2)),
                
                # Cumulative Cost
                go.Scatter(x=frame_data['timestamp'], y=frame_data['cumulative_cost'],
                          mode='lines', name='Import Cost',
                          line=dict(color='red', width=2)),
                
                # Cumulative Credit
                go.Scatter(x=frame_data['timestamp'], y=frame_data['cumulative_credit'],
                          mode='lines', name='Export Credit',
                          line=dict(color='green', width=2)),
                
                # Net Cost
                go.Scatter(x=frame_data['timestamp'], y=frame_data['net_cost'],
                          mode='lines', name='Net Cost',
                          line=dict(color='black', width=3)),
                
                # Summary table
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                               fill_color='lightblue',
                               align='left'),
                    cells=dict(
                        values=[
                            ['Time', 'Price', 'PV', 'Load', 'SOC', 'Action', 'Net Cost'],
                            [
                                frame_data.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M'),
                                f"${frame_data.iloc[-1]['price']:.3f}/kWh",
                                f"{frame_data.iloc[-1]['pv']:.1f} kW",
                                f"{frame_data.iloc[-1]['load']:.1f} kW",
                                f"{frame_data.iloc[-1]['soc']:.1f}%",
                                frame_data.iloc[-1]['action'],
                                f"${frame_data.iloc[-1]['net_cost']:.2f}"
                            ]
                        ],
                        fill_color='white',
                        align='left'
                    )
                )
            ],
            name=str(i)
        )
        frames.append(frame)
    
    # Initial data (first point)
    init_data = df.iloc[:1]
    
    # Add initial traces
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['price'],
                            mode='lines+markers', name='Price',
                            line=dict(color='blue', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['pv'],
                            mode='lines+markers', name='PV',
                            line=dict(color='green', width=2),
                            fill='tozeroy', fillcolor='rgba(0,255,0,0.2)'), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['load'],
                            mode='lines+markers', name='Load',
                            line=dict(color='red', width=2)), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['soc'],
                            mode='lines+markers', name='SOC',
                            line=dict(color='purple', width=3),
                            fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['grid_import'],
                            mode='lines+markers', name='Import',
                            line=dict(color='red', width=2)), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=-init_data['grid_export'],
                            mode='lines+markers', name='Export',
                            line=dict(color='green', width=2)), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['cumulative_cost'],
                            mode='lines', name='Import Cost',
                            line=dict(color='red', width=2)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['cumulative_credit'],
                            mode='lines', name='Export Credit',
                            line=dict(color='green', width=2)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=init_data['timestamp'], y=init_data['net_cost'],
                            mode='lines', name='Net Cost',
                            line=dict(color='black', width=3)), row=3, col=1)
    
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'], fill_color='lightblue', align='left'),
        cells=dict(values=[['Time', 'Price', 'PV', 'Load', 'SOC', 'Action', 'Net Cost'],
                          [init_data.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M'), '$0.08', '0 kW', '30 kW', '50%', 'IDLE', '$0']],
                  fill_color='white', align='left')
    ), row=3, col=2)
    
    # Add frames
    fig.frames = frames
    
    # Animation settings
    fig.update_layout(
        title=dict(
            text='<b>Building Energy Simulation - Live View</b>',
            x=0.5,
            font=dict(size=20)
        ),
        height=900,
        showlegend=True,
        legend=dict(
            orientation='v',  # Vertical legend
            yanchor='top',
            y=0.95,
            xanchor='left',
            x=1.02,  # Position to the right of plots
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 300, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Hour: ',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 100, 'redraw': True},
                                     'mode': 'immediate',
                                     'transition': {'duration': 100}}],
                 'label': df.iloc[i-1]['timestamp'].strftime('%m-%d %H:%M'),
                 'method': 'animate'}
                for i in range(1, len(df) + 1)
            ]
        }]
    )
    
    # Calculate dynamic axis ranges from data
    time_start = df['timestamp'].min()
    time_end = df['timestamp'].max()
    time_padding = (time_end - time_start) * 0.02  # 2% padding
    time_range = [time_start - time_padding, time_end + time_padding]
    
    max_price = df['price'].max() * 1.1
    max_power = max(df['pv'].max(), df['load'].max()) * 1.1
    max_grid = max(df['grid_import'].max(), df['grid_export'].max()) * 1.1
    
    # Update axes with dynamic datetime ranges
    fig.update_xaxes(title_text='Time', range=time_range, row=1, col=1)
    fig.update_xaxes(title_text='Time', range=time_range, row=1, col=2)
    fig.update_xaxes(title_text='Time', range=time_range, row=2, col=1)
    fig.update_xaxes(title_text='Time', range=time_range, row=2, col=2)
    fig.update_xaxes(title_text='Time', range=time_range, row=3, col=1)
    
    fig.update_yaxes(title_text='$/kWh', range=[0, max_price], row=1, col=1)
    fig.update_yaxes(title_text='kW', range=[0, max_power], row=1, col=2)
    fig.update_yaxes(title_text='%', range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text='kW', range=[-max_grid, max_grid], row=2, col=2)
    fig.update_yaxes(title_text='$', range=[0, max(df['cumulative_cost'].max() * 1.1, 50)], row=3, col=1)
    
    # Add horizontal lines for SOC limits (using shapes with datetime x-coordinates)
    fig.add_shape(type="line", x0=time_start, x1=time_end, y0=20, y1=20,
                  line=dict(color="red", width=1, dash="dash"),
                  xref="x3", yref="y3", opacity=0.5)
    fig.add_shape(type="line", x0=time_start, x1=time_end, y0=80, y1=80,
                  line=dict(color="green", width=1, dash="dash"),
                  xref="x3", yref="y3", opacity=0.5)
    
    # Add peak period shading on price chart for each day in simulation
    from datetime import timedelta
    current_day = time_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current_day <= time_end:
        peak_start = current_day.replace(hour=14)
        peak_end = current_day.replace(hour=20)
        if peak_start <= time_end and peak_end >= time_start:
            fig.add_shape(type="rect", x0=max(peak_start, time_start), x1=min(peak_end, time_end), 
                          y0=0, y1=max_price,
                          fillcolor="red", opacity=0.1, layer="below", line_width=0,
                          xref="x1", yref="y1")
        current_day += timedelta(days=1)
    
    # Save and open
    output_file = 'simulation_live.html'
    fig.write_html(output_file, auto_open=True)
    print(f"\nVisualization saved to: {output_file}")
    print("Opening in browser... Click 'Play' to start animation!")


if __name__ == "__main__":
    run_simulation_with_plotly()
