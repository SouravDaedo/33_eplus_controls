"""
Integrated Building Energy Simulation with LIVE Plotly Visualization

Runs EnergyPlus step-by-step with real building load, combined with:
- Solar PV model
- Battery storage model  
- Real-time electricity pricing
- Live updating Plotly dashboard

Usage:
    python integrated_simulation_live.py
    python integrated_simulation_live.py --idf models/your_model.idf --max-steps 288
"""

import os
import sys
import argparse
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eplus_env import EnergyPlusEnv
from battery_model import create_battery, BatteryAction, BatteryModel
from solar_pv_model import create_pv_system, SolarPVModel
from energy_price_model import (
    EnergyPriceModel, PricingConfig, PricingType, TOUPeriod
)


class LiveDashboard:
    """
    Live updating Plotly dashboard for simulation visualization.
    
    Creates an HTML file that auto-refreshes with simulation data.
    """
    
    def __init__(self, output_dir: str = "outputs", refresh_interval: int = 2):
        self.output_dir = output_dir
        self.refresh_interval = refresh_interval
        self.data_file = os.path.join(output_dir, 'live_data.json')
        self.html_file = os.path.join(output_dir, 'live_dashboard.html')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize empty data and write to JSON file immediately
        self.data = {
            'timestamps': [],
            'hours': [],
            'building_load': [],
            'pv_power': [],
            'battery_soc': [],
            'battery_action': [],
            'grid_import': [],
            'grid_export': [],
            'price': [],
            'cumulative_cost': [],
            'outdoor_temp': [],
            'zone_temps': [],
            'cooling_setpoint': [],
            'heating_setpoint': []
        }
        
        # Write initial empty JSON file
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f)
        
        self._create_dashboard_html()
    
    def _create_dashboard_html(self):
        """Create the auto-refreshing HTML dashboard."""
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Live Building Energy Simulation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        h1 {{ 
            text-align: center; 
            color: #333;
        }}
        .status {{
            text-align: center;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .status.running {{ background-color: #2196F3; }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }}
        .metric {{
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        #charts {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>Live Building Energy Simulation</h1>
    <div id="status" class="status running">Waiting for simulation data...</div>
    
    <div class="summary">
        <div class="metric">
            <div class="metric-value" id="building-load">--</div>
            <div class="metric-label">Building Load (kW)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="pv-power">--</div>
            <div class="metric-label">PV Power (kW)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="battery-soc">--</div>
            <div class="metric-label">Battery SOC</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="grid-power">--</div>
            <div class="metric-label">Grid Power (kW)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="price">--</div>
            <div class="metric-label">Price ($/kWh)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="cost">--</div>
            <div class="metric-label">Net Cost ($)</div>
        </div>
    </div>
    
    <div id="charts" style="width:100%; height:1000px;"></div>
    
    <script>
        let lastUpdate = 0;
        let chartInitialized = false;
        
        async function fetchData() {{
            try {{
                const response = await fetch('live_data.json?t=' + Date.now());
                const data = await response.json();
                
                console.log('Fetched data, length:', data.timestamps.length);
                
                if (data.timestamps.length > 0) {{
                    updateDashboard(data);
                    lastUpdate = data.timestamps.length;
                }}
            }} catch (e) {{
                console.log('Waiting for data...', e);
            }}
        }}
        
        function updateDashboard(data) {{
            const n = data.timestamps.length;
            if (n === 0) return;
            
            // Convert steps to hours (12 timesteps per hour = 5-min intervals)
            const timestepsPerHour = 12;
            const xHours = data.hours.map(i => i / timestepsPerHour);
            
            // Update status
            document.getElementById('status').textContent = 
                'Step ' + n + ' | ' + data.timestamps[n-1];
            
            // Update metrics
            document.getElementById('building-load').textContent = 
                data.building_load[n-1].toFixed(1);
            document.getElementById('pv-power').textContent = 
                data.pv_power[n-1].toFixed(1);
            document.getElementById('battery-soc').textContent = 
                (data.battery_soc[n-1] * 100).toFixed(0) + '%';
            document.getElementById('grid-power').textContent = 
                (data.grid_import[n-1] - data.grid_export[n-1]).toFixed(1);
            document.getElementById('price').textContent = 
                '$' + data.price[n-1].toFixed(3);
            document.getElementById('cost').textContent = 
                '$' + data.cumulative_cost[n-1].toFixed(2);
            
            // Update charts
            const traces = [
                // Price
                {{
                    x: xHours,
                    y: data.price,
                    name: 'Price',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'blue', width: 2}},
                    xaxis: 'x1',
                    yaxis: 'y1'
                }},
                // Building Load
                {{
                    x: xHours,
                    y: data.building_load,
                    name: 'Building',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(255,0,0,0.2)',
                    line: {{color: 'red', width: 2}},
                    xaxis: 'x2',
                    yaxis: 'y2'
                }},
                // PV Power
                {{
                    x: xHours,
                    y: data.pv_power,
                    name: 'PV',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0,255,0,0.2)',
                    line: {{color: 'green', width: 2}},
                    xaxis: 'x2',
                    yaxis: 'y2'
                }},
                // Battery SOC
                {{
                    x: xHours,
                    y: data.battery_soc.map(x => x * 100),
                    name: 'SOC',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(128,0,128,0.2)',
                    line: {{color: 'purple', width: 3}},
                    xaxis: 'x3',
                    yaxis: 'y3'
                }},
                // Grid Import
                {{
                    x: xHours,
                    y: data.grid_import,
                    name: 'Import',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'red', width: 2}},
                    xaxis: 'x4',
                    yaxis: 'y4'
                }},
                // Grid Export (negative)
                {{
                    x: xHours,
                    y: data.grid_export.map(x => -x),
                    name: 'Export',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'green', width: 2}},
                    xaxis: 'x4',
                    yaxis: 'y4'
                }},
                // Cumulative Cost
                {{
                    x: xHours,
                    y: data.cumulative_cost,
                    name: 'Net Cost',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'black', width: 3}},
                    xaxis: 'x5',
                    yaxis: 'y5'
                }},
                // Outdoor Temperature
                {{
                    x: xHours,
                    y: data.outdoor_temp,
                    name: 'Outdoor',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'orange', width: 2}},
                    xaxis: 'x6',
                    yaxis: 'y6'
                }},
                // Zone Temperature
                {{
                    x: xHours,
                    y: data.zone_temps || [],
                    name: 'Zone Temp',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'purple', width: 2}},
                    xaxis: 'x7',
                    yaxis: 'y7'
                }},
                // Cooling Setpoint
                {{
                    x: xHours,
                    y: data.cooling_setpoint || [],
                    name: 'Cooling SP',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'blue', width: 2, dash: 'dash'}},
                    xaxis: 'x7',
                    yaxis: 'y7'
                }},
                // Heating Setpoint
                {{
                    x: xHours,
                    y: data.heating_setpoint || [],
                    name: 'Heating SP',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: 'red', width: 2, dash: 'dash'}},
                    xaxis: 'x7',
                    yaxis: 'y7'
                }}
            ];
            
            const layout = {{
                height: 1000,
                showlegend: true,
                legend: {{orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center'}},
                grid: {{rows: 4, columns: 2, pattern: 'independent'}},
                
                xaxis1: {{title: 'Hour', domain: [0, 0.45], dtick: 1}},
                yaxis1: {{title: '$/kWh', domain: [0.78, 1], range: [0, 0.35]}},
                
                xaxis2: {{title: 'Hour', domain: [0.55, 1], dtick: 1}},
                yaxis2: {{title: 'kW', domain: [0.78, 1]}},
                
                xaxis3: {{title: 'Hour', domain: [0, 0.45], dtick: 1}},
                yaxis3: {{title: '%', domain: [0.53, 0.73], range: [0, 100]}},
                
                xaxis4: {{title: 'Hour', domain: [0.55, 1], dtick: 1}},
                yaxis4: {{title: 'kW', domain: [0.53, 0.73]}},
                
                xaxis5: {{title: 'Hour', domain: [0, 0.45], dtick: 1}},
                yaxis5: {{title: '$', domain: [0.27, 0.48]}},
                
                xaxis6: {{title: 'Hour', domain: [0.55, 1], dtick: 1}},
                yaxis6: {{title: 'C', domain: [0.27, 0.48]}},
                
                xaxis7: {{title: 'Hour', domain: [0, 0.45], dtick: 1}},
                yaxis7: {{title: 'C', domain: [0, 0.22]}},
                
                annotations: [
                    {{text: 'Electricity Price', x: 0.225, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Building Load vs PV', x: 0.775, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Battery SOC', x: 0.225, y: 0.75, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Grid Power', x: 0.775, y: 0.75, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Cumulative Cost', x: 0.225, y: 0.50, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Outdoor Temp', x: 0.775, y: 0.50, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}},
                    {{text: 'Zone Temp & Setpoints', x: 0.225, y: 0.24, xref: 'paper', yref: 'paper', showarrow: false, font: {{size: 14}}}}
                ]
            }};
            
            Plotly.newPlot('charts', traces, layout);
            console.log('Chart updated with', n, 'data points');
        }}
        
        // Fetch data every {self.refresh_interval} seconds
        setInterval(fetchData, {self.refresh_interval * 1000});
        // Initial fetch
        setTimeout(fetchData, 500);
    </script>
</body>
</html>'''
        
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard created: {self.html_file}")
    
    def update(self, result: Dict):
        """Update dashboard with new simulation result."""
        self.data['timestamps'].append(result['timestamp'].strftime('%Y-%m-%d %H:%M'))
        self.data['hours'].append(len(self.data['timestamps']) - 1)
        self.data['building_load'].append(result['building_load_kw'])
        self.data['pv_power'].append(result['pv_power_kw'])
        self.data['battery_soc'].append(result['battery_soc'])
        self.data['battery_action'].append(result['battery_action'])
        self.data['grid_import'].append(result['grid_import_kw'])
        self.data['grid_export'].append(result['grid_export_kw'])
        self.data['price'].append(result['price_per_kwh'])
        self.data['cumulative_cost'].append(result['cumulative_cost'])
        self.data['outdoor_temp'].append(result['outdoor_temp'])
        self.data['zone_temps'].append(result.get('zone_temp_avg', 21.0))
        self.data['cooling_setpoint'].append(result.get('cooling_setpoint', 24.0))
        self.data['heating_setpoint'].append(result.get('heating_setpoint', 21.0))
        
        # Write data to JSON file with flush
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
    
    def finalize(self):
        """Create final static Plotly HTML with all data embedded."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if not self.data['timestamps']:
            return
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Electricity Price', 'Building Load vs PV', 
                          'Battery SOC', 'Grid Power', 'Cumulative Cost', 'Outdoor Temperature',
                          'Zone Temperature & Setpoints', ''),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Convert steps to hours (IDF has 12 timesteps per hour = 5 min each)
        timesteps_per_hour = 12
        x_hours = [i / timesteps_per_hour for i in range(len(self.data['timestamps']))]
        
        # Price
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['price'], name='Price',
                                line=dict(color='blue', width=2)), row=1, col=1)
        
        # Building vs PV
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['building_load'], name='Building',
                                line=dict(color='red', width=2), fill='tozeroy'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['pv_power'], name='PV',
                                line=dict(color='green', width=2), fill='tozeroy'), row=1, col=2)
        
        # SOC
        soc_pct = [s * 100 for s in self.data['battery_soc']]
        fig.add_trace(go.Scatter(x=x_hours, y=soc_pct, name='SOC',
                                line=dict(color='purple', width=3), fill='tozeroy'), row=2, col=1)
        
        # Grid
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['grid_import'], name='Import',
                                line=dict(color='red', width=2)), row=2, col=2)
        export_neg = [-e for e in self.data['grid_export']]
        fig.add_trace(go.Scatter(x=x_hours, y=export_neg, name='Export',
                                line=dict(color='green', width=2)), row=2, col=2)
        
        # Cost
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['cumulative_cost'], name='Net Cost',
                                line=dict(color='black', width=3)), row=3, col=1)
        
        # Outdoor Temperature
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['outdoor_temp'], name='Outdoor Temp',
                                line=dict(color='orange', width=2)), row=3, col=2)
        
        # Zone Temperature & Setpoints
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['zone_temps'], name='Zone Temp',
                                line=dict(color='purple', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['cooling_setpoint'], name='Cooling SP',
                                line=dict(color='blue', width=2, dash='dash')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_hours, y=self.data['heating_setpoint'], name='Heating SP',
                                line=dict(color='red', width=2, dash='dash')), row=4, col=1)
        
        fig.update_layout(
            title='Building Energy Simulation Results',
            height=1000,
            showlegend=True,
            legend=dict(orientation='v', x=1.02, y=0.95)
        )
        
        fig.update_yaxes(title_text='$/kWh', row=1, col=1)
        fig.update_yaxes(title_text='kW', row=1, col=2)
        fig.update_yaxes(title_text='%', range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text='kW', row=2, col=2)
        fig.update_yaxes(title_text='$', row=3, col=1)
        fig.update_yaxes(title_text='°C', row=3, col=2)
        fig.update_yaxes(title_text='°C', row=4, col=1)
        
        # Add x-axis labels (hours)
        fig.update_xaxes(title_text='Hours', row=4, col=1)
        fig.update_xaxes(title_text='Hours', row=4, col=2)
        
        final_html = os.path.join(self.output_dir, 'simulation_results.html')
        fig.write_html(final_html)
        print(f"Final results saved to: {final_html}")
    
    def start_server(self, port: int = 8050):
        """Start a simple HTTP server for live updates."""
        import http.server
        import socketserver
        import functools
        
        # Save original directory and use absolute path for server
        self.original_cwd = os.getcwd()
        server_dir = os.path.abspath(self.output_dir)
        
        # Create handler that serves from output_dir without changing cwd
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=server_dir)
        
        self.httpd = socketserver.TCPServer(("", port), handler)
        
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        
        print(f"HTTP server started at http://localhost:{port}")
        return port
    
    def open_browser(self, port: int = 8050):
        """Open dashboard in browser via HTTP server."""
        import webbrowser
        webbrowser.open(f'http://localhost:{port}/live_dashboard.html')


def run_live_simulation(
    idf_path: str,
    epw_path: str,
    output_dir: str = "outputs/live_sim",
    pv_capacity_kw: float = 100.0,
    battery_capacity_kwh: float = 200.0,
    battery_power_kw: float = 50.0,
    max_steps: Optional[int] = None,
    open_browser: bool = True
):
    """
    Run integrated simulation with live Plotly dashboard.
    """
    print("=" * 70)
    print("LIVE INTEGRATED BUILDING ENERGY SIMULATION")
    print("=" * 70)
    
    # Validate paths
    if not os.path.exists(idf_path):
        print(f"ERROR: IDF file not found: {idf_path}")
        return
    if not os.path.exists(epw_path):
        print(f"ERROR: EPW file not found: {epw_path}")
        return
    
    print(f"\nConfiguration:")
    print(f"  IDF: {idf_path}")
    print(f"  Weather: {epw_path}")
    print(f"  PV: {pv_capacity_kw} kW")
    print(f"  Battery: {battery_capacity_kwh} kWh / {battery_power_kw} kW")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print("\nInitializing models...")
    
    # PV System
    pv = create_pv_system(capacity_kw=pv_capacity_kw)
    try:
        pv.load_weather(epw_path)
        print(f"  PV: Loaded")
    except Exception as e:
        print(f"  PV: Failed to load weather - {e}")
        pv = None
    
    # Battery
    battery = create_battery(
        capacity_kwh=battery_capacity_kwh,
        max_power_kw=battery_power_kw,
        efficiency=0.90,
        timestep_minutes=15  # 15-minute timesteps
    )
    print(f"  Battery: Loaded")
    
    # Pricing
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
    print(f"  Pricing: TOU loaded")
    
    # Create live dashboard
    dashboard = LiveDashboard(output_dir)
    
    # Start HTTP server for live updates (avoids CORS issues)
    port = dashboard.start_server(port=8050)
    
    if open_browser:
        print("\nOpening live dashboard in browser...")
        dashboard.open_browser(port)
    
    # Create EnergyPlus environment
    print("\nStarting EnergyPlus simulation...")
    env = EnergyPlusEnv(idf_path, epw_path, output_dir)
    
    # Tracking
    cumulative_import_cost = 0.0
    cumulative_export_credit = 0.0
    results = []
    
    try:
        obs = env.reset()
        step = 0
        
        print("\nSimulation running - watch the dashboard!")
        print("-" * 50)
        
        while not env.done:
            if max_steps and step >= max_steps:
                print(f"\nReached max steps ({max_steps})")
                break
            
            # Build datetime
            year = obs.get('year', 2007)
            month = obs.get('month', 1)
            day = obs.get('day', 1)
            hour = obs.get('hour', 0)
            minute = obs.get('minute', 0)
            if minute >= 60:
                minute = 0
                hour += 1
            if hour >= 24:
                hour = 0
            
            try:
                sim_datetime = datetime(year, month, day, hour, minute)
            except ValueError:
                sim_datetime = datetime(year, month, 1, hour, minute)
            
            # Get building load
            # EnergyPlus meter returns energy in Joules per timestep
            # IDF has Timestep=12 (12 per hour = 5-minute intervals = 300 seconds)
            timestep_seconds = 5 * 60  # 5 minutes = 300 seconds
            energy_joules = obs.get('total_power', 0)
            building_load_kw = energy_joules / timestep_seconds / 1000.0  # J -> kW
            outdoor_temp = obs.get('outdoor_temp', 20.0)
            
            # Get zone temperatures (average of all zones)
            zone_temps = obs.get('zone_temps', {})
            if zone_temps:
                zone_temp_avg = sum(zone_temps.values()) / len(zone_temps)
            else:
                zone_temp_avg = 21.0
            
            # Setpoints used for control
            cooling_setpoint = 24.0
            heating_setpoint = 21.0
            
            # Get PV
            pv_power_kw = 0.0
            if pv:
                try:
                    pv_state = pv.get_power_at_timestep(sim_datetime)
                    pv_power_kw = pv_state.ac_power_kw
                except:
                    pass
            
            # Get price
            price_state = price_model.get_price(sim_datetime)
            current_price = price_state.price_per_kwh
            is_peak = price_state.is_peak
            
            # Net load
            net_load_kw = building_load_kw - pv_power_kw
            
            # Battery control
            battery_soc = battery.get_soc()
            
            if net_load_kw < 0:
                action = BatteryAction.CHARGE_FROM_PV
                power = min(abs(net_load_kw), battery.get_available_charge_power())
            elif is_peak and battery_soc > 0.2:
                action = BatteryAction.DISCHARGE_TO_LOAD
                power = min(net_load_kw, battery.get_available_discharge_power())
            elif not is_peak and battery_soc < 0.8 and hour < 14:
                action = BatteryAction.CHARGE_FROM_GRID
                power = min(30, battery.get_available_charge_power())
            else:
                action = BatteryAction.IDLE
                power = 0
            
            batt_result = battery.step(action=action, power_kw=power)
            battery_soc = batt_result.soc_after
            battery_power = batt_result.power_actual_kw
            
            # Grid
            if action == BatteryAction.DISCHARGE_TO_LOAD:
                grid_import = max(0, net_load_kw - battery_power)
                grid_export = 0
            elif action == BatteryAction.CHARGE_FROM_GRID:
                grid_import = max(0, net_load_kw) + battery_power
                grid_export = 0
            elif action == BatteryAction.CHARGE_FROM_PV:
                grid_import = max(0, net_load_kw)
                grid_export = max(0, -net_load_kw - battery_power)
            else:
                grid_import = max(0, net_load_kw)
                grid_export = max(0, -net_load_kw)
            
            # Cost
            import_cost = grid_import * current_price
            export_credit = grid_export * current_price * 0.4
            cumulative_import_cost += import_cost
            cumulative_export_credit += export_credit
            
            # Result
            result = {
                'timestamp': sim_datetime,
                'building_load_kw': building_load_kw,
                'pv_power_kw': pv_power_kw,
                'battery_action': action.name,
                'battery_soc': battery_soc,
                'grid_import_kw': grid_import,
                'grid_export_kw': grid_export,
                'price_per_kwh': current_price,
                'cumulative_cost': cumulative_import_cost - cumulative_export_credit,
                'outdoor_temp': outdoor_temp,
                'zone_temp_avg': zone_temp_avg,
                'cooling_setpoint': cooling_setpoint,
                'heating_setpoint': heating_setpoint
            }
            results.append(result)
            
            # Update dashboard
            dashboard.update(result)
            
            # Log
            if step % 12 == 0:
                print(f"Step {step}: {sim_datetime.strftime('%m/%d %H:%M')} | "
                      f"Bldg: {building_load_kw:.0f}kW | PV: {pv_power_kw:.0f}kW | "
                      f"SOC: {battery_soc:.0%} | Cost: ${cumulative_import_cost - cumulative_export_credit:.2f}")
            
            # Small delay to allow live dashboard to update
            import time
            time.sleep(0.05)  # 50ms delay per step
            
            # Step EnergyPlus
            obs, reward, done, info = env.step({
                'cooling_setpoint': cooling_setpoint,
                'heating_setpoint': heating_setpoint
            })
            
            step += 1
        
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"Total Steps: {step}")
        print(f"Net Cost: ${cumulative_import_cost - cumulative_export_credit:.2f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        print(f"Results saved to {output_dir}/results.csv")
        
        # Generate final Plotly visualization
        dashboard.finalize()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Live Integrated Simulation')
    parser.add_argument('--idf', type=str, default='models/RefBldgMediumOfficeNew2004_Chicago.idf')
    parser.add_argument('--epw', type=str, default='weather/chicago/TMY_lat41.88_lon-87.63.epw')
    parser.add_argument('--output', type=str, default='outputs/live_sim')
    parser.add_argument('--pv-kw', type=float, default=100.0)
    parser.add_argument('--battery-kwh', type=float, default=200.0)
    parser.add_argument('--battery-kw', type=float, default=50.0)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--no-browser', action='store_true')
    
    args = parser.parse_args()
    
    run_live_simulation(
        idf_path=args.idf,
        epw_path=args.epw,
        output_dir=args.output,
        pv_capacity_kw=args.pv_kw,
        battery_capacity_kwh=args.battery_kwh,
        battery_power_kw=args.battery_kw,
        max_steps=args.max_steps,
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()
