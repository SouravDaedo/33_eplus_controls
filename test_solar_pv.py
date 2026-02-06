"""
Test script for Solar PV Model

Demonstrates the PV model with different configurations and weather files.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from solar_pv_model import (
    SolarPVModel, 
    PVSystemConfig, 
    create_pv_system,
    EPWReader
)


def test_epw_reader():
    """Test EPW file reading."""
    print("\n" + "=" * 60)
    print("TEST 1: EPW Weather File Reader")
    print("=" * 60)
    
    weather_files = [
        "weather/chicago/TMY_lat41.88_lon-87.63.epw",
        "weather/atlanta_2023/hourly_lat33.75_lon-84.39_2023-2023.epw",
    ]
    
    for wf in weather_files:
        if Path(wf).exists():
            print(f"\nReading: {wf}")
            reader = EPWReader(wf)
            loc = reader.get_location()
            data = reader.get_data()
            
            print(f"  Location: {loc['city']}, {loc['state']}, {loc['country']}")
            print(f"  Coordinates: ({loc['latitude']:.2f}, {loc['longitude']:.2f})")
            print(f"  Elevation: {loc['elevation']:.0f} m")
            print(f"  Timezone: UTC{loc['timezone']:+.0f}")
            print(f"  Data points: {len(data)}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            print(f"  GHI range: {data['ghi'].min():.0f} - {data['ghi'].max():.0f} W/m²")
            return wf  # Return first valid file
    
    print("No weather files found!")
    return None


def test_pv_system_config():
    """Test PV system configuration."""
    print("\n" + "=" * 60)
    print("TEST 2: PV System Configuration")
    print("=" * 60)
    
    # Default config
    config1 = PVSystemConfig()
    print("\nDefault Configuration:")
    print(f"  Capacity: {config1.system_capacity_kw} kW")
    print(f"  Tilt: {config1.tilt_deg}°")
    print(f"  Azimuth: {config1.azimuth_deg}°")
    print(f"  Losses: {config1.system_losses_pct}%")
    print(f"  DC/AC Ratio: {config1.dc_ac_ratio}")
    print(f"  Temp Coeff: {config1.temp_coeff_pmax} /°C")
    
    # Custom config
    config2 = PVSystemConfig(
        system_capacity_kw=250,
        tilt_deg=35,
        azimuth_deg=180,
        module_type=1,  # Premium
        system_losses_pct=10,
        dc_ac_ratio=1.3
    )
    print("\nCustom Configuration (Premium modules):")
    print(f"  Capacity: {config2.system_capacity_kw} kW")
    print(f"  Tilt: {config2.tilt_deg}°")
    print(f"  Azimuth: {config2.azimuth_deg}°")
    print(f"  Losses: {config2.system_losses_pct}%")
    print(f"  DC/AC Ratio: {config2.dc_ac_ratio}")
    print(f"  Temp Coeff: {config2.temp_coeff_pmax} /°C (better for premium)")


def test_single_timestep(weather_file: str):
    """Test single timestep calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: Single Timestep Calculation")
    print("=" * 60)
    
    # Create PV system
    pv = create_pv_system(capacity_kw=100)
    location = pv.load_weather(weather_file)
    
    print(f"\nPV System: 100 kW at {location['city']}")
    print(f"Tilt: {pv.config.tilt_deg:.1f}° (set to latitude)")
    
    # Get the year from the weather file
    weather_year = pv.weather_data.index[0].year
    
    # Test a few timestamps using the actual year in the weather file
    test_times = [
        datetime(weather_year, 6, 21, 12, 0),  # Summer solstice noon
        datetime(weather_year, 6, 21, 8, 0),   # Summer morning
        datetime(weather_year, 12, 21, 12, 0), # Winter solstice noon
        datetime(weather_year, 3, 21, 12, 0),  # Equinox noon
    ]
    
    print("\nTimestep Results:")
    print("-" * 80)
    print(f"{'Timestamp':<20} {'GHI':>8} {'POA':>8} {'Cell T':>8} {'DC kW':>8} {'AC kW':>8}")
    print("-" * 80)
    
    for ts in test_times:
        try:
            state = pv.get_power_at_timestep(ts)
            print(f"{ts.strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{state.ghi:>8.0f} {state.poa_irradiance:>8.0f} "
                  f"{state.cell_temperature_c:>8.1f} "
                  f"{state.dc_power_kw:>8.1f} {state.ac_power_kw:>8.1f}")
        except Exception as e:
            print(f"{ts.strftime('%Y-%m-%d %H:%M'):<20} Error: {e}")


def test_daily_simulation(weather_file: str):
    """Test daily simulation."""
    print("\n" + "=" * 60)
    print("TEST 4: Daily Simulation (Summer Day)")
    print("=" * 60)
    
    pv = create_pv_system(capacity_kw=100)
    pv.load_weather(weather_file)
    
    # Get the year from the weather file
    weather_year = pv.weather_data.index[0].year
    
    # Simulate one summer day
    start = datetime(weather_year, 7, 15, 0, 0)
    end = datetime(weather_year, 7, 15, 23, 0)
    
    print(f"\nSimulating: {start.date()}")
    results = pv.simulate(start, end)
    
    print("\nHourly Production:")
    print("-" * 60)
    for idx, row in results.iterrows():
        if row['ac_power_kw'] > 0:
            bar = '█' * int(row['ac_power_kw'] / 5)
            print(f"{idx.strftime('%H:%M')} | {row['ac_power_kw']:6.1f} kW | {bar}")
    
    daily_kwh = results['ac_power_kw'].sum()
    peak_kw = results['ac_power_kw'].max()
    print(f"\nDaily Total: {daily_kwh:.1f} kWh")
    print(f"Peak Power: {peak_kw:.1f} kW")


def test_annual_simulation(weather_file: str):
    """Test annual simulation."""
    print("\n" + "=" * 60)
    print("TEST 5: Annual Simulation")
    print("=" * 60)
    
    pv = create_pv_system(
        capacity_kw=100,
        azimuth_deg=180,
        losses_pct=14
    )
    location = pv.load_weather(weather_file)
    
    # Get the first full year from the weather file
    weather_year = pv.weather_data.index[0].year
    start = datetime(weather_year, 1, 1, 0, 0)
    end = datetime(weather_year, 12, 31, 23, 0)
    
    print(f"\nSimulating year {weather_year} for {location['city']}...")
    results = pv.simulate(start, end)
    
    # Annual summary
    annual = pv.get_annual_production()
    print(f"\nAnnual Summary:")
    print(f"  Total AC Energy: {annual['ac_energy_kwh']:,.0f} kWh")
    print(f"  Capacity Factor: {annual['capacity_factor']:.1%}")
    print(f"  Specific Yield: {annual['specific_yield_kwh_kwp']:,.0f} kWh/kWp")
    
    # Monthly breakdown
    monthly = pv.get_monthly_production()
    print(f"\nMonthly Production:")
    print("-" * 40)
    for idx, row in monthly.iterrows():
        bar = '█' * int(row['ac_energy_kwh'] / 500)
        print(f"{idx.strftime('%Y-%b'):>8} | {row['ac_energy_kwh']:>8,.0f} kWh | {bar}")


def test_different_configurations(weather_file: str):
    """Compare different PV configurations."""
    print("\n" + "=" * 60)
    print("TEST 6: Configuration Comparison")
    print("=" * 60)
    
    configs = [
        {"name": "South-facing, Latitude tilt", "azimuth": 180, "tilt": None},
        {"name": "South-facing, 20° tilt", "azimuth": 180, "tilt": 20},
        {"name": "South-facing, 45° tilt", "azimuth": 180, "tilt": 45},
        {"name": "West-facing, Latitude tilt", "azimuth": 270, "tilt": None},
        {"name": "East-facing, Latitude tilt", "azimuth": 90, "tilt": None},
    ]
    
    print(f"\nComparing 100 kW systems with different orientations:\n")
    print(f"{'Configuration':<35} {'Annual kWh':>12} {'Yield':>10}")
    print("-" * 60)
    
    for cfg in configs:
        pv = create_pv_system(
            capacity_kw=100,
            azimuth_deg=cfg['azimuth'],
            tilt_deg=cfg['tilt']
        )
        pv.load_weather(weather_file)
        pv.simulate()
        annual = pv.get_annual_production()
        
        print(f"{cfg['name']:<35} {annual['ac_energy_kwh']:>12,.0f} "
              f"{annual['specific_yield_kwh_kwp']:>10,.0f}")


def test_integration_with_battery():
    """Test integration with battery model."""
    print("\n" + "=" * 60)
    print("TEST 7: PV + Battery Integration Example")
    print("=" * 60)
    
    try:
        from battery_model import create_battery, BatteryAction
        
        # Find a weather file
        weather_file = None
        for wf in ["weather/chicago/TMY_lat41.88_lon-87.63.epw",
                   "weather/atlanta_2023/hourly_lat33.75_lon-84.39_2023-2023.epw"]:
            if Path(wf).exists():
                weather_file = wf
                break
        
        if not weather_file:
            print("No weather file found for integration test.")
            return
        
        # Create PV system
        pv = create_pv_system(capacity_kw=50)
        pv.load_weather(weather_file)
        
        # Get the year from the weather file
        weather_year = pv.weather_data.index[0].year
        
        # Create battery
        battery = create_battery(
            capacity_kwh=100,
            max_power_kw=25,
            efficiency=0.90,
            timestep_minutes=60  # Hourly to match weather data
        )
        
        print(f"\nSystem Configuration:")
        print(f"  PV: 50 kW")
        print(f"  Battery: 100 kWh, 25 kW max power")
        print(f"  Building Load: 30 kW constant (simplified)")
        
        # Simulate one day using the actual year from weather file
        building_load = 30  # kW constant
        start = datetime(weather_year, 7, 15, 0, 0)
        
        print(f"\nSimulating {start.date()}:")
        print("-" * 80)
        print(f"{'Hour':>5} {'PV kW':>8} {'Load':>8} {'Action':<20} {'Batt kW':>8} {'SOC':>8}")
        print("-" * 80)
        
        for hour in range(24):
            ts = start + timedelta(hours=hour)
            
            # Get PV production
            pv_state = pv.get_power_at_timestep(ts)
            pv_power = pv_state.ac_power_kw
            
            # Simple control logic
            net_power = pv_power - building_load
            
            if net_power > 0:
                # Excess PV - charge battery
                action = BatteryAction.CHARGE_FROM_PV
                power = min(net_power, battery.get_available_charge_power())
            elif net_power < 0:
                # Deficit - discharge battery
                action = BatteryAction.DISCHARGE_TO_LOAD
                power = min(abs(net_power), battery.get_available_discharge_power())
            else:
                action = BatteryAction.IDLE
                power = 0
            
            result = battery.step(
                action=action,
                power_kw=power,
                pv_available_kw=pv_power,
                load_demand_kw=building_load
            )
            
            print(f"{hour:>5} {pv_power:>8.1f} {building_load:>8.1f} "
                  f"{action.name:<20} {result.power_actual_kw:>8.1f} "
                  f"{result.soc_after:>7.1%}")
        
        print(f"\nBattery Final State:")
        print(f"  SOC: {battery.get_soc():.1%}")
        print(f"  Total charged: {battery.state.total_energy_charged_kwh:.1f} kWh")
        print(f"  Total discharged: {battery.state.total_energy_discharged_kwh:.1f} kWh")
        
    except ImportError:
        print("Battery model not found. Skipping integration test.")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SOLAR PV MODEL TEST SUITE")
    print("=" * 60)
    
    # Test 1: EPW Reader
    weather_file = test_epw_reader()
    
    if weather_file is None:
        print("\nNo weather files found. Cannot continue tests.")
        print("Please ensure EPW files exist in the weather/ directory.")
        return
    
    # Test 2: Configuration
    test_pv_system_config()
    
    # Test 3: Single timestep
    test_single_timestep(weather_file)
    
    # Test 4: Daily simulation
    test_daily_simulation(weather_file)
    
    # Test 5: Annual simulation
    test_annual_simulation(weather_file)
    
    # Test 6: Configuration comparison
    test_different_configurations(weather_file)
    
    # Test 7: Integration with battery
    test_integration_with_battery()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
