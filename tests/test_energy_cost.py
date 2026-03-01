"""
Test script for Energy Price Model with Building Simulation Integration

Demonstrates PV + Battery + RTP pricing for energy cost optimization.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from energy_price_model import (
    EnergyPriceModel,
    PricingConfig,
    PricingType,
    create_tou_pricing,
    create_rtp_pricing,
    create_wholesale_pricing
)
from battery_model import create_battery, BatteryAction
from solar_pv_model import create_pv_system


def test_pricing_types():
    """Test different pricing structures."""
    print("\n" + "=" * 60)
    print("TEST 1: Pricing Structure Comparison")
    print("=" * 60)
    
    # Create different pricing models
    flat = EnergyPriceModel(PricingConfig(pricing_type=PricingType.FLAT, flat_rate=0.12))
    tou = create_tou_pricing(peak_price=0.30, off_peak_price=0.08)
    rtp = create_rtp_pricing(base_price=0.12, volatility=0.4)
    wholesale = create_wholesale_pricing(base_price=0.05)
    
    models = [
        ("Flat Rate", flat),
        ("Time-of-Use", tou),
        ("Real-Time Pricing", rtp),
        ("Wholesale", wholesale)
    ]
    
    # Compare prices across a day
    print(f"\n{'Hour':<6}", end="")
    for name, _ in models:
        print(f"{name:>15}", end="")
    print()
    print("-" * 70)
    
    for hour in range(0, 24, 2):
        ts = datetime(2024, 7, 15, hour, 0)
        print(f"{hour:>4}:00", end="")
        for _, model in models:
            price = model.get_price(ts)
            print(f"  ${price.price_per_kwh:>10.3f}", end="")
        print()


def test_cost_calculation():
    """Test energy cost calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: Energy Cost Calculation")
    print("=" * 60)
    
    price_model = create_tou_pricing(
        peak_price=0.28,
        off_peak_price=0.08,
        peak_hours=(14, 20),
        feed_in_tariff=0.06
    )
    
    # Simulate building with constant 50 kW load
    print("\nScenario: 50 kW constant load, no solar")
    print("-" * 50)
    
    total_cost = 0
    for hour in range(24):
        ts = datetime(2024, 7, 15, hour, 0)
        result = price_model.calculate_cost(ts, energy_consumed_kwh=50)
        total_cost += result.net_cost
    
    summary = price_model.get_cost_summary()
    print(f"Daily consumption: {summary['total_consumed_kwh']:,.0f} kWh")
    print(f"Daily cost: ${summary['net_energy_cost']:,.2f}")
    print(f"Average price: ${summary['average_price_per_kwh']:.3f}/kWh")


def test_pv_battery_rtp_integration():
    """Test full integration: PV + Battery + RTP pricing."""
    print("\n" + "=" * 60)
    print("TEST 3: PV + Battery + RTP Integration")
    print("=" * 60)
    
    # Find weather file
    weather_file = None
    for wf in ["weather/chicago/TMY_lat41.88_lon-87.63.epw",
               "weather/atlanta_2023/hourly_lat33.75_lon-84.39_2023-2023.epw"]:
        if Path(wf).exists():
            weather_file = wf
            break
    
    if not weather_file:
        print("No weather file found. Skipping integration test.")
        return
    
    # Create models
    pv = create_pv_system(capacity_kw=75)  # 75 kW PV system
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
    print(f"  Building Load: Variable (30-80 kW)")
    
    # Simulate one summer day
    start = datetime(weather_year, 7, 15, 0, 0)
    
    print(f"\nSimulating {start.date()}:")
    print("-" * 100)
    print(f"{'Hour':>5} {'Load':>8} {'PV':>8} {'Price':>8} {'Action':<18} {'Batt':>8} {'SOC':>7} {'Grid':>8} {'Cost':>8}")
    print("-" * 100)
    
    daily_results = []
    
    for hour in range(24):
        ts = start + timedelta(hours=hour)
        
        # Variable building load (office pattern)
        if 8 <= hour <= 18:
            building_load = 60 + 20 * (1 - abs(hour - 13) / 5)  # Peak at 1 PM
        else:
            building_load = 30  # Base load
        
        # Get PV production
        pv_state = pv.get_power_at_timestep(ts)
        pv_power = pv_state.ac_power_kw
        
        # Get current price
        price_state = price_model.get_price(ts)
        current_price = price_state.price_per_kwh
        
        # Simple control strategy based on price
        net_load = building_load - pv_power
        
        if net_load > 0:
            # Need power from grid or battery
            if current_price > 0.15 and battery.get_soc() > 0.2:
                # High price - discharge battery
                action = BatteryAction.DISCHARGE_TO_LOAD
                power = min(net_load, battery.get_available_discharge_power())
            else:
                # Low price - just use grid
                action = BatteryAction.IDLE
                power = 0
        else:
            # Excess PV
            if current_price < 0.08 and battery.get_soc() < 0.85:
                # Low price - charge battery from PV
                action = BatteryAction.CHARGE_FROM_PV
                power = min(abs(net_load), battery.get_available_charge_power())
            elif battery.get_soc() < 0.9:
                # Charge battery with excess
                action = BatteryAction.CHARGE_FROM_PV
                power = min(abs(net_load), battery.get_available_charge_power())
            else:
                # Battery full - export
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
        
        daily_results.append({
            'hour': hour,
            'load': building_load,
            'pv': pv_power,
            'price': current_price,
            'action': action.name,
            'batt_power': batt_result.power_actual_kw,
            'soc': batt_result.soc_after,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'cost': cost_result.net_cost
        })
        
        # Print hourly results
        grid_net = grid_import - grid_export
        print(f"{hour:>5} {building_load:>8.1f} {pv_power:>8.1f} ${current_price:>6.3f} "
              f"{action.name:<18} {batt_result.power_actual_kw:>8.1f} {batt_result.soc_after:>6.1%} "
              f"{grid_net:>8.1f} ${cost_result.net_cost:>7.2f}")
    
    # Daily summary
    summary = price_model.get_cost_summary()
    total_load = sum(r['load'] for r in daily_results)
    total_pv = sum(r['pv'] for r in daily_results)
    
    print("\n" + "=" * 60)
    print("DAILY SUMMARY")
    print("=" * 60)
    print(f"Total Building Load: {total_load:,.0f} kWh")
    print(f"Total PV Generation: {total_pv:,.0f} kWh")
    print(f"Grid Import: {summary['total_consumed_kwh']:,.0f} kWh")
    print(f"Grid Export: {summary['total_exported_kwh']:,.0f} kWh")
    print(f"Import Cost: ${summary['total_import_cost']:,.2f}")
    print(f"Export Credit: ${summary['total_export_credit']:,.2f}")
    print(f"Net Energy Cost: ${summary['net_energy_cost']:,.2f}")
    print(f"Battery Final SOC: {battery.get_soc():.1%}")
    print(f"Battery Cycles: {battery.state.cycles:.2f}")


def test_cost_comparison():
    """Compare costs with and without battery/PV."""
    print("\n" + "=" * 60)
    print("TEST 4: Cost Comparison - With vs Without DER")
    print("=" * 60)
    
    # Find weather file
    weather_file = None
    for wf in ["weather/chicago/TMY_lat41.88_lon-87.63.epw",
               "weather/atlanta_2023/hourly_lat33.75_lon-84.39_2023-2023.epw"]:
        if Path(wf).exists():
            weather_file = wf
            break
    
    if not weather_file:
        print("No weather file found. Skipping comparison test.")
        return
    
    # Scenario 1: Grid only
    print("\nScenario 1: Grid Only (no PV, no battery)")
    price_model_1 = create_tou_pricing(peak_price=0.28, off_peak_price=0.08)
    
    for hour in range(24):
        ts = datetime(2024, 7, 15, hour, 0)
        load = 60 if 8 <= hour <= 18 else 30
        price_model_1.calculate_cost(ts, energy_consumed_kwh=load)
    
    summary_1 = price_model_1.get_cost_summary()
    
    # Scenario 2: With PV only
    print("Scenario 2: With 75 kW PV (no battery)")
    pv = create_pv_system(capacity_kw=75)
    pv.load_weather(weather_file)
    weather_year = pv.weather_data.index[0].year
    
    price_model_2 = create_tou_pricing(peak_price=0.28, off_peak_price=0.08)
    
    for hour in range(24):
        ts = datetime(weather_year, 7, 15, hour, 0)
        load = 60 if 8 <= hour <= 18 else 30
        pv_power = pv.get_power_at_timestep(ts).ac_power_kw
        
        grid_import = max(0, load - pv_power)
        grid_export = max(0, pv_power - load)
        price_model_2.calculate_cost(ts, grid_import, grid_export)
    
    summary_2 = price_model_2.get_cost_summary()
    
    # Scenario 3: With PV + Battery
    print("Scenario 3: With 75 kW PV + 200 kWh Battery")
    price_model_3 = create_tou_pricing(peak_price=0.28, off_peak_price=0.08)
    battery = create_battery(capacity_kwh=200, max_power_kw=50, timestep_minutes=60)
    
    for hour in range(24):
        ts = datetime(weather_year, 7, 15, hour, 0)
        load = 60 if 8 <= hour <= 18 else 30
        pv_power = pv.get_power_at_timestep(ts).ac_power_kw
        price = price_model_3.get_price(ts)
        
        net_load = load - pv_power
        
        # Smart battery control
        if net_load > 0 and price.is_peak and battery.get_soc() > 0.15:
            action = BatteryAction.DISCHARGE_TO_LOAD
            power = min(net_load, battery.get_available_discharge_power())
        elif net_load < 0 and battery.get_soc() < 0.9:
            action = BatteryAction.CHARGE_FROM_PV
            power = min(abs(net_load), battery.get_available_charge_power())
        elif not price.is_peak and battery.get_soc() < 0.5:
            action = BatteryAction.CHARGE_FROM_GRID
            power = min(20, battery.get_available_charge_power())
        else:
            action = BatteryAction.IDLE
            power = 0
        
        batt_result = battery.step(action=action, power_kw=power)
        
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
        
        price_model_3.calculate_cost(ts, grid_import, grid_export)
    
    summary_3 = price_model_3.get_cost_summary()
    
    # Print comparison
    print("\n" + "-" * 60)
    print(f"{'Scenario':<30} {'Consumed':>12} {'Exported':>12} {'Net Cost':>12}")
    print("-" * 60)
    print(f"{'Grid Only':<30} {summary_1['total_consumed_kwh']:>10.0f} kWh "
          f"{summary_1['total_exported_kwh']:>10.0f} kWh ${summary_1['net_energy_cost']:>10.2f}")
    print(f"{'With PV':<30} {summary_2['total_consumed_kwh']:>10.0f} kWh "
          f"{summary_2['total_exported_kwh']:>10.0f} kWh ${summary_2['net_energy_cost']:>10.2f}")
    print(f"{'With PV + Battery':<30} {summary_3['total_consumed_kwh']:>10.0f} kWh "
          f"{summary_3['total_exported_kwh']:>10.0f} kWh ${summary_3['net_energy_cost']:>10.2f}")
    
    # Savings
    savings_pv = summary_1['net_energy_cost'] - summary_2['net_energy_cost']
    savings_pv_batt = summary_1['net_energy_cost'] - summary_3['net_energy_cost']
    
    print("\n" + "-" * 60)
    print(f"Savings with PV: ${savings_pv:.2f} ({savings_pv/summary_1['net_energy_cost']*100:.1f}%)")
    print(f"Savings with PV + Battery: ${savings_pv_batt:.2f} ({savings_pv_batt/summary_1['net_energy_cost']*100:.1f}%)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ENERGY PRICE MODEL TEST SUITE")
    print("=" * 60)
    
    test_pricing_types()
    test_cost_calculation()
    test_pv_battery_rtp_integration()
    test_cost_comparison()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
