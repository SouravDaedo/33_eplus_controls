"""
Example usage of Dynamic Weather Scheduler

This demonstrates how to use the dynamic weather scheduler to run
RL HVAC episodes with different weather conditions.
"""

from dynamic_weather_scheduler import DynamicWeatherScheduler

def example_summer_training():
    """Example: Train RL agent on summer weather."""
    print("=== Summer Training Example ===")
    
    scheduler = DynamicWeatherScheduler()
    
    # Run 20 episodes during summer months (June-August)
    time_range = {
        'start_month': 6,   # June
        'start_day': 1,     # June 1st
        'end_month': 8,     # August  
        'end_day': 31       # August 31st
    }
    
    results = scheduler.run_dynamic_episodes(
        num_episodes=20,
        time_range=time_range
    )
    
    return results

def example_year_roundup():
    """Example: Test agent across all seasons."""
    print("=== Year-Round Test Example ===")
    
    scheduler = DynamicWeatherScheduler()
    
    # Run 12 episodes, one per month
    time_range = {
        'start_month': 1,   # January
        'start_day': 15,    # 15th of each month
        'end_month': 12,     # December
        'end_day': 15        # 15th of December
    }
    
    results = scheduler.run_dynamic_episodes(
        num_episodes=12,
        time_range=time_range
    )
    
    return results

def example_extreme_conditions():
    """Example: Test on extreme weather conditions."""
    print("=== Extreme Conditions Example ===")
    
    scheduler = DynamicWeatherScheduler()
    
    # Run episodes during extreme weather periods
    extreme_periods = [
        # Winter extreme
        {'start_month': 1, 'start_day': 15, 'end_month': 1, 'end_day': 20},
        # Summer extreme  
        {'start_month': 7, 'start_day': 15, 'end_month': 7, 'end_day': 20},
        # Spring transition
        {'start_month': 4, 'start_day': 1, 'end_month': 4, 'end_day': 7},
        # Fall transition
        {'start_month': 10, 'start_day': 1, 'end_month': 10, 'end_day': 7}
    ]
    
    all_results = []
    for i, period in enumerate(extreme_periods):
        print(f"\n--- Extreme Period {i+1}/4 ---")
        results = scheduler.run_dynamic_episodes(
            num_episodes=3,
            time_range=period
        )
        all_results.extend(results)
    
    return all_results

if __name__ == "__main__":
    print("Dynamic Weather Scheduler Examples")
    print("Choose an example to run:")
    print("1. Summer training (20 episodes)")
    print("2. Year-round test (12 episodes)")  
    print("3. Extreme conditions (12 episodes)")
    
    # Uncomment to run specific example
    # example_summer_training()
    # example_year_roundup()
    # example_extreme_conditions()
