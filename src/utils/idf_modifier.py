"""
Utility functions for modifying EnergyPlus IDF files for custom simulation periods.
"""

import os
import shutil
from datetime import datetime, timedelta


def create_custom_idf(original_idf_path, start_month, start_day, end_month, end_day, output_dir=None):
    """
    Create a modified IDF file with custom simulation period.
    
    Args:
        original_idf_path: Path to original IDF file
        start_month: Start month (1-12)
        start_day: Start day (1-31)
        end_month: End month (1-12)
        end_day: End day (1-31)
        output_dir: Directory to save modified IDF (defaults to same as original)
    
    Returns:
        Path to modified IDF file
    """
    if output_dir is None:
        output_dir = os.path.dirname(original_idf_path)
    
    # Read original IDF
    with open(original_idf_path, 'r') as f:
        idf_content = f.read()
    
    # Debug: Count original SizingPeriod objects
    original_sizing = [line for line in idf_content.split('\n') if 'SizingPeriod' in line]
    print(f"Original IDF has {len(original_sizing)} SizingPeriod objects")
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(original_idf_path))[0]
    modified_idf_path = os.path.join(output_dir, f"{base_name}_custom_{start_month}{start_day:02d}_to_{end_month}{end_day:02d}.idf")
    
    # Find and replace RunPeriod section
    lines = idf_content.split('\n')
    new_lines = []
    in_runperiod = False
    runperiod_found = False
    
    for line in lines:
        if line.strip().startswith('RunPeriod,'):
            in_runperiod = True
            runperiod_found = True
            # Replace with custom run period
            new_lines.extend([
                '  RunPeriod,',
                f'    CUSTOM_PERIOD,          !- Name',
                f'    {start_month},            !- Begin Month',
                f'    {start_day},              !- Begin Day of Month',
                f'    ,                        !- Begin Year',
                f'    {end_month},              !- End Month',
                f'    {end_day},                !- End Day of Month',
                f'    ,                        !- End Year',
                '    Sunday,                  !- Day of Week for Start Day',
                '    No,                      !- Use Weather File Holidays and Special Days',
                '    No,                      !- Use Weather File Daylight Saving Period',
                '    No,                      !- Apply Weekend Holiday Rule',
                '    Yes,                     !- Use Weather File Rain Indicators',
                '    Yes;                     !- Use Weather File Snow Indicators',
                ''
            ])
        elif in_runperiod and line.strip().endswith(';'):
            # End of RunPeriod block - skip this line since we already added the semicolon
            in_runperiod = False
        elif in_runperiod:
            # Skip lines within the original RunPeriod block
            continue
        else:
            # Keep all other lines (including SizingPeriod objects)
            new_lines.append(line)
    
    if not runperiod_found:
        raise ValueError("No RunPeriod found in IDF file")
    
    # Check if SizingPeriod objects are preserved
    sizing_periods = [line for line in new_lines if 'SizingPeriod:' in line]
    print(f"Preserved {len(sizing_periods)} SizingPeriod objects")
    if sizing_periods:
        print(f"First SizingPeriod: {sizing_periods[0].strip()}")
    else:
        # Check for any SizingPeriod lines
        all_sizing = [line for line in new_lines if 'SizingPeriod' in line]
        print(f"Found {len(all_sizing)} lines with 'SizingPeriod'")
        if all_sizing:
            print(f"First SizingPeriod line: {all_sizing[0].strip()}")
    
    # Write modified IDF
    with open(modified_idf_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')
    
    print(f"Created custom IDF: {modified_idf_path}")
    print(f"Simulation period: {start_month}/{start_day} to {end_month}/{end_day}")
    
    return modified_idf_path


def calculate_simulation_days(start_month, start_day, end_month, end_day):
    """
    Calculate the number of days between start and end dates.
    
    Args:
        start_month: Start month (1-12)
        start_day: Start day (1-31)
        end_month: End month (1-12)
        end_day: End day (1-31)
    
    Returns:
        Number of days in simulation period
    """
    # Assume same year for simplicity
    start_date = datetime(2023, start_month, start_day)
    end_date = datetime(2023, end_month, end_day)
    
    if end_date < start_date:
        # If end date is before start, assume next year
        end_date = datetime(2024, end_month, end_day)
    
    return (end_date - start_date).days + 1


def get_season_info(month):
    """
    Get season information for a given month.
    
    Args:
        month: Month number (1-12)
    
    Returns:
        Dictionary with season info
    """
    seasons = {
        12: {"name": "Winter", "description": "Heating season"},
        1: {"name": "Winter", "description": "Heating season"},
        2: {"name": "Winter", "description": "Heating season"},
        3: {"name": "Spring", "description": "Transition season"},
        4: {"name": "Spring", "description": "Mild weather"},
        5: {"name": "Spring", "description": "Mild weather"},
        6: {"name": "Summer", "description": "Cooling season"},
        7: {"name": "Summer", "description": "Cooling season"},
        8: {"name": "Summer", "description": "Cooling season"},
        9: {"name": "Fall", "description": "Transition season"},
        10: {"name": "Fall", "description": "Mild weather"},
        11: {"name": "Fall", "description": "Transition season"}
    }
    
    return seasons.get(month, {"name": "Unknown", "description": "Unknown"})
