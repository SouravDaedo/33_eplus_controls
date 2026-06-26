"""
Utility functions for modifying EnergyPlus IDF files for custom simulation periods.
"""

import os
import re
import shutil
from datetime import datetime, timedelta


def _has_output_variable(lines, variable_name):
    pattern = re.compile(
        r'^\s*Output:Variable\s*,[^,;]*,\s*' + re.escape(variable_name) + r'\s*,',
        re.IGNORECASE,
    )
    return any(pattern.search(line) for line in lines if not line.strip().startswith('!'))


def _ensure_output_variable(lines, variable_name, key='*', frequency='Timestep', comment=None):
    if _has_output_variable(lines, variable_name):
        return
    if comment:
        lines.append('')
        lines.append(comment)
    lines.append(f'  Output:Variable,{key},{variable_name},{frequency};')


def _has_output_meter(lines, meter_name):
    pattern = re.compile(
        r'^\s*Output:Meter\s*,\s*' + re.escape(meter_name) + r'\s*,',
        re.IGNORECASE,
    )
    return any(pattern.search(line) for line in lines if not line.strip().startswith('!'))


def _ensure_output_meter(lines, meter_name, frequency='Timestep', comment=None):
    if _has_output_meter(lines, meter_name):
        return
    if comment:
        lines.append('')
        lines.append(comment)
    lines.append(f'  Output:Meter,{meter_name},{frequency};')


def _apply_current_energyplus_compatibility(lines):
    """Apply small IDF compatibility fixes for the installed EnergyPlus schema."""
    updated_lines = []
    zone_averaged_count = 0

    for line in lines:
        if 'ZoneAveraged' in line and 'Mean Radiant Temperature Calculation Type' in line:
            line = line.replace('ZoneAveraged', 'EnclosureAveraged')
            zone_averaged_count += 1
        updated_lines.append(line)

    if zone_averaged_count:
        print(
            "EnergyPlus compatibility: replaced "
            f"{zone_averaged_count} People MRT entries from ZoneAveraged to EnclosureAveraged"
        )

    return updated_lines


def create_custom_idf(
    original_idf_path,
    start_month,
    start_day,
    end_month,
    end_day,
    output_dir=None,
    outdoor_co2_ppm=None,
    outdoor_co2_csv_path=None,
    outdoor_co2_fallback_ppm=None,
):
    """
    Create a modified IDF file with custom simulation period.

    Args:
        original_idf_path: Path to original IDF file
        start_month: Start month (1-12)
        start_day: Start day (1-31)
        end_month: End month (1-12)
        end_day: End day (1-31)
        output_dir: Directory to save modified IDF (defaults to same as original)
        outdoor_co2_ppm: If set (and outdoor_co2_csv_path not set), constant outdoor CO2 (ppm)
        outdoor_co2_csv_path: If set, use time-varying outdoor CO2 from CSV (Schedule:File)
        outdoor_co2_fallback_ppm: Required ppm value when building a schedule from CSV with missing hours

    Returns:
        Path to modified IDF file
    """
    if output_dir is None:
        output_dir = os.path.dirname(original_idf_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read original IDF
    with open(original_idf_path, 'r') as f:
        idf_content = f.read()

    # Debug: Count original SizingPeriod objects
    original_sizing = [line for line in idf_content.split('\n') if 'SizingPeriod' in line]
    print(f"Original IDF has {len(original_sizing)} SizingPeriod objects")

    # Create output filename
    base_name = os.path.splitext(os.path.basename(original_idf_path))[0]
    modified_idf_path = os.path.join(output_dir, f"{base_name}_custom_{start_month}{start_day:02d}_to_{end_month}{end_day:02d}.idf")

    # Time-varying outdoor CO2: write 8760-hour schedule file and we'll replace Schedule:Constant with Schedule:File
    schedule_file_name = None
    if outdoor_co2_csv_path:
        if outdoor_co2_fallback_ppm is None:
            raise ValueError(
                "outdoor_co2_fallback_ppm is required when outdoor_co2_csv_path is set"
            )
        from src.utils.outdoor_co2_schedule import write_ep_outdoor_co2_schedule
        schedule_file_name = "outdoor_co2_schedule.csv"
        out_schedule_path = os.path.join(output_dir, schedule_file_name)
        write_ep_outdoor_co2_schedule(
            outdoor_co2_csv_path,
            out_schedule_path,
            fallback_ppm=outdoor_co2_fallback_ppm,
            year=2023,
        )
        print(f"Outdoor CO2: time-varying from {outdoor_co2_csv_path} -> {schedule_file_name}")

    # Find and replace RunPeriod section; optionally replace or override outdoor CO2 schedule
    lines = idf_content.split('\n')
    new_lines = []
    in_runperiod = False
    runperiod_found = False
    constant_block_buffer = []  # collect Schedule:Constant + 3 lines, then decide

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('RunPeriod,'):
            in_runperiod = True
            runperiod_found = True
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
            i += 1
        elif in_runperiod:
            if 'Use Weather File Snow Indicators' in line:
                in_runperiod = False
            i += 1
            continue
        elif line.strip().startswith('Schedule:Constant,'):
            # Buffer this line and the next 3 to see if it's Outdoor CO2 Schedule
            constant_block_buffer = [line]
            i += 1
            for _ in range(3):
                if i < len(lines):
                    constant_block_buffer.append(lines[i])
                    i += 1
            # Check if name line (second line) contains "Outdoor CO2 Schedule"
            name_line = constant_block_buffer[1] if len(constant_block_buffer) > 1 else ''
            is_outdoor_co2 = 'Outdoor CO2 Schedule' in name_line
            if is_outdoor_co2 and schedule_file_name:
                new_lines.append('  Schedule:File,')
                new_lines.append('    Outdoor CO2 Schedule,    !- Name')
                new_lines.append('    ,                        !- Schedule Type Limits Name')
                new_lines.append(f'    {schedule_file_name},    !- File Name')
                new_lines.append('    1,                        !- Column Number')
                new_lines.append('    0,                        !- Rows to Skip at Top')
                new_lines.append('    8760,                     !- Number of Hours of Data')
                new_lines.append('    Comma;                    !- Column Separator')
            else:
                # Emit block; override hourly value if constant outdoor CO2 and this is the outdoor schedule
                for j, bline in enumerate(constant_block_buffer):
                    if is_outdoor_co2 and outdoor_co2_ppm is not None and '!- Hourly Value {ppm}' in bline:
                        m = re.match(r'^(\s*)\d+;(.*)$', bline)
                        if m:
                            bline = m.group(1) + str(int(outdoor_co2_ppm)) + ';' + m.group(2)
                    new_lines.append(bline)
            constant_block_buffer = []
            continue
        else:
            if outdoor_co2_ppm is not None and not outdoor_co2_csv_path and '!- Hourly Value {ppm}' in line:
                m = re.match(r'^(\s*)\d+;(.*)$', line)
                if m:
                    line = m.group(1) + str(int(outdoor_co2_ppm)) + ';' + m.group(2)
            new_lines.append(line)
            i += 1

    if not runperiod_found:
        raise ValueError("No RunPeriod found in IDF file")

    new_lines = _apply_current_energyplus_compatibility(new_lines)

    # Check if SizingPeriod objects are preserved
    sizing_periods = [line for line in new_lines if 'SizingPeriod:' in line]
    print(f"Preserved {len(sizing_periods)} SizingPeriod objects")
    if sizing_periods:
        print(f"First SizingPeriod: {sizing_periods[0].strip()}")
    else:
        all_sizing = [line for line in new_lines if 'SizingPeriod' in line]
        print(f"Found {len(all_sizing)} lines with 'SizingPeriod'")
        if all_sizing:
            print(f"First SizingPeriod line: {all_sizing[0].strip()}")

    # Ensure required RL variables are declared. EnergyPlus Python API returns
    # invalid handles for variables that are only referenced by EMS or reports.
    _ensure_output_variable(
        new_lines,
        'Facility Total Electricity Demand Rate',
        comment='! Injected for RL: facility total electricity demand rate (W)',
    )
    _ensure_output_meter(
        new_lines,
        'NaturalGas:Facility',
        comment='! Injected for RL: facility natural gas meter (J per timestep)',
    )

    weather_variables = [
        'Site Outdoor Air Drybulb Temperature',
        'Site Outdoor Air Relative Humidity',
        'Site Sky Temperature',
    ]
    for idx, variable_name in enumerate(weather_variables):
        _ensure_output_variable(
            new_lines,
            variable_name,
            comment='! Injected for RL state: site weather variables (required for Python API handle)' if idx == 0 else None,
        )

    # Write modified IDF
    with open(modified_idf_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

    print(f"Created custom IDF: {modified_idf_path}")
    print(f"Simulation period: {start_month}/{start_day} to {end_month}/{end_day}")
    if outdoor_co2_csv_path and schedule_file_name:
        print(f"Outdoor CO2: time-varying from CSV ({schedule_file_name})")
    elif outdoor_co2_ppm is not None:
        print(f"Outdoor CO2 set to {outdoor_co2_ppm} ppm")

    return modified_idf_path


def end_date_from_duration(start_month, start_day, duration_hours, year=2023):
    """
    Compute end (month, day) from start date and duration in hours.
    RunPeriod must span at least duration_hours; we use whole days.
    """
    start = datetime(year, start_month, start_day)
    end = start + timedelta(hours=int(duration_hours))
    return end.month, end.day


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
