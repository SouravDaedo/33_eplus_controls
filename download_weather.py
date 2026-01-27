"""
Download weather data for EnergyPlus simulations.

Data Sources:
- Open-Meteo: Global coverage, historical data from 1940 to present (5-day delay)
              Supports 2024/2025 data. Best for recent/current weather.
- PVGIS: Europe, Africa, Mediterranean, parts of Asia/Americas
         TMY and historical data up to 2023.

Features:
- Download EPW and CSV formats
- Specify date range (start/end dates)
- Automatic IDF run period update to match weather data
- Global coverage via Open-Meteo

Usage:
    # Download 2024 summer data for Atlanta (Open-Meteo)
    python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30 --output weather/atlanta_summer_2024

    # Download TMY data (PVGIS)
    python download_weather.py --lat 41.88 --lon -87.63 --tmy --output weather/chicago

    # Download and update IDF run period
    python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30 --update-idf model.idf
"""

import os
import sys
import argparse
import requests
from datetime import datetime, timedelta
import json

PVGIS_API_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"
OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"


def download_open_meteo(lat, lon, start_date, end_date, output_dir):
    """
    Download historical weather data from Open-Meteo API.
    Supports data from 1940 to present (with 5-day delay).
    Global coverage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open-Meteo hourly variables needed for EPW
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m", 
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
    ]
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly_vars),
        'timezone': 'auto',
    }
    
    print(f"Downloading from Open-Meteo: {start_date} to {end_date}...")
    
    try:
        response = requests.get(OPEN_METEO_API, params=params, timeout=120)
        
        if response.status_code != 200:
            print(f"  ✗ Failed: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"    Error: {error.get('reason', 'Unknown error')}")
            except:
                print(f"    Response: {response.text[:300]}")
            return {}
        
        data = response.json()
        
        # Save JSON
        json_filename = f"openmeteo_lat{lat}_lon{lon}_{start_date}_{end_date}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved JSON: {json_path}")
        
        # Convert to CSV
        csv_filename = f"weather_lat{lat}_lon{lon}_{start_date}_{end_date}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        convert_open_meteo_to_csv(data, csv_path)
        print(f"  ✓ Saved CSV: {csv_path}")
        
        # Convert to EPW
        epw_filename = f"weather_lat{lat}_lon{lon}_{start_date}_{end_date}.epw"
        epw_path = os.path.join(output_dir, epw_filename)
        convert_open_meteo_to_epw(data, epw_path, lat, lon)
        print(f"  ✓ Saved EPW: {epw_path}")
        
        return {
            'json': json_path,
            'csv': csv_path,
            'epw': epw_path,
            'start_date': start_date,
            'end_date': end_date,
        }
        
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return {}


def convert_open_meteo_to_csv(data, csv_path):
    """Convert Open-Meteo JSON response to CSV."""
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    
    if not times:
        return
    
    with open(csv_path, 'w') as f:
        # Header
        headers = ['time'] + [k for k in hourly.keys() if k != 'time']
        f.write(','.join(headers) + '\n')
        
        # Data rows
        for i, time in enumerate(times):
            row = [time]
            for key in headers[1:]:
                val = hourly.get(key, [None] * len(times))[i]
                row.append(str(val) if val is not None else '')
            f.write(','.join(row) + '\n')


def convert_open_meteo_to_epw(data, epw_path, lat, lon):
    """Convert Open-Meteo JSON response to EPW format."""
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    
    if not times:
        print("  ✗ No hourly data found")
        return
    
    elevation = data.get('elevation', 0)
    timezone = data.get('timezone', 'UTC')
    
    # Calculate UTC offset from timezone
    utc_offset = data.get('utc_offset_seconds', 0) / 3600
    
    with open(epw_path, 'w', newline='') as f:
        # EPW Header (8 lines)
        city = "OpenMeteo_Location"
        f.write(f"LOCATION,{city},-,-,OpenMeteo,999999,{lat},{lon},{utc_offset},{elevation}\n")
        f.write("DESIGN CONDITIONS,0\n")
        f.write("TYPICAL/EXTREME PERIODS,0\n")
        f.write("GROUND TEMPERATURES,0\n")
        f.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n")
        f.write(f"COMMENTS 1,Generated from Open-Meteo historical data\n")
        f.write(f"COMMENTS 2,Lat={lat} Lon={lon} Elevation={elevation}m\n")
        
        # Parse date range for DATA PERIODS
        first_date = datetime.fromisoformat(times[0])
        last_date = datetime.fromisoformat(times[-1])
        f.write(f"DATA PERIODS,1,1,Data,{first_date.strftime('%A')},{first_date.month}/{first_date.day},{last_date.month}/{last_date.day}\n")
        
        # Write hourly data
        for i, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str)
            
            # Get weather variables with defaults
            temp = hourly.get('temperature_2m', [20] * len(times))[i] or 20
            rh = hourly.get('relative_humidity_2m', [50] * len(times))[i] or 50
            dew_point = hourly.get('dew_point_2m', [10] * len(times))[i] or (temp - (100 - rh) / 5)
            pressure = hourly.get('surface_pressure', [101325] * len(times))[i] or 101325
            
            # Solar radiation
            ghi = hourly.get('global_tilted_irradiance', [0] * len(times))[i] or 0
            dni = hourly.get('direct_normal_irradiance', [0] * len(times))[i] or 0
            dhi = hourly.get('diffuse_radiation', [0] * len(times))[i] or 0
            
            # If GHI not available, estimate from direct + diffuse
            if ghi == 0 and (dni > 0 or dhi > 0):
                direct_horiz = hourly.get('direct_radiation', [0] * len(times))[i] or 0
                ghi = direct_horiz + dhi
            
            # Wind
            ws = hourly.get('wind_speed_10m', [0] * len(times))[i] or 0
            wd = hourly.get('wind_direction_10m', [0] * len(times))[i] or 0
            
            # Cloud cover
            cloud = hourly.get('cloud_cover', [0] * len(times))[i] or 0
            sky_cover = int(cloud / 10)  # Convert 0-100 to 0-10
            
            # Precipitation
            precip = hourly.get('precipitation', [0] * len(times))[i] or 0
            
            # EPW uses 1-24 hour format
            epw_hour = dt.hour + 1
            if epw_hour == 0:
                epw_hour = 24
            
            # EPW line format
            line = (
                f"{dt.year},{dt.month},{dt.day},{epw_hour},{dt.minute},"
                f"?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,"
                f"{temp:.1f},{dew_point:.1f},{rh:.0f},{pressure*100:.0f},"
                f"9999,9999,9999,"
                f"{ghi:.0f},{dni:.0f},{dhi:.0f},"
                f"999999,999999,999999,9999,"
                f"{wd:.0f},{ws:.1f},"
                f"{sky_cover},{sky_cover},9999,99999,9,999999999,"
                f"999,999,999,999,{precip:.1f},999,999\n"
            )
            f.write(line)


def download_pvgis_tmy(lat, lon, output_dir):
    """Download TMY (Typical Meteorological Year) data from PVGIS."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for fmt in ['epw', 'csv']:
        url = f"{PVGIS_API_BASE}/tmy"
        params = {
            'lat': lat,
            'lon': lon,
            'outputformat': fmt,
        }
        
        print(f"Downloading PVGIS TMY ({fmt.upper()})...")
        
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                filename = f"TMY_lat{lat}_lon{lon}.{fmt}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"  ✓ Saved: {filepath} ({len(response.content):,} bytes)")
                results[fmt] = filepath
            else:
                print(f"  ✗ Failed: HTTP {response.status_code}")
                try:
                    error = response.json()
                    print(f"    Error: {error.get('message', 'Unknown error')}")
                except:
                    pass
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return results


def update_idf_run_period(idf_path, start_month, start_day, end_month, end_day, year=None):
    """Update the RunPeriod in an IDF file to match weather data dates."""
    import re
    
    print(f"Updating IDF run period: {start_month}/{start_day} - {end_month}/{end_day}" + 
          (f" ({year})" if year else ""))
    
    with open(idf_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    year_str = str(year) if year else ''
    
    # Simple pattern to replace entire RunPeriod object
    simple_pattern = r'(RunPeriod,\s*\n\s*[^;]+;)'
    
    # Determine day of week for start date
    if year:
        try:
            start_dt = datetime(year, start_month, start_day)
            day_of_week = start_dt.strftime('%A')
        except:
            day_of_week = 'Sunday'
    else:
        day_of_week = 'Sunday'
    
    new_run_period = f"""RunPeriod,
    CustomPeriod,            !- Name
    {start_month},                       !- Begin Month
    {start_day},                       !- Begin Day of Month
    {year_str},                        !- Begin Year
    {end_month},                      !- End Month
    {end_day},                      !- End Day of Month
    {year_str},                        !- End Year
    {day_of_week},                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes;                     !- Use Weather File Snow Indicators"""
    
    new_content, count = re.subn(simple_pattern, new_run_period, content)
    
    if count > 0:
        with open(idf_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✓ Updated RunPeriod in {idf_path}")
        return True
    else:
        print(f"  ✗ Could not find RunPeriod object in {idf_path}")
        return False


def get_epw_date_range(epw_path):
    """Extract the date range from an EPW file."""
    with open(epw_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header (8 lines), find first and last data lines
    data_lines = [l for l in lines[8:] if l.strip() and ',' in l]
    
    if not data_lines:
        return None, None, None, None, None
    
    first = data_lines[0].split(',')
    last = data_lines[-1].split(',')
    
    try:
        start_year = int(first[0])
        start_month = int(first[1])
        start_day = int(first[2])
        end_year = int(last[0])
        end_month = int(last[1])
        end_day = int(last[2])
        
        year = start_year if start_year == end_year else None
        
        return start_month, start_day, end_month, end_day, year
    except (ValueError, IndexError):
        return None, None, None, None, None


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Download weather data for EnergyPlus simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  Open-Meteo: Global coverage, 1940 to present (5-day delay)
              Best for 2024/2025 and recent data
  PVGIS:      Europe/Africa/Mediterranean, TMY and historical up to 2023

Examples:
  # Download 2024 summer data for Atlanta
  python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30

  # Download TMY data (PVGIS)
  python download_weather.py --lat 41.88 --lon -87.63 --tmy

  # Download and auto-update IDF run period
  python download_weather.py --lat 33.75 --lon -84.39 --start 2024-06-01 --end 2024-09-30 --update-idf model.idf

  # Sync existing EPW to IDF
  python download_weather.py --sync-epw weather.epw --update-idf model.idf

Common Locations:
  Atlanta, GA:     --lat 33.75 --lon -84.39
  Chicago, IL:     --lat 41.88 --lon -87.63
  New York, NY:    --lat 40.71 --lon -74.01
  Los Angeles, CA: --lat 34.05 --lon -118.24
  Miami, FL:       --lat 25.76 --lon -80.19
        """
    )
    
    parser.add_argument('--lat', type=float, help='Latitude (decimal degrees)')
    parser.add_argument('--lon', type=float, help='Longitude (decimal degrees)')
    parser.add_argument('--output', '-o', default='weather', help='Output directory (default: weather)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD format, e.g., 2024-06-01)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD format, e.g., 2024-09-30)')
    parser.add_argument('--tmy', action='store_true', help='Download TMY data from PVGIS')
    parser.add_argument('--source', choices=['auto', 'open-meteo', 'pvgis'], default='auto',
                        help='Data source (default: auto - uses Open-Meteo for dates, PVGIS for TMY)')
    parser.add_argument('--update-idf', type=str, metavar='IDF_FILE',
                        help='Update IDF run period to match downloaded weather')
    parser.add_argument('--sync-epw', type=str, metavar='EPW_FILE',
                        help='Sync IDF run period to existing EPW file (use with --update-idf)')
    
    args = parser.parse_args()
    
    # Sync-only mode (no download)
    if args.sync_epw and args.update_idf:
        print("=" * 70)
        print("SYNCING EPW TO IDF RUN PERIOD")
        print("=" * 70 + "\n")
        
        start_month, start_day, end_month, end_day, year = get_epw_date_range(args.sync_epw)
        if start_month:
            update_idf_run_period(args.update_idf, start_month, start_day, end_month, end_day, year)
            print("\n✓ IDF run period updated to match EPW file")
        else:
            print(f"✗ Could not parse date range from {args.sync_epw}")
        return 0
    
    # Validate lat/lon for download
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon are required for downloading weather data")
    
    print("=" * 70)
    print("WEATHER DATA DOWNLOADER")
    print("=" * 70)
    print(f"Location: {args.lat}, {args.lon}")
    print(f"Output: {args.output}")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Download based on mode
    if args.tmy:
        # TMY mode - use PVGIS
        print("Mode: TMY (Typical Meteorological Year) from PVGIS\n")
        results = download_pvgis_tmy(args.lat, args.lon, args.output)
        
    elif args.start and args.end:
        # Date range mode - use Open-Meteo
        start_dt = parse_date(args.start)
        end_dt = parse_date(args.end)
        
        if not start_dt or not end_dt:
            parser.error("Dates must be in YYYY-MM-DD format (e.g., 2024-06-01)")
        
        print(f"Mode: Historical data from Open-Meteo")
        print(f"Period: {args.start} to {args.end}\n")
        
        results = download_open_meteo(args.lat, args.lon, args.start, args.end, args.output)
        
    else:
        parser.error("Specify either --tmy for TMY data, or --start and --end for date range")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    
    if results:
        print("\nDownloaded files:")
        for key, path in results.items():
            if key not in ['start_date', 'end_date']:
                print(f"  - {path}")
    else:
        print("\nNo files were downloaded.")
        return 1
    
    # Update IDF run period if requested
    if args.update_idf and results.get('epw'):
        print("\n" + "=" * 70)
        print("UPDATING IDF RUN PERIOD")
        print("=" * 70 + "\n")
        
        epw_file = results.get('epw')
        start_month, start_day, end_month, end_day, year = get_epw_date_range(epw_file)
        
        if start_month:
            update_idf_run_period(args.update_idf, start_month, start_day, end_month, end_day, year)
        else:
            print(f"  ✗ Could not parse date range from {epw_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
