"""
Download and parse hourly weather CSV from Open-Meteo (no API key required).

Usage
-----
    from src.utils.weather_csv import download_weather_csv, WeatherLookup

    # Download once (free, no key needed)
    download_weather_csv(
        lat=33.75, lon=-84.39,
        start="2024-06-01", end="2024-09-30",
        output_path="data/weather_atlanta_2024.csv"
    )

    # Load and query in simulation
    wx = WeatherLookup("data/weather_atlanta_2024.csv")
    controller.env.weather_override = wx.get(month=6, day=15, hour=14)
    # -> {'dry_bulb': 33.2, 'humidity': 55.0, 'wind_speed': 3.1,
    #     'beam_solar': 820.0, 'diffuse_solar': 110.0}

Open-Meteo historical endpoint: https://archive-api.open-meteo.com/v1/archive
"""

import csv
import urllib.request
import urllib.parse
from datetime import datetime


# Open-Meteo variable names -> weather_override keys and units
_VARIABLES = {
    'temperature_2m':            ('dry_bulb',      '°C'),
    'relative_humidity_2m':      ('humidity',      '%'),
    'wind_speed_10m':            ('wind_speed',    'm/s'),
    'direct_normal_irradiance':  ('beam_solar',    'W/m²'),
    'diffuse_radiation':         ('diffuse_solar', 'W/m²'),
}


def download_weather_csv(lat: float, lon: float, start: str, end: str, output_path: str) -> str:
    """Download hourly weather data from Open-Meteo and save as CSV.

    Parameters
    ----------
    lat, lon    : location in decimal degrees
    start, end  : date range as "YYYY-MM-DD" strings (inclusive)
    output_path : where to write the CSV

    Returns the output_path on success.

    Data source: Open-Meteo Historical Weather API (free, no key required).
    Resolution: hourly. Typical latency: seconds.
    """
    params = {
        'latitude':  lat,
        'longitude': lon,
        'start_date': start,
        'end_date':   end,
        'hourly':     ','.join(_VARIABLES.keys()),
        'wind_speed_unit': 'ms',   # m/s
        'format':    'csv',
    }
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(params)
    print(f"Downloading weather: {lat:.3f},{lon:.3f}  {start} → {end}")
    print(f"  URL: {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode('utf-8')
    with open(output_path, 'w', newline='') as f:
        f.write(raw)
    lines = raw.strip().splitlines()
    print(f"  Saved {len(lines) - 1} rows to {output_path}")
    return output_path


class WeatherLookup:
    """Load a downloaded Open-Meteo CSV and look up weather by (month, day, hour).

    The CSV must have been downloaded with download_weather_csv().
    Returns a dict compatible with env.weather_override.
    """

    def __init__(self, csv_path: str):
        self._data: dict[tuple[int, int, int], dict] = {}
        self._load(csv_path)

    def _load(self, csv_path: str):
        with open(csv_path, newline='') as f:
            # Open-Meteo CSV has 3 header rows; skip until line starting with "time"
            lines = f.readlines()

        # Find the header row
        header_idx = next(i for i, l in enumerate(lines) if l.startswith('time'))
        reader = csv.DictReader(lines[header_idx:])

        for row in reader:
            if not row.get('time'):
                continue
            try:
                dt = datetime.fromisoformat(row['time'])   # "2024-06-01T00:00"
            except ValueError:
                continue
            key = (dt.month, dt.day, dt.hour)
            entry = {}
            for om_var, (override_key, _unit) in _VARIABLES.items():
                val = row.get(om_var, '')
                if val not in ('', 'nan', None):
                    try:
                        entry[override_key] = float(val)
                    except ValueError:
                        pass
            self._data[key] = entry

        print(f"WeatherLookup: loaded {len(self._data)} hourly rows from {csv_path}")

    def get(self, month: int, day: int, hour: int) -> dict:
        """Return weather_override dict for the given hour, or empty dict if not found."""
        return dict(self._data.get((month, day, hour), {}))
