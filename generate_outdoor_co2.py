"""
Generate a synthetic outdoor CO2 (ppm) CSV for use with simulation.outdoor_co2_csv_path.

Output format: month, day, hour, ppm  (one row per hour, hour 0-23)

CO2 model:
  - Baseline: global background (~420 ppm, adjustable)
  - Seasonal (Keeling curve): sinusoidal, peaks in late winter/spring, troughs in late summer
  - Diurnal: small swing driven by photosynthesis, lower during daylight hours
  - Urban increment: constant offset for urban sites
  - Recurring events: rush-hour traffic spikes (weekday mornings/evenings)
  - One-time events: wildfire smoke, industrial episodes, pollution alerts
  - Optional Gaussian noise
"""

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def day_of_year(dt: datetime) -> int:
    return dt.timetuple().tm_yday


# ---------------------------------------------------------------------------
# Built-in event presets
# ---------------------------------------------------------------------------

RUSH_HOUR_EVENTS = [
    # (label, weekdays_only, start_hour, end_hour, peak_hour, amplitude_ppm)
    ("morning_rush", True,  7,  9, 8,  30.0),
    ("evening_rush", True, 16, 19, 17, 25.0),
]

DEFAULT_ONETIME_EVENTS = [
    # (label, start_month, start_day, start_hour, duration_hours, peak_ppm_increment)
    # Wildfire smoke episode — early June
    ("wildfire_smoke",  6,  5,  6, 72,  150.0),
    # Short industrial/traffic pollution spike — mid-July
    ("pollution_spike", 7, 14, 10, 12,   60.0),
    # Heat inversion smog — late April
    ("smog_inversion",  4, 22,  0, 36,   80.0),
]


def _gaussian_pulse(hour_offset: float, duration_hours: float) -> float:
    """Smooth bell-shaped pulse: 1.0 at centre, ~0.05 at ±duration edges."""
    sigma = duration_hours / 4.0
    return math.exp(-0.5 * (hour_offset / sigma) ** 2)


def generate_outdoor_co2(
    year: int = 2023,
    baseline_ppm: float = 420.0,
    seasonal_amplitude: float = 8.0,
    seasonal_peak_doy: int = 100,
    diurnal_amplitude: float = 3.0,
    diurnal_trough_hour: int = 14,
    urban_increment: float = 0.0,
    noise_std: float = 0.5,
    seed: int = 42,
    rush_hour: bool = True,
    onetime_events: list | None = None,   # list of dicts with keys below
) -> pd.DataFrame:
    """
    Return a DataFrame [month, day, hour, ppm] for every hour of the year.

    onetime_events dicts:
        label           str   — name (for printing)
        month           int   — start month
        day             int   — start day
        hour            int   — start hour (0-23)
        duration_hours  float — total event duration
        peak_increment  float — ppm added at the peak of the event
    """
    import random
    rng = random.Random(seed)

    if onetime_events is None:
        onetime_events = DEFAULT_ONETIME_EVENTS

    # Pre-build a lookup: absolute_hour -> onetime_increment
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    total_hours = 8784 if is_leap else 8760

    onetime_map = [0.0] * total_hours

    def _abs_hour(month, day, hour):
        try:
            return int((datetime(year, month, day, hour) - datetime(year, 1, 1)).total_seconds() // 3600)
        except ValueError:
            return None

    for ev in onetime_events:
        if isinstance(ev, (list, tuple)):
            _, m, d, h, dur, peak = ev
        else:
            m, d, h = ev["month"], ev["day"], ev["hour"]
            dur  = ev["duration_hours"]
            peak = ev["peak_increment"]

        centre_abs = _abs_hour(m, d, h) + dur / 2.0
        for offset in range(int(dur * 2) + 1):
            abs_h = _abs_hour(m, d, h) + offset
            if abs_h is None or abs_h >= total_hours:
                continue
            t_offset = abs_h - centre_abs
            increment = peak * _gaussian_pulse(t_offset, dur)
            onetime_map[abs_h] += increment

    # Generate hourly rows
    rows = []
    dt = datetime(year, 1, 1, 0, 0)

    for abs_h in range(total_hours):
        doy  = day_of_year(dt)
        hour = dt.hour
        weekday = dt.weekday()  # 0=Mon … 6=Sun

        # Seasonal
        seasonal = seasonal_amplitude * math.cos(
            2 * math.pi * (doy - seasonal_peak_doy) / 365.0
        )

        # Diurnal
        diurnal = diurnal_amplitude * math.cos(
            2 * math.pi * (hour - diurnal_trough_hour) / 24.0
        )

        # Rush-hour traffic (weekdays only, smooth bell per rush window)
        rush = 0.0
        if rush_hour:
            for _, weekdays_only, h_start, h_end, h_peak, amp in RUSH_HOUR_EVENTS:
                if weekdays_only and weekday >= 5:
                    continue
                if h_start <= hour <= h_end:
                    dur_h = h_end - h_start
                    rush += amp * _gaussian_pulse(hour - h_peak, dur_h)

        # One-time events
        event_inc = onetime_map[abs_h]

        # Noise
        noise = rng.gauss(0, noise_std) if noise_std > 0 else 0.0

        ppm = round(baseline_ppm + seasonal + diurnal + urban_increment + rush + event_inc + noise, 2)

        rows.append({"month": dt.month, "day": dt.day, "hour": dt.hour, "ppm": ppm})
        dt += timedelta(hours=1)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate outdoor CO2 CSV for EnergyPlus RL simulation.")
    parser.add_argument("--year",                type=int,   default=2023,  help="Year (default: 2023)")
    parser.add_argument("--baseline",            type=float, default=420.0, help="Baseline CO2 ppm (default: 420)")
    parser.add_argument("--seasonal-amplitude",  type=float, default=8.0,   help="Seasonal swing ±ppm (default: 8)")
    parser.add_argument("--seasonal-peak-doy",   type=int,   default=100,   help="Day-of-year seasonal peak (default: 100)")
    parser.add_argument("--diurnal-amplitude",   type=float, default=3.0,   help="Diurnal swing ±ppm (default: 3)")
    parser.add_argument("--diurnal-trough-hour", type=int,   default=14,    help="Hour of daily CO2 minimum (default: 14)")
    parser.add_argument("--urban-increment",     type=float, default=0.0,   help="Constant urban offset ppm (default: 0)")
    parser.add_argument("--noise-std",           type=float, default=0.5,   help="Gaussian noise std ppm (default: 0.5)")
    parser.add_argument("--seed",                type=int,   default=42,    help="Random seed (default: 42)")
    parser.add_argument("--no-rush-hour",        action="store_true",       help="Disable weekday rush-hour spikes")
    parser.add_argument("--no-events",           action="store_true",       help="Disable one-time pollution events")
    parser.add_argument("--output",              type=str,   default="data/outdoor_co2_{year}.csv",
                        help="Output path (default: data/outdoor_co2_{year}.csv)")
    args = parser.parse_args()

    output_path = Path(args.output.format(year=args.year))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onetime = [] if args.no_events else DEFAULT_ONETIME_EVENTS

    df = generate_outdoor_co2(
        year=args.year,
        baseline_ppm=args.baseline,
        seasonal_amplitude=args.seasonal_amplitude,
        seasonal_peak_doy=args.seasonal_peak_doy,
        diurnal_amplitude=args.diurnal_amplitude,
        diurnal_trough_hour=args.diurnal_trough_hour,
        urban_increment=args.urban_increment,
        noise_std=args.noise_std,
        seed=args.seed,
        rush_hour=not args.no_rush_hour,
        onetime_events=onetime,
    )

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} hourly rows -> {output_path}")
    print(f"  ppm range: {df['ppm'].min():.1f} – {df['ppm'].max():.1f}")
    print(f"  ppm mean:  {df['ppm'].mean():.1f}")
    print()
    print("Events included:")
    if not args.no_rush_hour:
        print("  Rush hour  — weekday morning 07-09 (+30 ppm peak), evening 16-19 (+25 ppm peak)")
    for ev in onetime:
        if isinstance(ev, (list, tuple)):
            label, m, d, h, dur, peak = ev
        else:
            label, m, d, h, dur, peak = ev["label"], ev["month"], ev["day"], ev["hour"], ev["duration_hours"], ev["peak_increment"]
        print(f"  {label:<20} {m:02d}/{d:02d} {h:02d}:00  duration={dur}h  peak=+{peak:.0f} ppm")
    print()
    print("To use in simulation, set in config/hvac_config.yaml:")
    print(f"  simulation:")
    print(f"    outdoor_co2_csv_path: \"{output_path}\"")
    print(f"    outdoor_co2_fallback_ppm: {int(args.baseline)}")


if __name__ == "__main__":
    main()
