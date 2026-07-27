"""
Generate a synthetic outdoor CO2 (ppm) CSV for use with simulation.outdoor_co2_csv_path.

Output format: month, day, hour, ppm  (one row per hour, hour 0-23)

By default a full calendar year is generated. Use --start-month/--start-day and
--end-month/--end-day to limit output to an inclusive date range.

CO2 model:
  - Baseline: global background (~420 ppm, adjustable)
  - Seasonal (Keeling curve): sinusoidal, peaks in late winter/spring, troughs in late summer
  - Diurnal: small swing driven by photosynthesis, lower during daylight hours
  - Urban increment: constant offset for urban sites
  - Recurring events: rush-hour traffic spikes (weekday mornings/evenings)
  - One-time events: wildfire smoke, industrial episodes, pollution alerts
    (built-in defaults, or custom via --event / --events-file).
    Events are specified by month/day/hour only; --year sets the calendar year.
  - Optional Gaussian noise
"""

import argparse
import json
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
    ("morning_rush", True,  7,  9, 8,  85.0),
    ("evening_rush", True, 16, 19, 17, 70.0),
]

DEFAULT_ONETIME_EVENTS = [
    # (label, start_month, start_day, start_hour, duration_hours, peak_ppm_increment)
    # Winter stagnant-air pool — mid-January
    ("winter_inversion",    1, 18,  0, 48,  200.0),
    # Factory/plant upset — mid-March afternoon
    ("industrial_release",  3, 10, 14,  8,  140.0),
    # Heat inversion smog — late April
    ("smog_inversion",      4, 22,  0, 36,  220.0),
    # Regional air-quality alert — early May
    ("smoke_alert",         5,  8,  6, 18,  100.0),
    # Wildfire smoke episode — early June
    ("wildfire_smoke",      6,  5,  6, 72,  350.0),
    # Holiday fireworks / evening combustion — July 4
    ("holiday_fireworks",   7,  4, 22,  4,   60.0),
    # Short industrial/traffic pollution spike — mid-July
    ("pollution_spike",     7, 14, 10, 12,  170.0),
    # Late-summer wildfire — mid-August
    ("wildfire_smoke_2",    8, 18,  8, 60,  300.0),
    # Agricultural field burning — mid-October
    ("harvest_burn",       10, 15,  7, 12,  125.0),
]

# Localized highway / corridor backups — 20 one-time jams spread across the year.
# Morning peaks ~07-09, evening peaks ~16-18; duration 4-7 h; +85 to +125 ppm.
TRAFFIC_JAM_EVENTS = [
    ("traffic_jam_01",  1,  9,  8, 5,  90.0),
    ("traffic_jam_02",  1, 22, 17, 6, 105.0),
    ("traffic_jam_03",  2,  6,  8, 5,  85.0),
    ("traffic_jam_04",  2, 19, 17, 6, 100.0),
    ("traffic_jam_05",  3,  4,  8, 6,  95.0),
    ("traffic_jam_06",  3, 27, 16, 5, 110.0),
    ("traffic_jam_07",  4,  3,  7, 5,  92.0),
    ("traffic_jam_08",  4, 16, 17, 6, 105.0),
    ("traffic_jam_09",  5,  1,  8, 5,  98.0),
    ("traffic_jam_10",  5, 20, 17, 7, 110.0),
    ("traffic_jam_11",  6, 11,  8, 6, 100.0),
    ("traffic_jam_12",  6, 24, 16, 5, 120.0),
    ("traffic_jam_13",  7,  8,  7, 5,  95.0),
    ("traffic_jam_14",  7, 25, 17, 6, 105.0),
    ("traffic_jam_15",  8,  5,  8, 5,  92.0),
    ("traffic_jam_16",  8, 28, 16, 7, 115.0),
    ("traffic_jam_17",  9, 12,  8, 6,  98.0),
    ("traffic_jam_18",  9, 26, 17, 5, 105.0),
    ("traffic_jam_19", 10,  7,  8, 6,  95.0),
    ("traffic_jam_20", 12, 18, 16, 7, 125.0),
]

DEFAULT_ONETIME_EVENTS = DEFAULT_ONETIME_EVENTS + TRAFFIC_JAM_EVENTS

# --event format: label,month,day,hour,duration_hours,peak_ppm
# Month/day/hour only — no year field. The calendar year comes from --year.
EVENT_ARG_HELP = (
    "One-time pollution event. Repeat for multiple events. "
    "Format: label,month,day,hour,duration_hours,peak_ppm (no year — use --year). "
    "Example: --event wildfire,6,5,6,72,150"
)


def _parse_event_tuple(event) -> tuple[str, int, int, int, float, float]:
    if isinstance(event, (list, tuple)):
        label, month, day, hour, duration, peak = event
    else:
        label = event["label"]
        month = event["month"]
        day = event["day"]
        hour = event["hour"]
        duration = event["duration_hours"]
        peak = event["peak_increment"]
    return (
        str(label),
        int(month),
        int(day),
        int(hour),
        float(duration),
        float(peak),
    )


def _parse_event_arg(value: str) -> tuple[str, int, int, int, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Invalid --event value '{value}'. Expected: label,month,day,hour,duration_hours,peak_ppm"
        )
    label, month, day, hour, duration, peak = parts
    return (label, int(month), int(day), int(hour), float(duration), float(peak))


def _load_events_file(path: str | Path) -> list[tuple[str, int, int, int, float, float]]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Events file must contain a JSON list: {file_path}")
    return [_parse_event_tuple(item) for item in data]


def _resolve_onetime_events(
    no_events: bool,
    no_default_events: bool,
    custom_events: list[tuple[str, int, int, int, float, float]] | None,
) -> list[tuple[str, int, int, int, float, float]]:
    if no_events:
        return []
    events: list[tuple[str, int, int, int, float, float]] = []
    if not no_default_events:
        events.extend(_parse_event_tuple(ev) for ev in DEFAULT_ONETIME_EVENTS)
    if custom_events:
        events.extend(custom_events)
    return events


def _gaussian_pulse(hour_offset: float, duration_hours: float) -> float:
    """Smooth bell-shaped pulse: 1.0 at centre, ~0.05 at ±duration edges."""
    sigma = duration_hours / 4.0
    return math.exp(-0.5 * (hour_offset / sigma) ** 2)


def _resolve_period(
    year: int,
    start_month: int | None = None,
    start_day: int | None = None,
    end_month: int | None = None,
    end_day: int | None = None,
) -> tuple[datetime, datetime]:
    """
    Return inclusive (start_dt, end_dt) for hourly generation.

    When all range args are None, returns Jan 1 00:00 through Dec 31 23:00.
    When any range arg is set, all four must be provided.
    """
    range_args = (start_month, start_day, end_month, end_day)
    if all(arg is None for arg in range_args):
        return datetime(year, 1, 1, 0, 0), datetime(year, 12, 31, 23, 0)

    if any(arg is None for arg in range_args):
        raise ValueError(
            "Date range requires all of start_month, start_day, end_month, and end_day. "
            "Omit all four to generate a full calendar year."
        )

    try:
        start_dt = datetime(year, start_month, start_day, 0, 0)
        end_dt = datetime(year, end_month, end_day, 23, 0)
    except ValueError as exc:
        raise ValueError(
            f"Invalid date range for year {year}: "
            f"{start_month}/{start_day} to {end_month}/{end_day}"
        ) from exc

    if start_dt > end_dt:
        raise ValueError(
            f"Start date {start_month}/{start_day} must be on or before "
            f"end date {end_month}/{end_day} in year {year}"
        )
    return start_dt, end_dt


def _abs_hour_of_year(year: int, month: int, day: int, hour: int) -> int | None:
    try:
        return int(
            (datetime(year, month, day, hour) - datetime(year, 1, 1, 0, 0)).total_seconds()
            // 3600
        )
    except ValueError:
        return None


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
    start_month: int | None = None,
    start_day: int | None = None,
    end_month: int | None = None,
    end_day: int | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame [month, day, hour, ppm] for each hour in the period.

    By default the period is the full calendar year. Pass start/end month/day
    to generate only that inclusive date range (hours 00-23 on each day).

    onetime_events dicts / --event tuples:
        label           str   — name (for printing)
        month           int   — start month (1-12); no year field
        day             int   — start day
        hour            int   — start hour (0-23)
        duration_hours  float — total event duration
        peak_increment  float — ppm added at the peak of the event

    The calendar year is set by the ``year`` argument / ``--year`` CLI flag.
    Events recur on that calendar date each time you generate a CSV for a year.
    """
    import random
    rng = random.Random(seed)

    if onetime_events is None:
        onetime_events = DEFAULT_ONETIME_EVENTS

    # Pre-build a lookup: absolute_hour -> onetime_increment
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    total_hours = 8784 if is_leap else 8760

    onetime_map = [0.0] * total_hours
    start_dt, end_dt = _resolve_period(year, start_month, start_day, end_month, end_day)

    for ev in onetime_events:
        _, m, d, h, dur, peak = _parse_event_tuple(ev)

        start_abs = _abs_hour_of_year(year, m, d, h)
        if start_abs is None:
            continue
        centre_abs = start_abs + dur / 2.0
        for offset in range(int(dur * 2) + 1):
            abs_h = start_abs + offset
            if abs_h < 0 or abs_h >= total_hours:
                continue
            t_offset = abs_h - centre_abs
            increment = peak * _gaussian_pulse(t_offset, dur)
            onetime_map[abs_h] += increment

    # Generate hourly rows
    rows = []
    dt = start_dt
    year_start = datetime(year, 1, 1, 0, 0)

    while dt <= end_dt:
        abs_h = int((dt - year_start).total_seconds() // 3600)
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


def _event_overlaps_period(
    year: int,
    event,
    start_dt: datetime,
    end_dt: datetime,
) -> bool:
    _, m, d, h, dur, _ = _parse_event_tuple(event)

    event_start = _abs_hour_of_year(year, m, d, h)
    if event_start is None:
        return False
    event_end = event_start + int(dur)
    period_start = int((start_dt - datetime(year, 1, 1, 0, 0)).total_seconds() // 3600)
    period_end = int((end_dt - datetime(year, 1, 1, 0, 0)).total_seconds() // 3600)
    return event_start <= period_end and event_end >= period_start


def main():
    parser = argparse.ArgumentParser(description="Generate outdoor CO2 CSV for EnergyPlus RL simulation.")
    parser.add_argument("--year",                type=int,   default=2023,  help="Year (default: 2023)")
    parser.add_argument("--start-month",         type=int,   default=None,  help="Range start month (1-12); requires --start-day, --end-month, --end-day")
    parser.add_argument("--start-day",           type=int,   default=None,  help="Range start day (1-31)")
    parser.add_argument("--end-month",           type=int,   default=None,  help="Range end month (1-12)")
    parser.add_argument("--end-day",             type=int,   default=None,  help="Range end day (1-31); inclusive through 23:00")
    parser.add_argument("--baseline",            type=float, default=420.0, help="Baseline CO2 ppm (default: 420)")
    parser.add_argument("--seasonal-amplitude",  type=float, default=8.0,   help="Seasonal swing ±ppm (default: 8)")
    parser.add_argument("--seasonal-peak-doy",   type=int,   default=100,   help="Day-of-year seasonal peak (default: 100)")
    parser.add_argument("--diurnal-amplitude",   type=float, default=3.0,   help="Diurnal swing ±ppm (default: 3)")
    parser.add_argument("--diurnal-trough-hour", type=int,   default=14,    help="Hour of daily CO2 minimum (default: 14)")
    parser.add_argument("--urban-increment",     type=float, default=0.0,   help="Constant urban offset ppm (default: 0)")
    parser.add_argument("--noise-std",           type=float, default=0.5,   help="Gaussian noise std ppm (default: 0.5)")
    parser.add_argument("--seed",                type=int,   default=42,    help="Random seed (default: 42)")
    parser.add_argument("--no-rush-hour",        action="store_true",       help="Disable weekday rush-hour spikes")
    parser.add_argument("--no-events",           action="store_true",       help="Disable all one-time pollution events")
    parser.add_argument(
        "--no-default-events",
        action="store_true",
        help="Skip built-in events (wildfire, smog, pollution spike); use with --event or --events-file",
    )
    parser.add_argument("--event",               action="append", default=[], metavar="SPEC", help=EVENT_ARG_HELP)
    parser.add_argument(
        "--events-file",
        type=str,
        default=None,
        help="JSON file with a list of events (month/day/hour only; year from --year): "
             '[{"label":"wildfire","month":6,"day":5,"hour":6,"duration_hours":72,"peak_increment":150}]',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/outdoor_co2_{year}.csv or data/outdoor_co2_{year}_{start}_to_{end}.csv)",
    )
    args = parser.parse_args()

    start_dt, end_dt = _resolve_period(
        args.year,
        args.start_month,
        args.start_day,
        args.end_month,
        args.end_day,
    )
    has_range = args.start_month is not None

    if args.output is None:
        if has_range:
            output_name = (
                f"data/outdoor_co2_{args.year}_"
                f"{args.start_month:02d}{args.start_day:02d}_to_"
                f"{args.end_month:02d}{args.end_day:02d}.csv"
            )
        else:
            output_name = f"data/outdoor_co2_{args.year}.csv"
    else:
        output_name = args.output.format(
            year=args.year,
            start_month=args.start_month or 1,
            start_day=args.start_day or 1,
            end_month=args.end_month or 12,
            end_day=args.end_day or 31,
        )

    output_path = Path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    custom_events: list[tuple[str, int, int, int, float, float]] = []
    for event_arg in args.event:
        custom_events.append(_parse_event_arg(event_arg))
    if args.events_file:
        custom_events.extend(_load_events_file(args.events_file))

    onetime = _resolve_onetime_events(
        no_events=args.no_events,
        no_default_events=args.no_default_events,
        custom_events=custom_events or None,
    )

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
        start_month=args.start_month,
        start_day=args.start_day,
        end_month=args.end_month,
        end_day=args.end_day,
    )

    df.to_csv(output_path, index=False)

    period_label = (
        f"{start_dt.month:02d}/{start_dt.day:02d} to {end_dt.month:02d}/{end_dt.day:02d}, {args.year}"
        if has_range
        else f"full year {args.year}"
    )
    print(f"Generated {len(df)} hourly rows -> {output_path}")
    print(f"  Period:    {period_label}")
    print(f"  ppm range: {df['ppm'].min():.1f} – {df['ppm'].max():.1f}")
    print(f"  ppm mean:  {df['ppm'].mean():.1f}")
    print()
    print("Events included:")
    if not args.no_rush_hour:
        print("  Rush hour  — weekday morning 07-09 (+85 ppm peak), evening 16-19 (+70 ppm peak)")
    events_in_period = [
        ev for ev in onetime
        if _event_overlaps_period(args.year, ev, start_dt, end_dt)
    ]
    for ev in events_in_period:
        label, m, d, h, dur, peak = _parse_event_tuple(ev)
        print(f"  {label:<20} {m:02d}/{d:02d} {h:02d}:00  duration={dur}h  peak=+{peak:.0f} ppm")
    if onetime and not events_in_period:
        print("  (none overlap the selected date range)")
    print()
    print("To use in simulation, set in config/hvac_config.yaml:")
    print(f"  simulation:")
    print(f"    outdoor_co2_csv_path: \"{output_path}\"")
    print(f"    outdoor_co2_fallback_ppm: {int(args.baseline)}")


if __name__ == "__main__":
    main()
