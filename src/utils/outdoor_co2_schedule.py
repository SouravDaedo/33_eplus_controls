"""
Outdoor CO2 (ppm) from CSV for use at each simulation timestep.

CSV format (one of):
  - month, day, hour, ppm   (hour 0-23; one row per hour or coarser)
  - datetime, ppm           (datetime parseable, e.g. "2023-04-15 10:00:00")
  - Month, Day, Hour, ppm   (same, case-insensitive column names)

Lookup by (month, day, hour) returns the matching ppm; if multiple rows match (e.g. sub-hourly),
the first is used. If no row matches, returns fallback_ppm.
"""

from pathlib import Path
from datetime import timedelta
from typing import Optional, Union
import pandas as pd


def load_outdoor_co2_csv(
    csv_path: Union[str, Path],
    fallback_ppm: float = 400.0,
) -> "OutdoorCO2Lookup":
    """
    Load a CSV of outdoor CO2 (ppm) and return a lookup callable.

    Args:
        csv_path: Path to CSV with columns (month, day, hour, ppm) or (datetime, ppm).
        fallback_ppm: Value to return when no row matches the requested (month, day, hour).

    Returns:
        OutdoorCO2Lookup instance with get_ppm(month, day, hour) -> float.
    """
    path = Path(csv_path)
    if not path.is_absolute():
        # Assume relative to project root (parent of src/)
        path = Path(__file__).resolve().parent.parent.parent / path
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return OutdoorCO2Lookup(df, fallback_ppm=fallback_ppm)


class OutdoorCO2Lookup:
    """Look up outdoor CO2 ppm by (month, day, hour)."""

    def __init__(self, df: pd.DataFrame, fallback_ppm: float = 400.0):
        self._fallback = fallback_ppm
        # Normalize columns
        cols = [c.lower() for c in df.columns]
        if "datetime" in cols or "date" in cols:
            dt_col = "datetime" if "datetime" in cols else "date"
            df = df.copy()
            df["_dt"] = pd.to_datetime(df[dt_col])
            df["_month"] = df["_dt"].dt.month
            df["_day"] = df["_dt"].dt.day
            df["_hour"] = df["_dt"].dt.hour
            ppm_col = "ppm" if "ppm" in cols else [c for c in cols if "ppm" in c or "co2" in c][:1]
            if isinstance(ppm_col, list):
                ppm_col = ppm_col[0] if ppm_col else None
            if ppm_col is None:
                raise ValueError("CSV must have a 'ppm' or CO2 column")
            self._df = df[["_month", "_day", "_hour", ppm_col]].rename(columns={ppm_col: "_ppm"})
        else:
            month_col = next((c for c in cols if "month" in c), None)
            day_col = next((c for c in cols if "day" in c and "hour" not in c), None)
            hour_col = next((c for c in cols if "hour" in c), None)
            ppm_col = next((c for c in cols if c == "ppm" or "ppm" in c or "co2" in c), None)
            if not all([month_col, day_col, hour_col, ppm_col]):
                raise ValueError(
                    "CSV must have columns (month, day, hour, ppm) or (datetime, ppm). "
                    f"Found: {list(df.columns)}"
                )
            self._df = df[[month_col, day_col, hour_col, ppm_col]].rename(
                columns={month_col: "_month", day_col: "_day", hour_col: "_hour", ppm_col: "_ppm"}
            )
        self._df["_month"] = self._df["_month"].astype(int)
        self._df["_day"] = self._df["_day"].astype(int)
        self._df["_hour"] = self._df["_hour"].astype(int)

    def get_ppm(self, month: int, day: int, hour: int) -> float:
        """Return outdoor CO2 (ppm) for the given simulation time. Hour 0-23."""
        if hour >= 24:
            hour = 23
        match = self._df[
            (self._df["_month"] == month)
            & (self._df["_day"] == day)
            & (self._df["_hour"] == hour)
        ]
        if match.empty:
            return self._fallback
        return float(match["_ppm"].iloc[0])

    def to_ep_schedule_file(self, output_path: Union[str, Path], year: int = 2023) -> None:
        """
        Write an EnergyPlus Schedule:File CSV (8760 rows, one ppm value per hour).
        Hour 1 = Jan 1 00:00, hour 8760 = Dec 31 23:00.
        """
        from datetime import datetime
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = []
        dt = datetime(year, 1, 1, 0, 0, 0)
        for _ in range(8760):
            ppm = self.get_ppm(dt.month, dt.day, dt.hour)
            values.append(f"{ppm}")
            dt = dt + timedelta(hours=1)
        path.write_text("\n".join(values), encoding="utf-8")


def write_ep_outdoor_co2_schedule(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    fallback_ppm: float = 400.0,
    year: int = 2023,
) -> None:
    """
    Load user CSV (month/day/hour/ppm or datetime/ppm) and write an 8760-row
    file for EnergyPlus Schedule:File (one ppm per hour).
    """
    lookup = load_outdoor_co2_csv(csv_path, fallback_ppm=fallback_ppm)
    lookup.to_ep_schedule_file(output_path, year=year)
