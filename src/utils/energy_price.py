"""
Real-time electricity price for reward calculation.

Provides a single function get_realtime_price(month, day, hour, reward_config)
used by the RL reward to compute $/kWh. Config-driven: constant, time-of-use
(TOU), or precomputed CSV price schedule.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple


_PRICE_CSV_CACHE: Dict[str, Dict[Tuple[int, int, int], float]] = {}


def _load_price_csv(csv_path: str) -> Dict[Tuple[int, int, int], float]:
    """Load hourly price CSV keyed by (month, day, hour)."""
    resolved = str(Path(csv_path).expanduser().resolve())
    if resolved in _PRICE_CSV_CACHE:
        return _PRICE_CSV_CACHE[resolved]

    prices: Dict[Tuple[int, int, int], float] = {}
    with open(resolved, newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Price CSV has no header: {csv_path}")

        fields = {name.lower(): name for name in reader.fieldnames}
        timestamp_col = fields.get('timestamp') or fields.get('time') or fields.get('datetime')
        price_col = fields.get('price') or fields.get('price_per_kwh') or fields.get('energy_price_used')
        if not timestamp_col or not price_col:
            raise ValueError(
                "Price CSV must include timestamp/time/datetime and price/price_per_kwh columns"
            )

        for row in reader:
            timestamp = row.get(timestamp_col)
            price = row.get(price_col)
            if not timestamp or price in ('', None):
                continue
            dt = datetime.fromisoformat(timestamp)
            prices[(dt.month, dt.day, dt.hour)] = float(price)

    if not prices:
        raise ValueError(f"Price CSV has no usable price rows: {csv_path}")

    _PRICE_CSV_CACHE[resolved] = prices
    return prices


def get_realtime_price(month: int, day: int, hour: int, reward_config: Dict[str, Any]) -> float:
    """
    Return electricity price ($/kWh) for the given simulation time.

    Used by the reward function for energy cost. Reads from reward_config:

    - If realtime_price.enabled is false (or missing): returns energy_price_per_kwh.
    - If realtime_price.type == "constant": returns realtime_price.constant_price.
    - If realtime_price.type == "tou": returns TOU price by hour from realtime_price.tou
      (peak / off_peak / mid; mid is default for hours not in peak or off_peak).
    - If realtime_price.type == "csv": returns price from realtime_price.csv_path,
      matched by month/day/hour.

    Args:
        month: Month (1-12).
        day: Day of month (1-31).
        hour: Hour of day (0-23).
        reward_config: The reward section from HVAC config (e.g. hvac_config.config['reward']).

    Returns:
        Price in $/kWh.
    """
    rtp = reward_config.get('realtime_price') or {}
    if not rtp.get('enabled', False):
        return float(reward_config.get('energy_price_per_kwh', 0.1))

    kind = rtp.get('type', 'constant')
    if kind == 'constant':
        return float(rtp.get('constant_price', reward_config.get('energy_price_per_kwh', 0.1)))

    if kind == 'tou':
        tou = rtp.get('tou') or {}
        peak = tou.get('peak') or {}
        off_peak = tou.get('off_peak') or {}
        mid = tou.get('mid') or {}
        peak_hours: List[int] = peak.get('hours', [])
        off_peak_hours: List[int] = off_peak.get('hours', [])
        if hour in peak_hours:
            return float(peak.get('price', 0.15))
        if hour in off_peak_hours:
            return float(off_peak.get('price', 0.08))
        return float(mid.get('price', 0.10))

    if kind == 'csv':
        csv_path = rtp.get('csv_path')
        if not csv_path:
            raise ValueError("reward.realtime_price.csv_path is required when type is 'csv'")
        prices = _load_price_csv(csv_path)
        key = (int(month), int(day), int(hour))
        if key in prices:
            return prices[key]
        fallback = rtp.get('fallback_price', reward_config.get('energy_price_per_kwh', 0.1))
        print(
            "[PRICE] Fallback price used "
            f"for month={key[0]}, day={key[1]}, hour={key[2]}: "
            f"${float(fallback):.4f}/kWh"
        )
        return float(fallback)

    # Fallback
    return float(reward_config.get('energy_price_per_kwh', 0.1))
