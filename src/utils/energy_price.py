"""
Real-time electricity price for reward calculation.

Provides a single function get_realtime_price(month, day, hour, reward_config)
used by the RL reward to compute $/kWh. Config-driven: constant or time-of-use (TOU).
"""

from typing import Dict, Any, List


def get_realtime_price(month: int, day: int, hour: int, reward_config: Dict[str, Any]) -> float:
    """
    Return electricity price ($/kWh) for the given simulation time.

    Used by the reward function for energy cost. Reads from reward_config:

    - If realtime_price.enabled is false (or missing): returns energy_price_per_kwh.
    - If realtime_price.type == "constant": returns realtime_price.constant_price.
    - If realtime_price.type == "tou": returns TOU price by hour from realtime_price.tou
      (peak / off_peak / mid; mid is default for hours not in peak or off_peak).

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

    # Fallback
    return float(reward_config.get('energy_price_per_kwh', 0.1))
