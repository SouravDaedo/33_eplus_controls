"""
Real-Time Pricing (RTP) and Energy Price Model

A configurable Python-based electricity pricing model for use in building energy simulations.
Supports various pricing structures: Time-of-Use (TOU), Real-Time Pricing (RTP), 
dynamic pricing, and wholesale market prices.

Author: Generated for EnergyPlus Controls Project
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import math


class PricingType(Enum):
    """Types of electricity pricing structures."""
    FLAT = "flat"  # Constant price
    TOU = "tou"  # Time-of-Use with fixed periods
    RTP = "rtp"  # Real-Time Pricing (hourly varying)
    DYNAMIC = "dynamic"  # Dynamic pricing based on conditions
    WHOLESALE = "wholesale"  # Simulated wholesale market prices


@dataclass
class TOUPeriod:
    """Definition of a Time-of-Use period."""
    name: str  # e.g., "peak", "off-peak", "mid-peak"
    start_hour: int  # Start hour (0-23)
    end_hour: int  # End hour (0-23)
    price_per_kwh: float  # Price in $/kWh
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Weekdays by default


@dataclass
class PricingConfig:
    """Configuration for the energy pricing model."""
    
    pricing_type: PricingType = PricingType.TOU
    
    # Flat rate pricing
    flat_rate: float = 0.12  # $/kWh
    
    # Time-of-Use periods (default: typical utility TOU)
    tou_periods: List[TOUPeriod] = field(default_factory=lambda: [
        TOUPeriod("off-peak", 0, 6, 0.08, [0, 1, 2, 3, 4, 5, 6]),
        TOUPeriod("mid-peak", 6, 14, 0.12, [0, 1, 2, 3, 4]),
        TOUPeriod("peak", 14, 20, 0.25, [0, 1, 2, 3, 4]),
        TOUPeriod("mid-peak", 20, 24, 0.12, [0, 1, 2, 3, 4]),
        TOUPeriod("off-peak", 6, 24, 0.08, [5, 6]),  # Weekends
    ])
    
    # RTP/Dynamic pricing parameters
    base_price: float = 0.10  # Base price for RTP ($/kWh)
    price_volatility: float = 0.3  # Price volatility factor (0-1)
    min_price: float = 0.02  # Minimum RTP price ($/kWh)
    max_price: float = 0.60  # Maximum RTP price ($/kWh)
    weather_sensitivity: float = 0.35  # RTP sensitivity to weather-driven grid demand
    solar_sensitivity: float = 0.15  # RTP discount when solar generation is abundant
    
    # Wholesale market simulation parameters
    wholesale_base: float = 0.05  # Base wholesale price ($/kWh)
    wholesale_peak_multiplier: float = 5.0  # Peak price multiplier
    
    # Demand charges (optional)
    demand_charge_per_kw: float = 0.0  # $/kW for peak demand
    
    # Feed-in tariff for solar export
    feed_in_tariff: float = 0.05  # $/kWh for exported energy
    
    # Net metering
    net_metering: bool = False  # If True, exports offset imports at retail rate
    
    # Carbon pricing (optional)
    carbon_price_per_kg: float = 0.0  # $/kg CO2
    grid_carbon_intensity: float = 0.4  # kg CO2/kWh


@dataclass
class PriceState:
    """Current price state at a timestep."""
    timestamp: datetime
    price_per_kwh: float  # Current electricity price ($/kWh)
    period_name: str  # Name of current pricing period
    is_peak: bool  # Whether this is a peak period
    feed_in_rate: float  # Rate for exported energy ($/kWh)
    carbon_cost_per_kwh: float  # Carbon cost component ($/kWh)


@dataclass
class EnergyCostResult:
    """Result of energy cost calculation for a timestep."""
    timestamp: datetime
    energy_consumed_kwh: float
    energy_exported_kwh: float
    price_per_kwh: float
    import_cost: float  # Cost for imported energy ($)
    export_credit: float  # Credit for exported energy ($)
    net_cost: float  # Net energy cost ($)
    carbon_cost: float  # Carbon cost ($)
    total_cost: float  # Total cost including carbon ($)


class EnergyPriceModel:
    """
    Energy Price Model for building simulations.
    
    Generates electricity prices based on various pricing structures
    and calculates energy costs for building operations.
    
    Example usage:
        config = PricingConfig(pricing_type=PricingType.TOU)
        price_model = EnergyPriceModel(config)
        
        # Get price at specific time
        price = price_model.get_price(datetime(2024, 7, 15, 14, 0))
        
        # Calculate cost for energy consumption
        cost = price_model.calculate_cost(
            timestamp=datetime(2024, 7, 15, 14, 0),
            energy_consumed_kwh=50,
            energy_exported_kwh=10
        )
    """
    
    def __init__(self, config: Optional[PricingConfig] = None):
        """
        Initialize the energy price model.
        
        Args:
            config: Pricing configuration. Uses defaults if not provided.
        """
        self.config = config or PricingConfig()
        self.price_history: List[PriceState] = []
        self.cost_history: List[EnergyCostResult] = []
        self._rtp_cache: Dict[Tuple[Any, ...], float] = {}
        
        # Seed for reproducible RTP generation
        self._rng = np.random.default_rng(42)
    
    def reset(self, seed: Optional[int] = None):
        """Reset the model state."""
        self.price_history = []
        self.cost_history = []
        self._rtp_cache = {}
        if seed is not None:
            self._rng = np.random.default_rng(seed)
    
    def get_price(self, timestamp: datetime, weather: Optional[Dict[str, float]] = None) -> PriceState:
        """
        Get electricity price for a specific timestamp.
        
        Args:
            timestamp: Datetime for price lookup
            weather: Optional weather/grid context for RTP, e.g.
                {"dry_bulb_c": 34.0, "ghi": 800.0}
            
        Returns:
            PriceState with current pricing information
        """
        if self.config.pricing_type == PricingType.FLAT:
            price, period_name, is_peak = self._get_flat_price(timestamp)
        elif self.config.pricing_type == PricingType.TOU:
            price, period_name, is_peak = self._get_tou_price(timestamp)
        elif self.config.pricing_type == PricingType.RTP:
            price, period_name, is_peak = self._get_rtp_price(timestamp, weather)
        elif self.config.pricing_type == PricingType.DYNAMIC:
            price, period_name, is_peak = self._get_dynamic_price(timestamp)
        elif self.config.pricing_type == PricingType.WHOLESALE:
            price, period_name, is_peak = self._get_wholesale_price(timestamp)
        else:
            raise ValueError(f"Unknown pricing type: {self.config.pricing_type}")
        
        # Calculate feed-in rate
        if self.config.net_metering:
            feed_in_rate = price  # Net metering: export at retail rate
        else:
            feed_in_rate = self.config.feed_in_tariff
        
        # Calculate carbon cost
        carbon_cost = (self.config.carbon_price_per_kg * 
                      self.config.grid_carbon_intensity)
        
        state = PriceState(
            timestamp=timestamp,
            price_per_kwh=price,
            period_name=period_name,
            is_peak=is_peak,
            feed_in_rate=feed_in_rate,
            carbon_cost_per_kwh=carbon_cost
        )
        
        self.price_history.append(state)
        return state
    
    def _get_flat_price(self, timestamp: datetime) -> Tuple[float, str, bool]:
        """Get flat rate price."""
        return self.config.flat_rate, "flat", False
    
    def _get_tou_price(self, timestamp: datetime) -> Tuple[float, str, bool]:
        """Get Time-of-Use price based on configured periods."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        for period in self.config.tou_periods:
            if day_of_week in period.days:
                if period.start_hour <= hour < period.end_hour:
                    is_peak = period.name.lower() == "peak"
                    return period.price_per_kwh, period.name, is_peak
        
        # Default to base price if no period matches
        return self.config.base_price, "default", False
    
    def _get_rtp_price(
        self,
        timestamp: datetime,
        weather: Optional[Dict[str, float]] = None
    ) -> Tuple[float, str, bool]:
        """
        Generate Real-Time Price using a demand-shaped model.
        
        Uses a combination of:
        - Daily grid demand pattern (morning/evening peaks, lower overnight)
        - Weekend and seasonal effects
        - Weather-driven cooling/heating demand when dry-bulb temperature is provided
        - Solar scarcity/abundance when GHI is provided
        - Small bounded volatility so prices are not fully random
        """
        # Round to hour for caching
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        weather_key = self._weather_cache_key(weather)
        cache_key = (hour_key, weather_key)
        
        if cache_key in self._rtp_cache:
            price = self._rtp_cache[cache_key]
        else:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Grid demand shape: low overnight, morning ramp, afternoon/evening peak.
            overnight = 0.65
            morning_peak = 0.30 * math.exp(-((hour - 8) / 3.0) ** 2)
            evening_peak = 0.55 * math.exp(-((hour - 17) / 4.0) ** 2)
            daily_factor = overnight + morning_peak + evening_peak
            
            # Weekend discount
            weekend_factor = 0.8 if day_of_week >= 5 else 1.0
            
            # Seasonal factor (simplified - higher in summer/winter)
            month = timestamp.month
            if month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.3
            elif month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.2
            else:
                seasonal_factor = 1.0
            
            weather_factor = self._weather_grid_demand_factor(timestamp, weather)
            solar_factor = self._solar_price_factor(weather)
            
            # Small bounded stochastic term. The weather and daily shape remain dominant.
            random_factor = 1 + min(self.config.price_volatility, 0.25) * (self._rng.random() - 0.5)
            
            # Calculate price
            price = (self.config.base_price * 
                    daily_factor * 
                    weekend_factor * 
                    seasonal_factor * 
                    weather_factor *
                    solar_factor *
                    random_factor)
            
            price = min(max(self.config.min_price, price), self.config.max_price)
            
            self._rtp_cache[cache_key] = price
        
        # Determine period name based on price level
        if price > self.config.base_price * 1.5:
            period_name = "high"
            is_peak = True
        elif price > self.config.base_price:
            period_name = "medium"
            is_peak = False
        else:
            period_name = "low"
            is_peak = False
        
        return price, period_name, is_peak

    def _weather_cache_key(self, weather: Optional[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
        """Cache RTP by rounded weather inputs that affect grid demand."""
        if not weather:
            return (None, None)
        dry_bulb = self._weather_value(weather, "dry_bulb_c", "temperature_c", "ambient_temp_c")
        ghi = self._weather_value(weather, "ghi", "global_horizontal_irradiance")
        return (
            round(dry_bulb, 1) if dry_bulb is not None else None,
            round(ghi, 0) if ghi is not None else None,
        )

    def _weather_value(self, weather: Dict[str, float], *keys: str) -> Optional[float]:
        for key in keys:
            if key in weather and weather[key] is not None:
                return float(weather[key])
        return None

    def _weather_grid_demand_factor(
        self,
        timestamp: datetime,
        weather: Optional[Dict[str, float]]
    ) -> float:
        """Approximate grid load pressure from heating/cooling degree stress."""
        if not weather:
            return 1.0
        dry_bulb = self._weather_value(weather, "dry_bulb_c", "temperature_c", "ambient_temp_c")
        if dry_bulb is None:
            return 1.0

        cooling_stress = max(0.0, dry_bulb - 24.0) / 12.0
        heating_stress = max(0.0, 12.0 - dry_bulb) / 18.0
        stress = min(1.5, cooling_stress + heating_stress)

        # Hot afternoons tend to stress summer grids more than hot nights.
        afternoon_multiplier = 1.25 if 14 <= timestamp.hour <= 20 else 1.0
        return 1.0 + self.config.weather_sensitivity * stress * afternoon_multiplier

    def _solar_price_factor(self, weather: Optional[Dict[str, float]]) -> float:
        """Discount RTP when solar irradiance is high, raise it slightly when scarce."""
        if not weather:
            return 1.0
        ghi = self._weather_value(weather, "ghi", "global_horizontal_irradiance")
        if ghi is None:
            return 1.0

        solar_abundance = min(1.0, max(0.0, ghi / 900.0))
        return 1.0 + self.config.solar_sensitivity * (0.5 - solar_abundance)
    
    def _get_dynamic_price(self, timestamp: datetime, 
                           load_factor: float = 1.0) -> Tuple[float, str, bool]:
        """
        Get dynamic price based on grid conditions.
        
        Args:
            load_factor: Current grid load factor (0-2, 1=normal)
        """
        # Start with RTP base
        base_price, _, _ = self._get_rtp_price(timestamp)
        
        # Adjust for load factor
        price = base_price * load_factor
        
        if load_factor > 1.5:
            period_name = "critical"
            is_peak = True
        elif load_factor > 1.2:
            period_name = "high"
            is_peak = True
        elif load_factor < 0.7:
            period_name = "low"
            is_peak = False
        else:
            period_name = "normal"
            is_peak = False
        
        return price, period_name, is_peak
    
    def _get_wholesale_price(self, timestamp: datetime) -> Tuple[float, str, bool]:
        """
        Simulate wholesale market price (e.g., LMP - Locational Marginal Price).
        
        More volatile than retail, can go negative during high renewable periods.
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base pattern similar to RTP but more volatile
        daily_factor = 0.3 + 0.7 * math.sin((hour - 3) * math.pi / 12)
        
        # Higher volatility
        random_factor = 1 + 2 * self.config.price_volatility * (self._rng.random() - 0.5)
        
        # Weekend effect
        weekend_factor = 0.6 if day_of_week >= 5 else 1.0
        
        # Calculate price
        price = (self.config.wholesale_base * 
                daily_factor * 
                weekend_factor * 
                random_factor)
        
        # Wholesale can go negative (curtailment periods)
        if self._rng.random() < 0.05:  # 5% chance of negative prices
            price = -0.02
        
        # Occasional price spikes
        if self._rng.random() < 0.02:  # 2% chance of spike
            price = self.config.wholesale_base * self.config.wholesale_peak_multiplier
        
        if price > self.config.wholesale_base * 2:
            period_name = "spike"
            is_peak = True
        elif price < 0:
            period_name = "negative"
            is_peak = False
        elif price > self.config.wholesale_base:
            period_name = "high"
            is_peak = True
        else:
            period_name = "normal"
            is_peak = False
        
        return price, period_name, is_peak
    
    def calculate_cost(
        self,
        timestamp: datetime,
        energy_consumed_kwh: float,
        energy_exported_kwh: float = 0.0,
        timestep_hours: float = 1.0,
        weather: Optional[Dict[str, float]] = None
    ) -> EnergyCostResult:
        """
        Calculate energy cost for a timestep.
        
        Args:
            timestamp: Datetime of the timestep
            energy_consumed_kwh: Energy consumed from grid (kWh)
            energy_exported_kwh: Energy exported to grid (kWh)
            timestep_hours: Duration of timestep in hours
            weather: Optional weather/grid context passed to RTP pricing
            
        Returns:
            EnergyCostResult with cost breakdown
        """
        # Get current price
        price_state = self.get_price(timestamp, weather=weather)
        
        # Calculate import cost
        import_cost = energy_consumed_kwh * price_state.price_per_kwh
        
        # Calculate export credit
        export_credit = energy_exported_kwh * price_state.feed_in_rate
        
        # Net cost
        net_cost = import_cost - export_credit
        
        # Carbon cost (only for imported energy)
        carbon_cost = energy_consumed_kwh * price_state.carbon_cost_per_kwh
        
        # Total cost
        total_cost = net_cost + carbon_cost
        
        result = EnergyCostResult(
            timestamp=timestamp,
            energy_consumed_kwh=energy_consumed_kwh,
            energy_exported_kwh=energy_exported_kwh,
            price_per_kwh=price_state.price_per_kwh,
            import_cost=import_cost,
            export_credit=export_credit,
            net_cost=net_cost,
            carbon_cost=carbon_cost,
            total_cost=total_cost
        )
        
        self.cost_history.append(result)
        return result
    
    def get_price_forecast(
        self,
        start: datetime,
        hours_ahead: int = 24
    ) -> pd.DataFrame:
        """
        Get price forecast for planning/optimization.
        
        Args:
            start: Start datetime
            hours_ahead: Number of hours to forecast
            
        Returns:
            DataFrame with hourly price forecast
        """
        forecasts = []
        for h in range(hours_ahead):
            ts = start + timedelta(hours=h)
            price_state = self.get_price(ts)
            forecasts.append({
                'timestamp': ts,
                'price_per_kwh': price_state.price_per_kwh,
                'period': price_state.period_name,
                'is_peak': price_state.is_peak,
                'feed_in_rate': price_state.feed_in_rate
            })
        
        return pd.DataFrame(forecasts)
    
    def get_cost_summary(self) -> Dict:
        """Get summary of accumulated costs."""
        if not self.cost_history:
            return {}
        
        total_consumed = sum(r.energy_consumed_kwh for r in self.cost_history)
        total_exported = sum(r.energy_exported_kwh for r in self.cost_history)
        total_import_cost = sum(r.import_cost for r in self.cost_history)
        total_export_credit = sum(r.export_credit for r in self.cost_history)
        total_carbon_cost = sum(r.carbon_cost for r in self.cost_history)
        total_cost = sum(r.total_cost for r in self.cost_history)
        
        avg_price = total_import_cost / total_consumed if total_consumed > 0 else 0
        
        return {
            'total_consumed_kwh': total_consumed,
            'total_exported_kwh': total_exported,
            'total_import_cost': total_import_cost,
            'total_export_credit': total_export_credit,
            'net_energy_cost': total_import_cost - total_export_credit,
            'total_carbon_cost': total_carbon_cost,
            'total_cost': total_cost,
            'average_price_per_kwh': avg_price,
            'num_timesteps': len(self.cost_history)
        }
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get cost history as DataFrame."""
        if not self.cost_history:
            return pd.DataFrame()
        
        data = [{
            'timestamp': r.timestamp,
            'energy_consumed_kwh': r.energy_consumed_kwh,
            'energy_exported_kwh': r.energy_exported_kwh,
            'price_per_kwh': r.price_per_kwh,
            'import_cost': r.import_cost,
            'export_credit': r.export_credit,
            'net_cost': r.net_cost,
            'carbon_cost': r.carbon_cost,
            'total_cost': r.total_cost
        } for r in self.cost_history]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


def create_tou_pricing(
    peak_price: float = 0.25,
    off_peak_price: float = 0.08,
    peak_hours: Tuple[int, int] = (14, 20),
    feed_in_tariff: float = 0.05
) -> EnergyPriceModel:
    """
    Create a simple TOU pricing model.
    
    Args:
        peak_price: Peak period price ($/kWh)
        off_peak_price: Off-peak price ($/kWh)
        peak_hours: (start_hour, end_hour) for peak period
        feed_in_tariff: Export rate ($/kWh)
    """
    config = PricingConfig(
        pricing_type=PricingType.TOU,
        tou_periods=[
            TOUPeriod("off-peak", 0, peak_hours[0], off_peak_price, [0,1,2,3,4,5,6]),
            TOUPeriod("peak", peak_hours[0], peak_hours[1], peak_price, [0,1,2,3,4]),
            TOUPeriod("off-peak", peak_hours[1], 24, off_peak_price, [0,1,2,3,4,5,6]),
            TOUPeriod("off-peak", peak_hours[0], peak_hours[1], off_peak_price, [5,6]),  # Weekend
        ],
        feed_in_tariff=feed_in_tariff
    )
    return EnergyPriceModel(config)


def create_rtp_pricing(
    base_price: float = 0.10,
    volatility: float = 0.3,
    feed_in_tariff: float = 0.05,
    weather_sensitivity: float = 0.35,
    solar_sensitivity: float = 0.15,
    min_price: float = 0.02,
    max_price: float = 0.60
) -> EnergyPriceModel:
    """
    Create a Real-Time Pricing model.
    
    Args:
        base_price: Base electricity price ($/kWh)
        volatility: Small bounded price volatility factor (0-1)
        feed_in_tariff: Export rate ($/kWh)
        weather_sensitivity: Price sensitivity to heating/cooling grid stress
        solar_sensitivity: Price discount when solar irradiance is abundant
        min_price: Minimum RTP price ($/kWh)
        max_price: Maximum RTP price ($/kWh)
    """
    config = PricingConfig(
        pricing_type=PricingType.RTP,
        base_price=base_price,
        price_volatility=volatility,
        weather_sensitivity=weather_sensitivity,
        solar_sensitivity=solar_sensitivity,
        min_price=min_price,
        max_price=max_price,
        feed_in_tariff=feed_in_tariff
    )
    return EnergyPriceModel(config)


def create_wholesale_pricing(
    base_price: float = 0.05,
    peak_multiplier: float = 5.0
) -> EnergyPriceModel:
    """
    Create a wholesale market pricing model.
    
    Args:
        base_price: Base wholesale price ($/kWh)
        peak_multiplier: Multiplier for price spikes
    """
    config = PricingConfig(
        pricing_type=PricingType.WHOLESALE,
        wholesale_base=base_price,
        wholesale_peak_multiplier=peak_multiplier,
        feed_in_tariff=base_price  # Export at wholesale rate
    )
    return EnergyPriceModel(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Energy Price Model Example")
    print("=" * 60)
    
    # Test TOU pricing
    print("\n--- Time-of-Use Pricing ---")
    tou = create_tou_pricing(peak_price=0.30, off_peak_price=0.08)
    
    test_times = [
        datetime(2024, 7, 15, 3, 0),   # Night (off-peak)
        datetime(2024, 7, 15, 10, 0),  # Morning (mid-peak)
        datetime(2024, 7, 15, 15, 0),  # Afternoon (peak)
        datetime(2024, 7, 15, 21, 0),  # Evening (mid-peak)
        datetime(2024, 7, 14, 15, 0),  # Sunday afternoon (off-peak)
    ]
    
    print(f"\n{'Timestamp':<25} {'Price':>10} {'Period':<12}")
    print("-" * 50)
    for ts in test_times:
        state = tou.get_price(ts)
        print(f"{ts.strftime('%Y-%m-%d %H:%M %a'):<25} ${state.price_per_kwh:>8.3f} {state.period_name:<12}")
    
    # Test RTP pricing
    print("\n--- Real-Time Pricing (24 hours) ---")
    rtp = create_rtp_pricing(base_price=0.10, volatility=0.4)
    
    start = datetime(2024, 7, 15, 0, 0)
    forecast = rtp.get_price_forecast(start, hours_ahead=24)
    
    print(f"\nHourly prices for {start.date()}:")
    for _, row in forecast.iterrows():
        bar = '█' * int(row['price_per_kwh'] * 50)
        print(f"{row['timestamp'].strftime('%H:%M')} | ${row['price_per_kwh']:.3f} | {bar}")
    
    # Test cost calculation
    print("\n--- Cost Calculation Example ---")
    price_model = create_tou_pricing()
    
    # Simulate a day with varying consumption and export
    print(f"\n{'Hour':<6} {'Consumed':>10} {'Exported':>10} {'Price':>8} {'Net Cost':>10}")
    print("-" * 50)
    
    for hour in range(24):
        ts = datetime(2024, 7, 15, hour, 0)
        
        # Simulated consumption pattern
        if 8 <= hour <= 18:
            consumed = 50 + 20 * math.sin((hour - 8) * math.pi / 10)
        else:
            consumed = 20
        
        # Simulated PV export (midday)
        if 10 <= hour <= 16:
            exported = 30 * math.sin((hour - 10) * math.pi / 6)
        else:
            exported = 0
        
        result = price_model.calculate_cost(ts, consumed, exported)
        
        if hour % 4 == 0:  # Print every 4 hours
            print(f"{hour:>4}:00 {consumed:>10.1f} {exported:>10.1f} "
                  f"${result.price_per_kwh:>6.3f} ${result.net_cost:>9.2f}")
    
    # Summary
    summary = price_model.get_cost_summary()
    print(f"\n{'='*50}")
    print("DAILY COST SUMMARY")
    print(f"{'='*50}")
    print(f"Total Consumed: {summary['total_consumed_kwh']:,.1f} kWh")
    print(f"Total Exported: {summary['total_exported_kwh']:,.1f} kWh")
    print(f"Import Cost: ${summary['total_import_cost']:,.2f}")
    print(f"Export Credit: ${summary['total_export_credit']:,.2f}")
    print(f"Net Energy Cost: ${summary['net_energy_cost']:,.2f}")
    print(f"Average Price: ${summary['average_price_per_kwh']:.3f}/kWh")
