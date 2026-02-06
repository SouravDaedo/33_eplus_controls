"""
Solar PV Model using PVWatts Methodology

A configurable Python-based solar PV model that reads weather files (EPW format)
and calculates timestep power production based on system parameters and location.

Uses simplified PVWatts methodology for DC and AC power calculations.

Author: Generated for EnergyPlus Controls Project
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from datetime import datetime, timedelta
import math


@dataclass
class PVSystemConfig:
    """Configuration parameters for the PV system."""
    
    # System size
    system_capacity_kw: float = 100.0  # DC nameplate capacity (kW)
    
    # Module parameters
    module_type: int = 0  # 0=Standard, 1=Premium, 2=Thin film
    
    # Array configuration
    array_type: int = 0  # 0=Fixed open rack, 1=Fixed roof mount, 2=1-axis tracking, 3=2-axis tracking
    tilt_deg: float = 20.0  # Tilt angle (degrees from horizontal)
    azimuth_deg: float = 180.0  # Azimuth angle (degrees, 180=South)
    
    # System losses
    system_losses_pct: float = 14.0  # Total system losses (%)
    
    # Inverter
    dc_ac_ratio: float = 1.2  # DC to AC ratio
    inverter_efficiency_pct: float = 96.0  # Inverter efficiency (%)
    
    # Ground coverage ratio (for tracking systems)
    gcr: float = 0.4  # Ground coverage ratio
    
    # Temperature coefficients (per °C)
    temp_coeff_pmax: float = -0.0047  # Power temperature coefficient (%/°C for standard Si)
    noct: float = 45.0  # Nominal Operating Cell Temperature (°C)
    
    # Reference conditions
    ref_irradiance: float = 1000.0  # W/m² at STC
    ref_temperature: float = 25.0  # °C at STC
    
    def __post_init__(self):
        """Set module-specific parameters based on module_type."""
        if self.module_type == 0:  # Standard
            self.temp_coeff_pmax = -0.0047
        elif self.module_type == 1:  # Premium
            self.temp_coeff_pmax = -0.0035
        elif self.module_type == 2:  # Thin film
            self.temp_coeff_pmax = -0.0020


@dataclass
class PVState:
    """Current state of the PV system at a timestep."""
    timestamp: datetime
    ghi: float  # Global Horizontal Irradiance (W/m²)
    dni: float  # Direct Normal Irradiance (W/m²)
    dhi: float  # Diffuse Horizontal Irradiance (W/m²)
    ambient_temp_c: float  # Ambient temperature (°C)
    wind_speed_m_s: float  # Wind speed (m/s)
    poa_irradiance: float  # Plane of Array irradiance (W/m²)
    cell_temperature_c: float  # Cell temperature (°C)
    dc_power_kw: float  # DC power output (kW)
    ac_power_kw: float  # AC power output (kW)


class EPWReader:
    """Reader for EnergyPlus Weather (EPW) files."""
    
    EPW_COLUMNS = [
        'year', 'month', 'day', 'hour', 'minute', 'data_source',
        'dry_bulb_c', 'dew_point_c', 'rel_humidity', 'atm_pressure',
        'ext_hor_rad', 'ext_dir_rad', 'hor_ir_sky', 'ghi', 'dni', 'dhi',
        'global_illum', 'direct_illum', 'diffuse_illum', 'zenith_illum',
        'wind_dir', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather', 'precip_water',
        'aerosol_depth', 'snow_depth', 'days_since_snow', 'albedo',
        'liquid_precip_depth', 'liquid_precip_rate'
    ]
    
    def __init__(self, filepath: str):
        """
        Initialize EPW reader.
        
        Args:
            filepath: Path to EPW weather file
        """
        self.filepath = Path(filepath)
        self.location = {}
        self.data = None
        self._read_file()
    
    def _read_file(self):
        """Read and parse the EPW file."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header (first 8 lines)
        # Line 1: LOCATION
        loc_parts = lines[0].strip().split(',')
        self.location = {
            'city': loc_parts[1] if len(loc_parts) > 1 else '',
            'state': loc_parts[2] if len(loc_parts) > 2 else '',
            'country': loc_parts[3] if len(loc_parts) > 3 else '',
            'source': loc_parts[4] if len(loc_parts) > 4 else '',
            'wmo': loc_parts[5] if len(loc_parts) > 5 else '',
            'latitude': float(loc_parts[6]) if len(loc_parts) > 6 else 0,
            'longitude': float(loc_parts[7]) if len(loc_parts) > 7 else 0,
            'timezone': float(loc_parts[8]) if len(loc_parts) > 8 else 0,
            'elevation': float(loc_parts[9]) if len(loc_parts) > 9 else 0
        }
        
        # Parse data (skip first 8 header lines)
        data_lines = lines[8:]
        data = []
        for line in data_lines:
            if line.strip():
                values = line.strip().split(',')
                if len(values) >= 34:
                    data.append(values[:34])
        
        # Create DataFrame
        self.data = pd.DataFrame(data, columns=self.EPW_COLUMNS)
        
        # Convert numeric columns
        numeric_cols = ['year', 'month', 'day', 'hour', 'minute', 
                       'dry_bulb_c', 'ghi', 'dni', 'dhi', 'wind_speed']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # EPW uses hours 1-24, converting to 0-23 for datetime where hour 24 becomes hour 0 of next day
        self.data['hour_adj'] = self.data['hour'].astype(int) - 1
        self.data['day_adj'] = self.data['day'].astype(int)
        
        mask_hour24 = self.data['hour_adj'] == -1
        self.data.loc[mask_hour24, 'hour_adj'] = 23  # Will be previous day's hour 23
        
        # Create datetime index
        self.data['datetime'] = pd.to_datetime({
            'year': self.data['year'].astype(int),
            'month': self.data['month'].astype(int),
            'day': self.data['day_adj'],
            'hour': self.data['hour_adj']
        })
        self.data.set_index('datetime', inplace=True)
        
        # Sort index to ensure monotonic
        self.data.sort_index(inplace=True)
        
        # Drop helper columns
        self.data.drop(['hour_adj', 'day_adj'], axis=1, inplace=True)
    
    def get_data(self) -> pd.DataFrame:
        """Get weather data as DataFrame."""
        return self.data
    
    def get_location(self) -> dict:
        """Get location information."""
        return self.location


class SolarPVModel:
    """
    Solar PV System Model using PVWatts methodology.
    
    Calculates power output based on weather data and system configuration.
    
    Example usage:
        config = PVSystemConfig(system_capacity_kw=100, tilt_deg=30)
        pv = SolarPVModel(config)
        pv.load_weather('weather/chicago/TMY.epw')
        
        # Get production for specific timestep
        result = pv.get_power_at_timestep(timestamp)
        
        # Or simulate full year
        results = pv.simulate()
    """
    
    def __init__(self, config: Optional[PVSystemConfig] = None):
        """
        Initialize the PV model.
        
        Args:
            config: PV system configuration. Uses defaults if not provided.
        """
        self.config = config or PVSystemConfig()
        self.weather_data = None
        self.location = None
        self.results = None
    
    def load_weather(self, filepath: str) -> dict:
        """
        Load weather data from EPW file.
        
        Args:
            filepath: Path to EPW weather file
            
        Returns:
            Location dictionary
        """
        reader = EPWReader(filepath)
        self.weather_data = reader.get_data()
        self.location = reader.get_location()
        
        # Auto-set tilt to latitude if not explicitly set
        if self.config.tilt_deg == 20.0:  # Default value
            self.config.tilt_deg = abs(self.location['latitude'])
        
        return self.location
    
    def _calculate_sun_position(self, timestamp: datetime) -> Tuple[float, float]:
        """
        Calculate solar zenith and azimuth angles.
        
        Args:
            timestamp: Datetime for calculation
            
        Returns:
            (zenith_deg, azimuth_deg) tuple
        """
        lat = math.radians(self.location['latitude'])
        lon = self.location['longitude']
        tz = self.location['timezone']
        
        # Day of year
        doy = timestamp.timetuple().tm_yday
        
        # Solar declination (Spencer, 1971)
        B = 2 * math.pi * (doy - 1) / 365
        declination = (0.006918 - 0.399912 * math.cos(B) + 0.070257 * math.sin(B)
                      - 0.006758 * math.cos(2*B) + 0.000907 * math.sin(2*B)
                      - 0.002697 * math.cos(3*B) + 0.00148 * math.sin(3*B))
        
        # Equation of time (minutes)
        eot = 229.18 * (0.000075 + 0.001868 * math.cos(B) - 0.032077 * math.sin(B)
                       - 0.014615 * math.cos(2*B) - 0.040849 * math.sin(2*B))
        
        # Solar time
        solar_time = timestamp.hour + timestamp.minute/60 + (4 * (lon - 15*tz) + eot) / 60
        
        # Hour angle
        hour_angle = math.radians(15 * (solar_time - 12))
        
        # Solar zenith angle
        cos_zenith = (math.sin(lat) * math.sin(declination) + 
                     math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
        cos_zenith = max(-1, min(1, cos_zenith))
        zenith = math.acos(cos_zenith)
        
        # Solar azimuth angle
        if math.cos(zenith) != 0:
            cos_azimuth = ((math.sin(declination) * math.cos(lat) - 
                          math.cos(declination) * math.sin(lat) * math.cos(hour_angle)) / 
                         math.sin(zenith))
            cos_azimuth = max(-1, min(1, cos_azimuth))
            azimuth = math.acos(cos_azimuth)
            if hour_angle > 0:
                azimuth = 2 * math.pi - azimuth
        else:
            azimuth = math.pi
        
        return math.degrees(zenith), math.degrees(azimuth)
    
    def _calculate_poa_irradiance(self, ghi: float, dni: float, dhi: float,
                                   zenith_deg: float, azimuth_deg: float) -> float:
        """
        Calculate Plane of Array (POA) irradiance.
        
        Uses isotropic sky model for simplicity.
        
        Args:
            ghi: Global Horizontal Irradiance (W/m²)
            dni: Direct Normal Irradiance (W/m²)
            dhi: Diffuse Horizontal Irradiance (W/m²)
            zenith_deg: Solar zenith angle (degrees)
            azimuth_deg: Solar azimuth angle (degrees)
            
        Returns:
            POA irradiance (W/m²)
        """
        if zenith_deg >= 90:
            return 0.0
        
        tilt = math.radians(self.config.tilt_deg)
        surface_azimuth = math.radians(self.config.azimuth_deg)
        zenith = math.radians(zenith_deg)
        azimuth = math.radians(azimuth_deg)
        
        # Angle of incidence
        cos_aoi = (math.sin(zenith) * math.sin(tilt) * math.cos(azimuth - surface_azimuth) +
                  math.cos(zenith) * math.cos(tilt))
        cos_aoi = max(0, cos_aoi)
        
        # Direct beam on tilted surface
        beam = dni * cos_aoi
        
        # Diffuse (isotropic sky model)
        diffuse = dhi * (1 + math.cos(tilt)) / 2
        
        # Ground reflected
        albedo = 0.2
        ground = ghi * albedo * (1 - math.cos(tilt)) / 2
        
        poa = beam + diffuse + ground
        return max(0, poa)
    
    def _calculate_cell_temperature(self, poa: float, ambient_temp: float, 
                                     wind_speed: float) -> float:
        """
        Calculate cell temperature using NOCT model.
        
        Args:
            poa: POA irradiance (W/m²)
            ambient_temp: Ambient temperature (°C)
            wind_speed: Wind speed (m/s)
            
        Returns:
            Cell temperature (°C)
        """
        # Sandia cell temperature model (simplified)
        # For open rack: a=-3.56, b=-0.075
        # For roof mount: a=-2.81, b=-0.0455
        
        if self.config.array_type == 0:  # Open rack
            a, b = -3.56, -0.075
        else:  # Roof mount or tracking
            a, b = -2.81, -0.0455
        
        # Cell temperature
        cell_temp = poa * math.exp(a + b * wind_speed) + ambient_temp
        
        return cell_temp
    
    def _calculate_dc_power(self, poa: float, cell_temp: float) -> float:
        """
        Calculate DC power output.
        
        Args:
            poa: POA irradiance (W/m²)
            cell_temp: Cell temperature (°C)
            
        Returns:
            DC power (kW)
        """
        if poa <= 0:
            return 0.0
        
        # Temperature correction
        temp_diff = cell_temp - self.config.ref_temperature
        temp_factor = 1 + self.config.temp_coeff_pmax * temp_diff
        
        # DC power (PVWatts method)
        dc_power = (self.config.system_capacity_kw * 
                   (poa / self.config.ref_irradiance) * 
                   temp_factor)
        
        # Apply system losses
        dc_power *= (1 - self.config.system_losses_pct / 100)
        
        return max(0, dc_power)
    
    def _calculate_ac_power(self, dc_power: float) -> float:
        """
        Calculate AC power output with inverter model.
        
        Args:
            dc_power: DC power (kW)
            
        Returns:
            AC power (kW)
        """
        if dc_power <= 0:
            return 0.0
        
        # Inverter capacity
        inverter_capacity = self.config.system_capacity_kw / self.config.dc_ac_ratio
        
        # Inverter efficiency curve (simplified PVWatts)
        # Efficiency varies with load
        load_fraction = dc_power / (self.config.system_capacity_kw)
        
        if load_fraction < 0.1:
            efficiency = self.config.inverter_efficiency_pct / 100 * load_fraction / 0.1
        else:
            efficiency = self.config.inverter_efficiency_pct / 100
        
        ac_power = dc_power * efficiency
        
        # Clip to inverter capacity
        ac_power = min(ac_power, inverter_capacity)
        
        return max(0, ac_power)
    
    def get_power_at_timestep(self, timestamp: datetime) -> PVState:
        """
        Calculate PV power output for a specific timestep.
        
        Args:
            timestamp: Datetime for calculation
            
        Returns:
            PVState with all calculated values
        """
        if self.weather_data is None:
            raise ValueError("Weather data not loaded. Call load_weather() first.")
        
        # Find closest weather data
        try:
            # Try exact match first
            weather = self.weather_data.loc[timestamp]
        except KeyError:
            # Find nearest timestamp
            idx = self.weather_data.index.get_indexer([timestamp], method='nearest')[0]
            weather = self.weather_data.iloc[idx]
        
        ghi = float(weather['ghi'])
        dni = float(weather['dni'])
        dhi = float(weather['dhi'])
        ambient_temp = float(weather['dry_bulb_c'])
        wind_speed = float(weather['wind_speed'])
        
        # Calculate sun position
        zenith, azimuth = self._calculate_sun_position(timestamp)
        
        # Calculate POA irradiance
        poa = self._calculate_poa_irradiance(ghi, dni, dhi, zenith, azimuth)
        
        # Calculate cell temperature
        cell_temp = self._calculate_cell_temperature(poa, ambient_temp, wind_speed)
        
        # Calculate DC power
        dc_power = self._calculate_dc_power(poa, cell_temp)
        
        # Calculate AC power
        ac_power = self._calculate_ac_power(dc_power)
        
        return PVState(
            timestamp=timestamp,
            ghi=ghi,
            dni=dni,
            dhi=dhi,
            ambient_temp_c=ambient_temp,
            wind_speed_m_s=wind_speed,
            poa_irradiance=poa,
            cell_temperature_c=cell_temp,
            dc_power_kw=dc_power,
            ac_power_kw=ac_power
        )
    
    def simulate(self, start: Optional[datetime] = None, 
                 end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Simulate PV production over a time range.
        
        Args:
            start: Start datetime (uses first weather timestamp if None)
            end: End datetime (uses last weather timestamp if None)
            
        Returns:
            DataFrame with simulation results
        """
        if self.weather_data is None:
            raise ValueError("Weather data not loaded. Call load_weather() first.")
        
        # Determine time range
        if start is None:
            start = self.weather_data.index[0]
        if end is None:
            end = self.weather_data.index[-1]
        
        # Filter weather data
        mask = (self.weather_data.index >= start) & (self.weather_data.index <= end)
        timestamps = self.weather_data.index[mask]
        
        results = []
        for ts in timestamps:
            state = self.get_power_at_timestep(ts)
            results.append({
                'timestamp': state.timestamp,
                'ghi': state.ghi,
                'dni': state.dni,
                'dhi': state.dhi,
                'ambient_temp_c': state.ambient_temp_c,
                'wind_speed_m_s': state.wind_speed_m_s,
                'poa_irradiance': state.poa_irradiance,
                'cell_temperature_c': state.cell_temperature_c,
                'dc_power_kw': state.dc_power_kw,
                'ac_power_kw': state.ac_power_kw
            })
        
        if not results:
            raise ValueError(f"No weather data found for date range {start} to {end}. "
                           f"Weather file contains: {self.weather_data.index[0]} to {self.weather_data.index[-1]}")
        
        self.results = pd.DataFrame(results)
        self.results.set_index('timestamp', inplace=True)
        
        return self.results
    
    def get_daily_production(self) -> pd.DataFrame:
        """Get daily energy production summary."""
        if self.results is None:
            raise ValueError("No simulation results. Call simulate() first.")
        
        daily = self.results.resample('D').agg({
            'ac_power_kw': 'sum',  # kWh (assuming hourly data)
            'dc_power_kw': 'sum',
            'poa_irradiance': 'sum',
            'ghi': 'sum'
        })
        daily.columns = ['ac_energy_kwh', 'dc_energy_kwh', 'poa_wh_m2', 'ghi_wh_m2']
        return daily
    
    def get_monthly_production(self) -> pd.DataFrame:
        """Get monthly energy production summary."""
        if self.results is None:
            raise ValueError("No simulation results. Call simulate() first.")
        
        monthly = self.results.resample('M').agg({
            'ac_power_kw': 'sum',
            'dc_power_kw': 'sum',
            'poa_irradiance': 'sum',
            'ghi': 'sum'
        })
        monthly.columns = ['ac_energy_kwh', 'dc_energy_kwh', 'poa_wh_m2', 'ghi_wh_m2']
        return monthly
    
    def get_annual_production(self) -> dict:
        """Get annual energy production summary."""
        if self.results is None:
            raise ValueError("No simulation results. Call simulate() first.")
        
        return {
            'ac_energy_kwh': self.results['ac_power_kw'].sum(),
            'dc_energy_kwh': self.results['dc_power_kw'].sum(),
            'capacity_factor': self.results['ac_power_kw'].sum() / (self.config.system_capacity_kw / self.config.dc_ac_ratio * len(self.results)),
            'specific_yield_kwh_kwp': self.results['ac_power_kw'].sum() / self.config.system_capacity_kw
        }


def create_pv_system(
    capacity_kw: float = 100.0,
    tilt_deg: Optional[float] = None,
    azimuth_deg: float = 180.0,
    module_type: int = 0,
    losses_pct: float = 14.0
) -> SolarPVModel:
    """
    Create a PV system model with common parameters.
    
    Args:
        capacity_kw: DC nameplate capacity (kW)
        tilt_deg: Tilt angle (None = use latitude)
        azimuth_deg: Azimuth angle (180 = South)
        module_type: 0=Standard, 1=Premium, 2=Thin film
        losses_pct: System losses percentage
        
    Returns:
        Configured SolarPVModel instance
    """
    config = PVSystemConfig(
        system_capacity_kw=capacity_kw,
        tilt_deg=tilt_deg if tilt_deg is not None else 20.0,
        azimuth_deg=azimuth_deg,
        module_type=module_type,
        system_losses_pct=losses_pct
    )
    return SolarPVModel(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Solar PV Model Example")
    print("=" * 60)
    
    # Create a 100 kW PV system
    pv = create_pv_system(
        capacity_kw=100,
        azimuth_deg=180,  # South-facing
        module_type=0,  # Standard
        losses_pct=14
    )
    
    # Load weather file
    weather_file = "weather/chicago/TMY_lat41.88_lon-87.63.epw"
    print(f"\nLoading weather: {weather_file}")
    
    try:
        location = pv.load_weather(weather_file)
        print(f"Location: {location['city']}, {location['state']}")
        print(f"Latitude: {location['latitude']:.2f}°")
        print(f"Longitude: {location['longitude']:.2f}°")
        print(f"Tilt set to: {pv.config.tilt_deg:.1f}° (latitude)")
        
        # Simulate full year
        print("\nSimulating annual production...")
        results = pv.simulate()
        
        # Show sample results
        print(f"\nSample hourly results (first 5 daylight hours):")
        daylight = results[results['ac_power_kw'] > 0].head(5)
        print(daylight[['ghi', 'poa_irradiance', 'dc_power_kw', 'ac_power_kw']].to_string())
        
        # Annual summary
        annual = pv.get_annual_production()
        print(f"\n{'='*60}")
        print("ANNUAL PRODUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"  AC Energy: {annual['ac_energy_kwh']:,.0f} kWh")
        print(f"  DC Energy: {annual['dc_energy_kwh']:,.0f} kWh")
        print(f"  Capacity Factor: {annual['capacity_factor']:.1%}")
        print(f"  Specific Yield: {annual['specific_yield_kwh_kwp']:,.0f} kWh/kWp")
        
        # Monthly summary
        print(f"\nMonthly Production (kWh):")
        monthly = pv.get_monthly_production()
        for idx, row in monthly.iterrows():
            print(f"  {idx.strftime('%B')}: {row['ac_energy_kwh']:,.0f} kWh")
            
    except FileNotFoundError:
        print(f"Weather file not found: {weather_file}")
        print("Please provide a valid EPW weather file path.")
