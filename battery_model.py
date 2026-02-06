"""
Battery Energy Storage System (BESS) Model

A configurable Python-based battery model for use in building energy simulations.
Tracks state of charge (SOC) at each timestep based on control actions for 
charging/discharging from grid or solar PV.

Author: Generated for EnergyPlus Controls Project
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class BatteryAction(Enum):
    """Control actions for the battery."""
    IDLE = 0
    CHARGE_FROM_GRID = 1
    CHARGE_FROM_PV = 2
    DISCHARGE_TO_LOAD = 3
    DISCHARGE_TO_GRID = 4


@dataclass
class BatteryConfig:
    """Configuration parameters for the battery model."""
    
    # Timestep
    timestep_hours: float = 1/12  # Simulation timestep in hours (default: 5 minutes)
    
    # Capacity
    capacity_kwh: float = 100.0  # Total energy capacity (kWh)
    
    # State of Charge limits
    soc_min: float = 0.1  # Minimum SOC (fraction, 0-1)
    soc_max: float = 0.9  # Maximum SOC (fraction, 0-1)
    soc_initial: float = 0.5  # Initial SOC (fraction, 0-1)
    
    # Power limits
    max_charge_rate_kw: float = 50.0  # Maximum charging power (kW)
    max_discharge_rate_kw: float = 50.0  # Maximum discharging power (kW)
    
    # Efficiency
    charge_efficiency: float = 0.95  # Charging efficiency (0-1)
    discharge_efficiency: float = 0.95  # Discharging efficiency (0-1)
    roundtrip_efficiency: float = None  # If set, overrides charge/discharge efficiencies
    
    # Self-discharge
    self_discharge_rate: float = 0.0001  # Self-discharge rate per hour (fraction)
    
    # Degradation (optional)
    cycle_degradation_rate: float = 0.0  # Capacity loss per full cycle (fraction)
    calendar_degradation_rate: float = 0.0  # Capacity loss per day (fraction)
    
    # Thermal limits (optional)
    min_operating_temp_c: float = -20.0
    max_operating_temp_c: float = 45.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.roundtrip_efficiency is not None:
            # Derive charge/discharge efficiencies from roundtrip
            self.charge_efficiency = np.sqrt(self.roundtrip_efficiency)
            self.discharge_efficiency = np.sqrt(self.roundtrip_efficiency)
        
        # Validate SOC limits
        assert 0 <= self.soc_min < self.soc_max <= 1, "Invalid SOC limits"
        assert self.soc_min <= self.soc_initial <= self.soc_max, "Initial SOC outside limits"
        
        # Validate efficiencies
        assert 0 < self.charge_efficiency <= 1, "Invalid charge efficiency"
        assert 0 < self.discharge_efficiency <= 1, "Invalid discharge efficiency"


@dataclass
class BatteryState:
    """Current state of the battery."""
    soc: float  # State of charge (fraction, 0-1)
    energy_kwh: float  # Current stored energy (kWh)
    capacity_kwh: float  # Current capacity (may degrade over time)
    temperature_c: float = 25.0  # Battery temperature
    cycles: float = 0.0  # Cumulative equivalent full cycles
    total_energy_charged_kwh: float = 0.0
    total_energy_discharged_kwh: float = 0.0
    timestep: int = 0


@dataclass
class StepResult:
    """Result of a single simulation step."""
    action: BatteryAction
    power_requested_kw: float  # Power requested by control
    power_actual_kw: float  # Actual power (limited by constraints)
    energy_change_kwh: float  # Change in stored energy
    soc_before: float
    soc_after: float
    grid_power_kw: float  # Power from/to grid (positive = from grid)
    pv_power_used_kw: float  # PV power used for charging
    losses_kwh: float  # Energy losses
    constrained: bool  # Whether power was limited by constraints
    constraint_reason: str = ""


class BatteryModel:
    """
    Battery Energy Storage System Model.
    
    Simulates battery behavior at each timestep based on control actions.
    Supports charging from grid or PV, and discharging to load or grid.
    
    Example usage:
        config = BatteryConfig(capacity_kwh=100, max_charge_rate_kw=25)
        battery = BatteryModel(config)
        
        # Simulation loop
        for step in range(num_steps):
            result = battery.step(
                action=BatteryAction.CHARGE_FROM_PV,
                power_kw=20.0,
                pv_available_kw=25.0,
                timestep_hours=1/12  # 5-minute timestep
            )
            print(f"SOC: {result.soc_after:.2%}")
    """
    
    def __init__(self, config: Optional[BatteryConfig] = None):
        """
        Initialize the battery model.
        
        Args:
            config: Battery configuration. Uses defaults if not provided.
        """
        self.config = config or BatteryConfig()
        self.reset()
    
    def reset(self, soc_initial: Optional[float] = None) -> BatteryState:
        """
        Reset battery to initial state.
        
        Args:
            soc_initial: Initial SOC. Uses config value if not provided.
            
        Returns:
            Initial battery state.
        """
        soc = soc_initial if soc_initial is not None else self.config.soc_initial
        soc = np.clip(soc, self.config.soc_min, self.config.soc_max)
        
        self.state = BatteryState(
            soc=soc,
            energy_kwh=soc * self.config.capacity_kwh,
            capacity_kwh=self.config.capacity_kwh,
            temperature_c=25.0,
            cycles=0.0,
            total_energy_charged_kwh=0.0,
            total_energy_discharged_kwh=0.0,
            timestep=0
        )
        
        self.history: List[StepResult] = []
        return self.state
    
    def step(
        self,
        action: BatteryAction,
        power_kw: float = 0.0,
        pv_available_kw: float = 0.0,
        load_demand_kw: float = 0.0,
        timestep_hours: Optional[float] = None,  # Uses config if not provided
        ambient_temp_c: float = 25.0
    ) -> StepResult:
        """
        Execute one simulation step.
        
        Args:
            action: Control action (charge/discharge/idle)
            power_kw: Requested power for charging/discharging (kW)
            pv_available_kw: Available PV power (kW)
            load_demand_kw: Building load demand (kW)
            timestep_hours: Duration of timestep in hours (uses config default if not provided)
            ambient_temp_c: Ambient temperature (°C)
            
        Returns:
            StepResult with details of the step.
        """
        # Use configured timestep if not provided
        if timestep_hours is None:
            timestep_hours = self.config.timestep_hours
        
        soc_before = self.state.soc
        constrained = False
        constraint_reason = ""
        grid_power = 0.0
        pv_power_used = 0.0
        losses = 0.0
        
        # Apply self-discharge first
        self_discharge_energy = (
            self.state.energy_kwh * 
            self.config.self_discharge_rate * 
            timestep_hours
        )
        self.state.energy_kwh -= self_discharge_energy
        losses += self_discharge_energy
        
        # Process action
        if action == BatteryAction.IDLE:
            power_actual = 0.0
            energy_change = 0.0
            
        elif action == BatteryAction.CHARGE_FROM_GRID:
            power_actual, energy_change, constrained, constraint_reason = (
                self._charge(power_kw, timestep_hours)
            )
            grid_power = power_actual  # Positive = drawing from grid
            losses += power_actual * timestep_hours - energy_change
            
        elif action == BatteryAction.CHARGE_FROM_PV:
            # Limit to available PV
            charge_power = min(power_kw, pv_available_kw)
            power_actual, energy_change, constrained, constraint_reason = (
                self._charge(charge_power, timestep_hours)
            )
            pv_power_used = power_actual
            if power_kw > pv_available_kw:
                constrained = True
                constraint_reason = f"Limited by PV availability ({pv_available_kw:.1f} kW)"
            losses += power_actual * timestep_hours - energy_change
            
        elif action == BatteryAction.DISCHARGE_TO_LOAD:
            # Limit to load demand
            discharge_power = min(power_kw, load_demand_kw) if load_demand_kw > 0 else power_kw
            power_actual, energy_change, constrained, constraint_reason = (
                self._discharge(discharge_power, timestep_hours)
            )
            grid_power = -power_actual  # Negative = reducing grid draw
            losses += abs(energy_change) - power_actual * timestep_hours
            
        elif action == BatteryAction.DISCHARGE_TO_GRID:
            power_actual, energy_change, constrained, constraint_reason = (
                self._discharge(power_kw, timestep_hours)
            )
            grid_power = -power_actual  # Negative = exporting to grid
            losses += abs(energy_change) - power_actual * timestep_hours
        
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Update SOC
        self.state.soc = self.state.energy_kwh / self.state.capacity_kwh
        self.state.soc = np.clip(self.state.soc, 0.0, 1.0)
        
        # Update cycle count (simplified: based on energy throughput)
        if energy_change != 0:
            cycle_fraction = abs(energy_change) / (2 * self.state.capacity_kwh)
            self.state.cycles += cycle_fraction
        
        # Update totals
        if energy_change > 0:
            self.state.total_energy_charged_kwh += energy_change
        else:
            self.state.total_energy_discharged_kwh += abs(energy_change)
        
        self.state.timestep += 1
        
        # Create result
        result = StepResult(
            action=action,
            power_requested_kw=power_kw,
            power_actual_kw=power_actual,
            energy_change_kwh=energy_change,
            soc_before=soc_before,
            soc_after=self.state.soc,
            grid_power_kw=grid_power,
            pv_power_used_kw=pv_power_used,
            losses_kwh=losses,
            constrained=constrained,
            constraint_reason=constraint_reason
        )
        
        self.history.append(result)
        return result
    
    def _charge(
        self, 
        power_kw: float, 
        timestep_hours: float
    ) -> Tuple[float, float, bool, str]:
        """
        Process charging action.
        
        Returns:
            (actual_power, energy_change, constrained, reason)
        """
        constrained = False
        reason = ""
        
        # Limit by max charge rate
        if power_kw > self.config.max_charge_rate_kw:
            power_kw = self.config.max_charge_rate_kw
            constrained = True
            reason = f"Limited by max charge rate ({self.config.max_charge_rate_kw} kW)"
        
        # Calculate energy that would be stored (after efficiency losses)
        energy_in = power_kw * timestep_hours
        energy_stored = energy_in * self.config.charge_efficiency
        
        # Limit by available capacity
        max_energy = self.config.soc_max * self.state.capacity_kwh
        available_capacity = max_energy - self.state.energy_kwh
        
        if energy_stored > available_capacity:
            energy_stored = available_capacity
            energy_in = energy_stored / self.config.charge_efficiency
            power_kw = energy_in / timestep_hours
            constrained = True
            reason = f"Limited by SOC max ({self.config.soc_max:.0%})"
        
        # Update energy
        self.state.energy_kwh += energy_stored
        
        return power_kw, energy_stored, constrained, reason
    
    def _discharge(
        self, 
        power_kw: float, 
        timestep_hours: float
    ) -> Tuple[float, float, bool, str]:
        """
        Process discharging action.
        
        Returns:
            (actual_power, energy_change, constrained, reason)
        """
        constrained = False
        reason = ""
        
        # Limit by max discharge rate
        if power_kw > self.config.max_discharge_rate_kw:
            power_kw = self.config.max_discharge_rate_kw
            constrained = True
            reason = f"Limited by max discharge rate ({self.config.max_discharge_rate_kw} kW)"
        
        # Calculate energy needed from battery (before efficiency losses)
        energy_out = power_kw * timestep_hours
        energy_from_battery = energy_out / self.config.discharge_efficiency
        
        # Limit by available energy
        min_energy = self.config.soc_min * self.state.capacity_kwh
        available_energy = self.state.energy_kwh - min_energy
        
        if energy_from_battery > available_energy:
            energy_from_battery = available_energy
            energy_out = energy_from_battery * self.config.discharge_efficiency
            power_kw = energy_out / timestep_hours
            constrained = True
            reason = f"Limited by SOC min ({self.config.soc_min:.0%})"
        
        # Update energy (negative change for discharge)
        self.state.energy_kwh -= energy_from_battery
        
        return power_kw, -energy_from_battery, constrained, reason
    
    def get_state(self) -> BatteryState:
        """Get current battery state."""
        return self.state
    
    def get_soc(self) -> float:
        """Get current state of charge (0-1)."""
        return self.state.soc
    
    def get_available_charge_power(self) -> float:
        """Get maximum power that can be used for charging right now (kW)."""
        available_capacity = (
            self.config.soc_max * self.state.capacity_kwh - 
            self.state.energy_kwh
        )
        # Assume 1-hour timestep for instantaneous calculation
        max_power_by_capacity = available_capacity / self.config.charge_efficiency
        return min(self.config.max_charge_rate_kw, max_power_by_capacity)
    
    def get_available_discharge_power(self) -> float:
        """Get maximum power that can be discharged right now (kW)."""
        available_energy = (
            self.state.energy_kwh - 
            self.config.soc_min * self.state.capacity_kwh
        )
        # Assume 1-hour timestep for instantaneous calculation
        max_power_by_energy = available_energy * self.config.discharge_efficiency
        return min(self.config.max_discharge_rate_kw, max_power_by_energy)
    
    def get_history_dataframe(self):
        """Get simulation history as a pandas DataFrame."""
        import pandas as pd
        
        data = {
            'timestep': range(len(self.history)),
            'action': [r.action.name for r in self.history],
            'power_requested_kw': [r.power_requested_kw for r in self.history],
            'power_actual_kw': [r.power_actual_kw for r in self.history],
            'energy_change_kwh': [r.energy_change_kwh for r in self.history],
            'soc_before': [r.soc_before for r in self.history],
            'soc_after': [r.soc_after for r in self.history],
            'grid_power_kw': [r.grid_power_kw for r in self.history],
            'pv_power_used_kw': [r.pv_power_used_kw for r in self.history],
            'losses_kwh': [r.losses_kwh for r in self.history],
            'constrained': [r.constrained for r in self.history],
        }
        return pd.DataFrame(data)


# Convenience function for quick testing
def create_battery(
    capacity_kwh: float = 100.0,
    max_power_kw: float = 50.0,
    efficiency: float = 0.90,
    soc_limits: Tuple[float, float] = (0.1, 0.9),
    initial_soc: float = 0.5,
    timestep_minutes: float = 5.0
) -> BatteryModel:
    """
    Create a battery model with common parameters.
    
    Args:
        capacity_kwh: Total energy capacity
        max_power_kw: Max charge/discharge power (symmetric)
        efficiency: Roundtrip efficiency
        soc_limits: (min_soc, max_soc) tuple
        initial_soc: Initial state of charge
        timestep_minutes: Simulation timestep in minutes (default: 5)
        
    Returns:
        Configured BatteryModel instance
    """
    config = BatteryConfig(
        timestep_hours=timestep_minutes / 60.0,
        capacity_kwh=capacity_kwh,
        max_charge_rate_kw=max_power_kw,
        max_discharge_rate_kw=max_power_kw,
        roundtrip_efficiency=efficiency,
        soc_min=soc_limits[0],
        soc_max=soc_limits[1],
        soc_initial=initial_soc
    )
    return BatteryModel(config)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Battery Model Example")
    print("=" * 60)
    
    # Create a 100 kWh battery with 50 kW max power, 10-minute timestep
    battery = create_battery(
        capacity_kwh=100,
        max_power_kw=50,
        efficiency=0.90,
        initial_soc=0.5,
        timestep_minutes=10  # Configurable timestep
    )
    
    print(f"\nInitial State:")
    print(f"  SOC: {battery.get_soc():.1%}")
    print(f"  Energy: {battery.state.energy_kwh:.1f} kWh")
    print(f"  Available charge power: {battery.get_available_charge_power():.1f} kW")
    print(f"  Available discharge power: {battery.get_available_discharge_power():.1f} kW")
    
    print(f"  Configured timestep: {battery.config.timestep_hours * 60:.0f} minutes")
    
    # Simulate charging from PV (uses configured timestep)
    print(f"\n--- Charging from PV (30 kW for one timestep) ---")
    result = battery.step(
        action=BatteryAction.CHARGE_FROM_PV,
        power_kw=30,
        pv_available_kw=40
        # timestep_hours not provided - uses config default (10 min)
    )
    print(f"  Power actual: {result.power_actual_kw:.1f} kW")
    print(f"  Energy stored: {result.energy_change_kwh:.1f} kWh")
    print(f"  SOC: {result.soc_before:.1%} -> {result.soc_after:.1%}")
    print(f"  Losses: {result.losses_kwh:.2f} kWh")
    
    # Simulate discharging to load
    print(f"\n--- Discharging to load (20 kW for 2 hours) ---")
    result = battery.step(
        action=BatteryAction.DISCHARGE_TO_LOAD,
        power_kw=20,
        load_demand_kw=25,
        timestep_hours=2.0
    )
    print(f"  Power actual: {result.power_actual_kw:.1f} kW")
    print(f"  Energy discharged: {result.energy_change_kwh:.1f} kWh")
    print(f"  SOC: {result.soc_before:.1%} -> {result.soc_after:.1%}")
    print(f"  Grid power offset: {result.grid_power_kw:.1f} kW")
    
    # Final state
    print(f"\nFinal State:")
    print(f"  SOC: {battery.get_soc():.1%}")
    print(f"  Total charged: {battery.state.total_energy_charged_kwh:.1f} kWh")
    print(f"  Total discharged: {battery.state.total_energy_discharged_kwh:.1f} kWh")
    print(f"  Equivalent cycles: {battery.state.cycles:.2f}")
