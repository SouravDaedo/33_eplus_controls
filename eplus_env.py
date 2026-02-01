"""
EnergyPlus Step Environment

A gym-like interface for EnergyPlus that allows step-by-step control.
External controllers can call step() to advance the simulation one timestep
and receive observations, then send actions back.

Usage:
    from eplus_env import EnergyPlusEnv
    
    env = EnergyPlusEnv(idf_path, epw_path)
    obs = env.reset()
    
    while not env.done:
        action = my_controller(obs)  # External control logic
        obs, reward, done, info = env.step(action)
    
    env.close()
"""

import os
import sys
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Any


class EnergyPlusEnv:
    """
    Step-based EnergyPlus simulation environment.
    
    Provides a gym-like interface where:
    - reset() initializes the simulation
    - step(action) advances one timestep and returns observations
    - External controller logic stays completely separate
    """
    
    def __init__(self, idf_path: str, epw_path: str, output_dir: str = "outputs/step_sim"):
        """
        Initialize the environment.
        
        Args:
            idf_path: Path to IDF model file
            epw_path: Path to weather file
            output_dir: Directory for simulation outputs
        """
        self.idf_path = os.path.abspath(idf_path)
        self.epw_path = os.path.abspath(epw_path)
        self.output_dir = os.path.abspath(output_dir)
        
        # Validate paths
        if not os.path.exists(self.idf_path):
            raise FileNotFoundError(f"IDF file not found: {self.idf_path}")
        if not os.path.exists(self.epw_path):
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")
        
        # EnergyPlus API
        self.api = None
        self.state = None
        
        # Threading for async simulation
        self.sim_thread = None
        self.action_queue = queue.Queue()
        self.obs_queue = queue.Queue()
        self.stop_requested = False
        
        # State
        self.done = False
        self.initialized = False
        self.timestep = 0
        
        # Handles (set during simulation)
        self.handles = {}
        self.actuator_handles = {}
        
        # Zone names for this model
        self.zone_names = [
            "Core_bottom", "Core_mid", "Core_top",
            "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
            "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
            "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4"
        ]
        
        # Current observations
        self.current_obs = {}
        
    def _init_handles(self):
        """Initialize data exchange handles."""
        if self.initialized:
            return
            
        exchange = self.api.exchange
        
        # Sensor handles
        self.handles['oat'] = exchange.get_variable_handle(
            self.state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        self.handles['power'] = exchange.get_meter_handle(
            self.state, "Electricity:Facility"
        )
        self.handles['hour'] = exchange.get_variable_handle(
            self.state, "Site Current Time", "Environment"
        )
        
        # Zone temperature handles
        for zone in self.zone_names:
            self.handles[f'temp_{zone}'] = exchange.get_variable_handle(
                self.state, "Zone Mean Air Temperature", zone
            )
        
        # Actuator handles for setpoints
        for zone in self.zone_names:
            cooling_handle = exchange.get_actuator_handle(
                self.state, "Zone Temperature Control", "Cooling Setpoint", zone
            )
            heating_handle = exchange.get_actuator_handle(
                self.state, "Zone Temperature Control", "Heating Setpoint", zone
            )
            if cooling_handle > 0 and heating_handle > 0:
                self.actuator_handles[zone] = {
                    'cooling': cooling_handle,
                    'heating': heating_handle
                }
        
        self.initialized = True
        print(f"Initialized {len(self.actuator_handles)} zone actuators")
        
    def _get_observations(self) -> Dict[str, Any]:
        """Get current simulation state as observations."""
        exchange = self.api.exchange
        
        # Get simulation datetime
        month = exchange.month(self.state)
        day = exchange.day_of_month(self.state)
        hour = exchange.hour(self.state)
        minute = exchange.minutes(self.state)
        year = exchange.year(self.state)
        
        obs = {
            'timestep': self.timestep,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'outdoor_temp': exchange.get_variable_value(self.state, self.handles['oat']),
            'total_power': exchange.get_meter_value(self.state, self.handles['power']),
            'zone_temps': {}
        }
        
        # Get zone temperatures
        for zone in self.zone_names:
            handle = self.handles.get(f'temp_{zone}')
            if handle and handle > 0:
                obs['zone_temps'][zone] = exchange.get_variable_value(self.state, handle)
        
        return obs
        
    def _apply_actions(self, actions: Dict[str, Any]):
        """Apply control actions to the simulation."""
        if not actions:
            return
            
        exchange = self.api.exchange
        
        # Apply zone setpoints
        for zone, setpoints in actions.get('zone_setpoints', {}).items():
            if zone in self.actuator_handles:
                handles = self.actuator_handles[zone]
                if 'cooling' in setpoints:
                    exchange.set_actuator_value(self.state, handles['cooling'], setpoints['cooling'])
                if 'heating' in setpoints:
                    exchange.set_actuator_value(self.state, handles['heating'], setpoints['heating'])
        
        # Apply uniform setpoints to all zones
        if 'cooling_setpoint' in actions:
            for zone, handles in self.actuator_handles.items():
                exchange.set_actuator_value(self.state, handles['cooling'], actions['cooling_setpoint'])
        if 'heating_setpoint' in actions:
            for zone, handles in self.actuator_handles.items():
                exchange.set_actuator_value(self.state, handles['heating'], actions['heating_setpoint'])
    
    def _timestep_callback(self, state):
        """Called at each simulation timestep."""
        # Check if stop requested
        if self.stop_requested:
            self.api.runtime.stop_simulation(state)
            return
            
        # Skip warmup
        if self.api.exchange.warmup_flag(state):
            return
        
        # Skip sizing periods (only process actual run period)
        # kind_of_sim: 1=sizing, 3=run period
        kind_of_sim = self.api.exchange.kind_of_sim(state)
        if kind_of_sim != 3:  # Not a run period
            return
            
        # Initialize handles on first real timestep
        if not self.initialized:
            self._init_handles()
        
        self.timestep += 1
        
        # Get observations and send to main thread
        obs = self._get_observations()
        self.obs_queue.put(('obs', obs))
        
        # Wait for action from main thread
        try:
            action = self.action_queue.get(timeout=60)
            if action == 'STOP':
                self.api.runtime.stop_simulation(state)
                return
            if action is not None:
                self._apply_actions(action)
        except queue.Empty:
            print("Warning: No action received, using defaults")
    
    def _run_simulation(self):
        """Run simulation in background thread."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        args = [
            '-w', self.epw_path,
            '-d', self.output_dir,
            self.idf_path
        ]
        
        exit_code = self.api.runtime.run_energyplus(self.state, args)
        
        # Signal completion
        self.obs_queue.put(('done', exit_code))
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset and start a new simulation.
        
        Returns:
            Initial observations
        """
        from pyenergyplus.api import EnergyPlusAPI
        
        # Clean up previous simulation
        if self.state is not None:
            self.close()
        
        # Initialize API
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        
        # Reset state
        self.done = False
        self.initialized = False
        self.timestep = 0
        self.stop_requested = False
        self.handles = {}
        self.actuator_handles = {}
        
        # Clear queues
        while not self.action_queue.empty():
            self.action_queue.get()
        while not self.obs_queue.empty():
            self.obs_queue.get()
        
        # Register callback
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state, self._timestep_callback
        )
        
        # Start simulation in background thread
        self.sim_thread = threading.Thread(target=self._run_simulation)
        self.sim_thread.start()
        
        # Wait for first observation
        msg_type, data = self.obs_queue.get()
        if msg_type == 'done':
            self.done = True
            return {}
        
        self.current_obs = data
        return data
    
    def step(self, action: Optional[Dict[str, Any]] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance simulation by one timestep.
        
        Args:
            action: Control actions to apply. Can include:
                - 'cooling_setpoint': float - uniform cooling setpoint for all zones
                - 'heating_setpoint': float - uniform heating setpoint for all zones
                - 'zone_setpoints': dict - per-zone setpoints
                
        Returns:
            obs: Current observations
            reward: Reward signal (placeholder, always 0)
            done: Whether simulation is complete
            info: Additional info
        """
        if self.done:
            return self.current_obs, 0.0, True, {}
        
        # Send action to simulation thread
        self.action_queue.put(action)
        
        # Wait for next observation
        try:
            msg_type, data = self.obs_queue.get(timeout=120)
        except queue.Empty:
            print("Error: Simulation timeout")
            self.done = True
            return self.current_obs, 0.0, True, {'error': 'timeout'}
        
        if msg_type == 'done':
            self.done = True
            return self.current_obs, 0.0, True, {'exit_code': data}
        
        self.current_obs = data
        
        # Simple reward: negative power consumption
        reward = -data.get('total_power', 0) / 1e6  # Scale down
        
        return data, reward, False, {}
    
    def close(self):
        """Clean up simulation resources."""
        self.stop_requested = True
        
        if self.sim_thread and self.sim_thread.is_alive():
            # Send stop signal to unblock and terminate gracefully
            self.action_queue.put('STOP')
            self.sim_thread.join(timeout=10)
        
        if self.state is not None:
            try:
                self.api.state_manager.delete_state(self.state)
            except:
                pass  # Ignore cleanup errors
            self.state = None
        
        self.done = True
        
    def get_observation_space(self) -> Dict[str, str]:
        """Return description of observation space."""
        return {
            'timestep': 'int - current simulation timestep',
            'outdoor_temp': 'float - outdoor air temperature (°C)',
            'total_power': 'float - total facility power (W)',
            'zone_temps': 'dict - zone name -> temperature (°C)'
        }
    
    def get_action_space(self) -> Dict[str, str]:
        """Return description of action space."""
        return {
            'cooling_setpoint': 'float - cooling setpoint for all zones (°C)',
            'heating_setpoint': 'float - heating setpoint for all zones (°C)',
            'zone_setpoints': 'dict - zone name -> {cooling: float, heating: float}'
        }


if __name__ == "__main__":
    # Simple test
    print("Testing EnergyPlus Step Environment")
    print("=" * 50)
    
    env = EnergyPlusEnv(
        idf_path="energyplus/control_models/MediumOffice_Control.idf",
        epw_path="weather/chicago/TMY_lat41.88_lon-87.63.epw",
        output_dir="outputs/step_test"
    )
    
    print("\nObservation space:", env.get_observation_space())
    print("\nAction space:", env.get_action_space())
    
    print("\nStarting simulation...")
    obs = env.reset()
    print(f"Initial obs: OAT={obs.get('outdoor_temp', 'N/A'):.1f}°C")
    
    step_count = 0
    max_steps = 100  # Just run 100 steps for testing
    
    while not env.done and step_count < max_steps:
        # Simple control: fixed setpoints
        action = {
            'cooling_setpoint': 24.0,
            'heating_setpoint': 21.0
        }
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: OAT={obs.get('outdoor_temp', 0):.1f}°C, "
                  f"Power={obs.get('total_power', 0)/1000:.1f}kW")
    
    env.close()
    print(f"\nCompleted {step_count} steps")
