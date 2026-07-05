"""
RL-based HVAC Control with Setpoint and Airflow Management

This script uses an existing RL agent (SAC) to control:
- Heating and cooling setpoints with deadband
- Airflow rates

State includes:
- Current zone temperature
- Current weather (dry bulb temp, cloud cover)
- Forecasted weather
- Time of day

Timestep: 5 minutes
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from pyenergyplus.api import EnergyPlusAPI
except ImportError:
    print("Warning: EnergyPlus API not found. Install with: pip install pyenergyplus-lbnl")
    sys.exit(1)

from src.agents.sac_agent import SACAgent
from src.utils.hvac_config import get_hvac_config
from src.utils.energy_price import get_realtime_price
from src.utils.outdoor_co2_schedule import load_outdoor_co2_csv
from src.utils.idf_modifier import (
    create_custom_idf,
    calculate_simulation_days,
    get_season_info,
)


CONTROLLED_ZONE_NAMES = [
    "Core_bottom", "Core_mid", "Core_top",
    "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
    "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4",
]

FLOOR_ZONE_GROUPS = {
    "bottom": [
        "Core_bottom",
        "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
    ],
    "mid": [
        "Core_mid",
        "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
    ],
    "top": [
        "Core_top",
        "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4",
    ],
}


def _in_training_window(month, day, hour, window):
    """True if (month, day, hour) is inside the training window [start_date+start_hour, end_date+end_hour]."""
    if not window:
        return True
    sm, sd, sh = window.get('start_month'), window.get('start_day'), window.get('start_hour', 0)
    em, ed, eh = window.get('end_month'), window.get('end_day'), window.get('end_hour', 24)
    if (month, day) < (sm, sd):
        return False
    if (month, day) > (em, ed):
        return False
    if (month, day) == (sm, sd) and hour < sh:
        return False
    if (month, day) == (em, ed) and eh < 24 and hour >= eh:
        return False
    return True


def _sample_random_start_in_window(window, after_month=None, after_day=None, after_hour=None, year=2023):
    """
    Sample a random (month, day, hour) within the training window.
    If after_* is given, sample from [after, end_of_window] (for next episode start).
    Returns (month, day, hour) with hour in 0--23.
    """
    from datetime import datetime, timedelta
    sm = window.get('start_month', 1)
    sd = window.get('start_day', 1)
    sh = window.get('start_hour', 0)
    em = window.get('end_month', 12)
    ed = window.get('end_day', 31)
    eh = window.get('end_hour', 24)
    start_dt = datetime(year, sm, sd, min(sh, 23), 0, 0)
    end_dt = datetime(year, em, ed, 23, 59, 59) if eh >= 24 else datetime(year, em, ed, min(eh, 23), 0, 0)
    if after_month is not None and after_day is not None and after_hour is not None:
        after_dt = datetime(year, after_month, after_day, min(after_hour, 23), 0, 0)
        start_dt = max(start_dt, after_dt)
    if start_dt >= end_dt:
        return (em, ed, min(23, eh - 1) if eh < 24 else 23)
    delta_hours = (end_dt - start_dt).total_seconds() / 3600.0
    r = np.random.uniform(0, max(0.001, delta_hours))
    from_dt = start_dt + timedelta(hours=r)
    return (from_dt.month, from_dt.day, from_dt.hour)


class LiveRLPlotter:
    """Small matplotlib dashboard updated from the EnergyPlus callback."""

    def __init__(self, output_dir, update_every=1, episode_scope="current"):
        self.update_every = max(1, int(update_every))
        if episode_scope not in {"current", "all"}:
            raise ValueError("episode_scope must be 'current' or 'all'")
        self.episode_scope = episode_scope
        self.current_episode = None
        self.step_count = 0
        mpl_cache_dir = Path(output_dir) / ".matplotlib-cache"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Live plotting requires matplotlib. Install it in the active environment."
            ) from exc

        self.plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 3, figsize=(16, 10), num="RL HVAC Live Plot")
        self.fig.suptitle("RL HVAC Simulation Live Plot")

        self.series = {
            "step": [],
            "avg_zone_temp": [],
            "bottom_floor_temp": [],
            "mid_floor_temp": [],
            "top_floor_temp": [],
            "outdoor_temp": [],
            "electric_kw": [],
            "gas_kw": [],
            "reward": [],
            "avg_co2": [],
            "outdoor_co2": [],
            "heating_setpoint": [],
            "cooling_setpoint": [],
            "bottom_floor_temp_heating_setpoint": [],
            "bottom_floor_temp_cooling_setpoint": [],
            "mid_floor_temp_heating_setpoint": [],
            "mid_floor_temp_cooling_setpoint": [],
            "top_floor_temp_heating_setpoint": [],
            "top_floor_temp_cooling_setpoint": [],
            "airflow": [],
            "bottom_floor_airflow": [],
            "mid_floor_airflow": [],
            "top_floor_airflow": [],
            "people": [],
            "avg_ppd": [],
            "energy_cost": [],
            "gas_cost": [],
            "comfort_penalty": [],
            "setpoint_penalty": [],
            "demand_penalty": [],
            "total_cost": [],
        }

        self.lines = {}
        self.fills = {}
        self.plot_axes = []
        self._setup_axes()

    def _empty_series(self):
        return {
            "step": [],
            "avg_zone_temp": [],
            "bottom_floor_temp": [],
            "mid_floor_temp": [],
            "top_floor_temp": [],
            "outdoor_temp": [],
            "electric_kw": [],
            "gas_kw": [],
            "reward": [],
            "avg_co2": [],
            "outdoor_co2": [],
            "heating_setpoint": [],
            "cooling_setpoint": [],
            "bottom_floor_temp_heating_setpoint": [],
            "bottom_floor_temp_cooling_setpoint": [],
            "mid_floor_temp_heating_setpoint": [],
            "mid_floor_temp_cooling_setpoint": [],
            "top_floor_temp_heating_setpoint": [],
            "top_floor_temp_cooling_setpoint": [],
            "airflow": [],
            "bottom_floor_airflow": [],
            "mid_floor_airflow": [],
            "top_floor_airflow": [],
            "people": [],
            "avg_ppd": [],
            "energy_cost": [],
            "gas_cost": [],
            "comfort_penalty": [],
            "setpoint_penalty": [],
            "demand_penalty": [],
            "total_cost": [],
        }

    def _reset_for_episode(self, episode):
        self.series = self._empty_series()
        self.current_episode = episode
        self.step_count = 0
        for line in self.lines.values():
            line.set_data([], [])
        for fill in self.fills.values():
            fill.remove()
        self.fills = {}
        self.fig.suptitle(f"RL HVAC Simulation Live Plot - Episode {episode}")
        for ax in self.plot_axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def _setup_floor_temp_axis(self, ax, series_name, label, color):
        self.lines[series_name], = ax.plot([], [], label=label, color=color, linewidth=3.0)
        heat_key = f"{series_name}_heating_setpoint"
        cool_key = f"{series_name}_cooling_setpoint"
        self.lines[heat_key], = ax.plot(
            [], [], label="Heating Setpoint", color="indianred", linestyle="--", drawstyle="steps-post", alpha=0.45
        )
        self.lines[cool_key], = ax.plot(
            [], [], label="Cooling Setpoint", color="steelblue", linestyle="--", drawstyle="steps-post", alpha=0.45
        )
        self._set_panel_title(ax, label)
        self._add_inside_ylabel(ax, "Temperature (C)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self._style_axis(ax)

    def _set_panel_title(self, ax, label):
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)

    def _add_inside_ylabel(self, ax, label, side="left"):
        x = 0.02 if side == "left" else 0.98
        ha = "left" if side == "left" else "right"
        ax.set_ylabel("")
        ax.text(
            x, 0.5, label,
            transform=ax.transAxes,
            rotation=90,
            ha=ha,
            va="center",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "boxstyle": "round,pad=0.2",
            },
            zorder=10,
        )

    def _style_axis(self, ax):
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)

    def _setup_airflow_axis(self, ax, series_name, label, show_title=False, show_xticks=False):
        self.lines[series_name], = ax.plot([], [], label=label, color="tab:blue", linewidth=1.8)
        if show_title:
            self._set_panel_title(ax, "Avg Airflow")
        self._add_inside_ylabel(ax, "Airflow (m3/s)")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)
        if not show_xticks:
            ax.tick_params(axis="x", labelbottom=False)
        self._style_axis(ax)

    def _setup_axes(self):
        (
            ax_bottom_temp, ax_mid_temp, ax_top_temp,
            ax_airflow_panel, ax_people, ax_co2,
            ax_electric, ax_kpi_cost, ax_reward,
        ) = self.axes.ravel()

        self._setup_floor_temp_axis(ax_bottom_temp, "bottom_floor_temp", "Bottom Floor Avg Temp", "teal")
        self._setup_floor_temp_axis(ax_mid_temp, "mid_floor_temp", "Mid Floor Avg Temp", "teal")
        self._setup_floor_temp_axis(ax_top_temp, "top_floor_temp", "Top Floor Avg Temp", "teal")

        airflow_grid = ax_airflow_panel.get_subplotspec().subgridspec(3, 1, hspace=0.18)
        ax_airflow_panel.remove()
        ax_bottom_airflow = self.fig.add_subplot(airflow_grid[0])
        ax_mid_airflow = self.fig.add_subplot(airflow_grid[1])
        ax_top_airflow = self.fig.add_subplot(airflow_grid[2])
        self._setup_airflow_axis(ax_bottom_airflow, "bottom_floor_airflow", "Bottom Floor", show_title=True)
        self._setup_airflow_axis(ax_mid_airflow, "mid_floor_airflow", "Mid Floor")
        self._setup_airflow_axis(ax_top_airflow, "top_floor_airflow", "Top Floor", show_xticks=True)
        ax_top_airflow.set_xlabel("Timestep")
        self.plot_axes = [
            ax_bottom_temp, ax_mid_temp, ax_top_temp,
            ax_bottom_airflow, ax_mid_airflow, ax_top_airflow,
            ax_people, ax_co2, ax_electric, ax_kpi_cost, ax_reward,
        ]

        self.lines["people"], = ax_people.plot([], [], label="People", color="tab:olive", linewidth=2.0)
        self._set_panel_title(ax_people, "People and PPD")
        self._add_inside_ylabel(ax_people, "People")
        ax_ppd = ax_people.twinx()
        self.lines["avg_ppd"], = ax_ppd.plot([], [], label="Avg PPD", color="darkgoldenrod", linewidth=2.0)
        self._add_inside_ylabel(ax_ppd, "PPD (%)", side="right")
        people_lines = [self.lines["people"], self.lines["avg_ppd"]]
        ax_people.legend(people_lines, [line.get_label() for line in people_lines], loc="lower right", fontsize=8)
        ax_people.grid(True, alpha=0.3)
        self._style_axis(ax_people)
        self._style_axis(ax_ppd)
        self.plot_axes.append(ax_ppd)

        self.lines["avg_co2"], = ax_co2.plot([], [], label="Avg Zone CO2", color="tab:cyan")
        self.lines["outdoor_co2"], = ax_co2.plot(
            [], [], label="Outdoor CO2", color="tab:green", linestyle="--"
        )
        ax_co2.set_xlabel("Timestep")
        self._set_panel_title(ax_co2, "CO2 and Outdoor Temp")
        self._add_inside_ylabel(ax_co2, "CO2 (ppm)")
        ax_oat = ax_co2.twinx()
        self.lines["outdoor_temp"], = ax_oat.plot([], [], label="Outdoor Temp", color="tab:orange")
        self._add_inside_ylabel(ax_oat, "Outdoor Temp (C)", side="right")
        co2_lines = [
            self.lines["avg_co2"],
            self.lines["outdoor_co2"],
            self.lines["outdoor_temp"],
        ]
        ax_co2.legend(co2_lines, [line.get_label() for line in co2_lines], loc="lower right", fontsize=8)
        ax_co2.grid(True, alpha=0.3)
        self._style_axis(ax_co2)
        self._style_axis(ax_oat)
        self.plot_axes.append(ax_oat)

        self.lines["electric_kw"], = ax_electric.plot([], [], label="Electric", color="tab:red", alpha=0.85)
        self.lines["gas_kw"], = ax_electric.plot([], [], label="Gas", color="tab:green", alpha=0.85)
        ax_electric.set_xlabel("Timestep")
        self._set_panel_title(ax_electric, "Electric and Gas Power")
        self._add_inside_ylabel(ax_electric, "Power (kW)")
        ax_electric.legend(loc="lower right", fontsize=8)
        ax_electric.grid(True, alpha=0.3)
        self._style_axis(ax_electric)

        kpi_specs = [
            ("energy_cost", "Electric", "tab:red"),
            ("gas_cost", "Gas", "tab:green"),
            ("comfort_penalty", "Comfort", "darkgoldenrod"),
            ("setpoint_penalty", "Setpoint", "steelblue"),
            ("demand_penalty", "Demand", "tab:orange"),
            ("total_cost", "Total", "black"),
        ]
        for name, label, color in kpi_specs:
            linewidth = 2.0 if name == "total_cost" else 1.4
            alpha = 0.9 if name == "total_cost" else 0.75
            self.lines[name], = ax_kpi_cost.plot(
                [], [], label=label, color=color, linewidth=linewidth, alpha=alpha
            )
        ax_kpi_cost.set_xlabel("Timestep")
        self._set_panel_title(ax_kpi_cost, "Reward KPI Costs")
        self._add_inside_ylabel(ax_kpi_cost, "Cost / Penalty")
        ax_kpi_cost.legend(loc="lower right", fontsize=7, ncol=2)
        ax_kpi_cost.grid(True, alpha=0.3)
        self._style_axis(ax_kpi_cost)

        self.lines["reward"], = ax_reward.plot([], [], label="Step Reward", color="tab:purple")
        ax_reward.set_xlabel("Timestep")
        self._set_panel_title(ax_reward, "Step Reward")
        self._add_inside_ylabel(ax_reward, "Step Reward")
        ax_reward.legend(loc="lower right", fontsize=8)
        ax_reward.grid(True, alpha=0.3)
        self._style_axis(ax_reward)

        self.fig.tight_layout(pad=0.6, w_pad=0.35, h_pad=0.45)
        self.fig.subplots_adjust(wspace=0.18, hspace=0.28)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update(self, log_entry):
        episode = int(log_entry["episode"])
        if self.current_episode is None:
            self.current_episode = episode
            self.fig.suptitle(
                "RL HVAC Simulation Live Plot"
                if self.episode_scope == "all"
                else f"RL HVAC Simulation Live Plot - Episode {episode}"
            )
        elif self.episode_scope == "current" and episode != self.current_episode:
            self._reset_for_episode(episode)

        self.step_count += 1

        step = int(log_entry["timestep"]) if self.episode_scope == "current" else len(self.series["step"]) + 1
        self.series["step"].append(step)
        self.series["avg_zone_temp"].append(float(log_entry["avg_zone_temp"]))
        self.series["bottom_floor_temp"].append(self._avg_log_values(
            log_entry,
            [f"temp_{zone}" for zone in FLOOR_ZONE_GROUPS["bottom"]]
        ))
        self.series["mid_floor_temp"].append(self._avg_log_values(
            log_entry,
            [f"temp_{zone}" for zone in FLOOR_ZONE_GROUPS["mid"]]
        ))
        self.series["top_floor_temp"].append(self._avg_log_values(
            log_entry,
            [f"temp_{zone}" for zone in FLOOR_ZONE_GROUPS["top"]]
        ))
        self.series["outdoor_temp"].append(float(log_entry["outdoor_temp"]))
        self.series["electric_kw"].append(float(log_entry["current_power"]) / 1000.0)
        self.series["gas_kw"].append(float(log_entry["current_gas_power"]) / 1000.0)
        self.series["reward"].append(float(log_entry["reward"]))
        self.series["heating_setpoint"].append(float(log_entry["heating_setpoint"]))
        self.series["cooling_setpoint"].append(float(log_entry["cooling_setpoint"]))
        for floor in ("bottom_floor_temp", "mid_floor_temp", "top_floor_temp"):
            self.series[f"{floor}_heating_setpoint"].append(float(log_entry["heating_setpoint"]))
            self.series[f"{floor}_cooling_setpoint"].append(float(log_entry["cooling_setpoint"]))
        self.series["airflow"].append(float(log_entry["airflow"]))
        self.series["bottom_floor_airflow"].append(float(log_entry.get("bottom_floor_airflow", log_entry["airflow"])))
        self.series["mid_floor_airflow"].append(float(log_entry.get("mid_floor_airflow", log_entry["airflow"])))
        self.series["top_floor_airflow"].append(float(log_entry.get("top_floor_airflow", log_entry["airflow"])))
        self.series["outdoor_co2"].append(float(log_entry["outdoor_co2"]))
        self.series["people"].append(float(log_entry.get("people", np.nan)))
        self.series["energy_cost"].append(float(log_entry["energy_cost"]))
        self.series["gas_cost"].append(float(log_entry["gas_cost"]))
        self.series["comfort_penalty"].append(float(log_entry["comfort_penalty"]))
        self.series["setpoint_penalty"].append(float(log_entry["setpoint_penalty"]))
        self.series["demand_penalty"].append(float(log_entry["demand_penalty"]))
        self.series["total_cost"].append(float(log_entry["total_cost"]))

        co2_values = [
            float(value)
            for key, value in log_entry.items()
            if key.startswith("co2_") and not np.isnan(value)
        ]
        if not co2_values:
            raise RuntimeError("Live plot requested average zone CO2, but no zone CO2 values were logged")
        self.series["avg_co2"].append(float(np.mean(co2_values)))

        ppd_values = [
            float(value)
            for key, value in log_entry.items()
            if key.startswith("ppd_") and not np.isnan(value)
        ]
        self.series["avg_ppd"].append(float(np.mean(ppd_values)) if ppd_values else np.nan)

        if self.step_count % self.update_every != 0:
            return

        x = self.series["step"]
        for name, line in self.lines.items():
            line.set_data(x, self.series[name])

        for fill in self.fills.values():
            fill.remove()
        self.fills["electric_kw"] = self.axes[2, 0].fill_between(
            x, self.series["electric_kw"], color="tab:red", alpha=0.10
        )
        self.fills["gas_kw"] = self.axes[2, 0].fill_between(
            x, self.series["gas_kw"], color="tab:green", alpha=0.08
        )
        self.fills["reward"] = self.axes[2, 2].fill_between(
            x, self.series["reward"], color="tab:purple", alpha=0.12
        )
        self.fills["energy_cost"] = self.axes[2, 1].fill_between(
            x, self.series["energy_cost"], color="tab:red", alpha=0.08
        )
        self.fills["demand_penalty"] = self.axes[2, 1].fill_between(
            x, self.series["demand_penalty"], color="tab:orange", alpha=0.08
        )

        for ax in self.plot_axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _avg_log_values(self, log_entry, keys):
        values = [
            float(log_entry[key])
            for key in keys
            if key in log_entry and not np.isnan(log_entry[key])
        ]
        return float(np.mean(values)) if values else np.nan

    def finish(self, output_dir, hold=False):
        path = Path(output_dir) / "rl_hvac_live_plot.png"
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved live plot snapshot to {path}")
        if hold:
            print("Close the plot window to finish.")
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class LiveLossPlotter:
    """Small matplotlib dashboard for SAC training losses."""

    def __init__(self, output_dir, update_every=1):
        self.update_every = max(1, int(update_every))
        self.step_count = 0
        mpl_cache_dir = Path(output_dir) / ".matplotlib-cache"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Live loss plotting requires matplotlib. Install it in the active environment."
            ) from exc

        self.plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 7), num="SAC Training Losses")
        self.fig.suptitle("SAC Training Losses")
        self.series = {
            "update": [],
            "actor_loss": [],
            "critic1_loss": [],
            "critic2_loss": [],
            "alpha_loss": [],
        }
        self.lines = {}
        self._setup_axes()

    def _setup_axis(self, ax, series_name, label, color):
        self.lines[series_name], = ax.plot([], [], label=label, color=color, linewidth=1.8)
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Training Update", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _setup_axes(self):
        ax_actor, ax_critic1, ax_critic2, ax_alpha = self.axes.ravel()
        self._setup_axis(ax_actor, "actor_loss", "Actor Loss", "tab:purple")
        self._setup_axis(ax_critic1, "critic1_loss", "Critic 1 Loss", "tab:red")
        self._setup_axis(ax_critic2, "critic2_loss", "Critic 2 Loss", "tab:orange")
        self._setup_axis(ax_alpha, "alpha_loss", "Alpha Loss", "tab:blue")
        self.fig.tight_layout(pad=0.8, w_pad=0.5, h_pad=0.7)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update(self, update_no, losses):
        self.step_count += 1
        self.series["update"].append(int(update_no))
        self.series["actor_loss"].append(float(losses["actor_loss"]))
        self.series["critic1_loss"].append(float(losses["critic1_loss"]))
        self.series["critic2_loss"].append(float(losses["critic2_loss"]))
        alpha_loss = losses.get("alpha_loss")
        self.series["alpha_loss"].append(float(alpha_loss) if alpha_loss is not None else np.nan)

        if self.step_count % self.update_every != 0:
            return

        x = self.series["update"]
        for name, line in self.lines.items():
            line.set_data(x, self.series[name])
        for ax in self.axes.ravel():
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def finish(self, output_dir, hold=False):
        path = Path(output_dir) / "sac_training_losses.png"
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SAC loss plot snapshot to {path}")
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class HVACEnvironment:
    """Environment wrapper for EnergyPlus HVAC control with RL agent."""
    
    def __init__(self, api, state, config_path=None):
        self.api = api
        self.state = state
        
        # Load HVAC configuration
        self.hvac_config = get_hvac_config(config_path)
        
        # Control parameters from config
        self.base_temp = self.hvac_config.base_temperature
        self.deadband_min = self.hvac_config.deadband_min
        self.deadband_max = self.hvac_config.deadband_max
        self.min_airflow = self.hvac_config.min_airflow
        self.max_airflow = self.hvac_config.max_airflow
        
        # State and action dimensions from config
        self.state_size = self.hvac_config.get_state_size()
        self.action_size = self.hvac_config.get_action_size()
        self.timesteps_per_hour = self.hvac_config.get_timesteps_per_hour()
        self.episode_timesteps = self.hvac_config.get_episode_timesteps()
        
        # Current state tracking
        self.current_state = None
        self.current_action = None
        self.prev_temp = None
        self.prev_energy = None

        # Cached power (W) from callback_after_predictor_after_hvac_managers
        # (electricity and gas demand rates reset before zone timestep callback fires)
        self._cached_power_w = None
        self._cached_gas_power_w = None

        # Programmatic overrides — set these from outside before each timestep to inject custom data.
        # Applied via pre_timestep_callback (before zone heat-balance calcs) each step.
        #
        # outdoor_co2_override: float | None   — ppm; takes priority over CSV lookup and config static value
        # weather_override: dict | None         — keys: dry_bulb[°C], dew_point[°C], humidity[%],
        #                                          wind_speed[m/s], wind_dir[deg], sky_temp[°C],
        #                                          beam_solar[W/m²], diffuse_solar[W/m²]
        #                                          Only override the fields you include.
        self.outdoor_co2_override = None
        self.weather_override = None

        # Handles (initialized during warmup)
        self.handles_initialized = False
        self.handles = {}
        
        # Weather forecast: true forward-looking values, read directly from the EPW
        # file via EnergyPlus's today/tomorrow weather API (see _weather_value_at_offset).
        # Per-variable offsets (timesteps ahead) — e.g. temperature can look further
        # ahead than humidity/cloud cover.
        self.weather_forecast_offsets = self.hvac_config.weather_forecast_offsets

        # Optional past weather (lagged history) — disabled unless state_space.weather_history.enabled
        self.weather_history = []
        self.weather_history_enabled = self.hvac_config.weather_history_enabled
        self.weather_history_horizon = self.hvac_config.weather_history_horizon
        
        # Episode tracking
        self.timestep_count = 0
        self.episode_reward = 0.0
        self.energy_consumption = []
        # Variable episode length: set by controller via start_episode(); None = use config default
        self._episode_duration_timesteps = None
        
        # Outdoor CO2 from CSV (optional): override at each timestep
        sim_cfg = self.hvac_config.config.get('simulation', {})
        csv_path = sim_cfg.get('outdoor_co2_csv_path')
        if csv_path:
            try:
                fallback_ppm = sim_cfg.get('outdoor_co2_fallback_ppm')
                if fallback_ppm is None:
                    raise ValueError(
                        "simulation.outdoor_co2_fallback_ppm is required when "
                        "simulation.outdoor_co2_csv_path is set"
                    )
                self.outdoor_co2_lookup = load_outdoor_co2_csv(csv_path, fallback_ppm=fallback_ppm)
            except Exception as e:
                raise RuntimeError(f"Could not load outdoor CO2 CSV '{csv_path}': {e}") from e
        else:
            if sim_cfg.get('outdoor_co2_ppm') is None:
                raise ValueError(
                    "simulation.outdoor_co2_ppm is required when "
                    "simulation.outdoor_co2_csv_path is not set"
                )
            self.outdoor_co2_lookup = None

    def _required_variable_handle(self, variable_name, keys):
        """Return a variable handle for one of the allowed keys, or raise."""
        for key in keys:
            handle = self.api.exchange.get_variable_handle(self.state, variable_name, key)
            if handle > 0:
                return handle
        key_list = ", ".join(repr(k) for k in keys)
        raise RuntimeError(
            f"EnergyPlus handle invalid for '{variable_name}' with keys [{key_list}]. "
            "Declare the variable as Output:Variable and check the key in eplusout.rdd."
        )
        
    def start_episode(self, duration_timesteps):
        """Set duration for the current episode (timesteps). Used when episode_duration_hours is a range."""
        self._episode_duration_timesteps = int(duration_timesteps)
        
    # State size is now calculated from config using self.hvac_config.get_state_size()
    
    def initialize_handles(self):
        """Initialize EnergyPlus data exchange handles."""
        if self.handles_initialized:
            return True
            
        exchange = self.api.exchange
        if not exchange.api_data_fully_ready(self.state):
            return False
        
        # Weather handles
        # Weather values are read from the EnergyPlus weather API in
        # get_current_state(). Site weather Output:Variable handles are not
        # stable across EnergyPlus versions/keys.

        # Zone handles for all occupied/conditioned zones. The RL agent still
        # observes the configured subset, but actions are applied building-wide.
        zone_names = CONTROLLED_ZONE_NAMES

        self.handles['zones'] = {}
        for zone_name in zone_names:
            zone_handles = {}
            zone_handles['temp'] = exchange.get_variable_handle(
                self.state, "Zone Mean Air Temperature", zone_name
            )
            zone_handles['cooling_sp'] = exchange.get_actuator_handle(
                self.state, "Zone Temperature Control", "Cooling Setpoint", zone_name
            )
            zone_handles['heating_sp'] = exchange.get_actuator_handle(
                self.state, "Zone Temperature Control", "Heating Setpoint", zone_name
            )
            zone_handles['airflow'] = exchange.get_actuator_handle(
                self.state, "Zone Infiltration", "Air Exchange Flow Rate",
                f"{zone_name} Infiltration"
            )
            zone_handles['co2'] = exchange.get_variable_handle(
                self.state, "Zone Air CO2 Concentration", zone_name
            )
            zone_handles['people'] = exchange.get_variable_handle(
                self.state, "Zone People Occupant Count", zone_name
            )
            # Thermal comfort (Fanger) – PPD and PMV; optional, logged as NaN if unavailable
            zone_handles['ppd'] = exchange.get_variable_handle(
                self.state, "Zone Thermal Comfort Fanger Model PPD", zone_name
            )
            zone_handles['pmv'] = exchange.get_variable_handle(
                self.state, "Zone Thermal Comfort Fanger Model PMV", zone_name
            )

            critical = [f"{zone_name}.{k}" for k, h in zone_handles.items()
                        if k in ('temp', 'cooling_sp', 'heating_sp', 'co2') and h <= 0]
            if critical:
                raise RuntimeError(f"EnergyPlus handle(s) invalid for zone '{zone_name}': {critical}")

            self.handles['zones'][zone_name] = zone_handles

        # Energy consumption: "Facility Total Electricity Demand Rate" [W], key "*"
        # This variable must be declared in IDF as Output:Variable (injected by create_custom_idf).
        # Note: EnergyPlus 23.2+ uses this name (not the old "Facility Total Electric Demand Power").
        self.handles['total_power'] = exchange.get_variable_handle(
            self.state, "Facility Total Electricity Demand Rate", "Whole Building"
        )
        if self.handles['total_power'] <= 0:
            raise RuntimeError("EnergyPlus handle invalid: 'Facility Total Electricity Demand Rate' — check Output:Variable in IDF")

        # Natural gas is exposed by this 25.2 model as a facility meter [J per timestep].
        self.handles['gas_meter'] = exchange.get_meter_handle(self.state, "NaturalGas:Facility")
        if self.handles['gas_meter'] <= 0:
            raise RuntimeError("EnergyPlus meter handle invalid: 'NaturalGas:Facility' — check Output:Meter in IDF")

        # Outdoor CO2 schedule actuator — set_actuator_value overrides the schedule each step
        self.handles['outdoor_co2'] = exchange.get_actuator_handle(
            self.state, "Schedule:Constant", "Schedule Value", "Outdoor CO2 Schedule"
        )

        # Weather Data actuators — override EPW values each step via apply_weather_override().
        # All use component type "Weather Data", key "Environment".
        _weather_ctrl_types = {
            'dry_bulb':      'Outdoor Dry Bulb',
            'dew_point':     'Outdoor Dew Point',
            'humidity':      'Outdoor Relative Humidity',
            'wind_speed':    'Wind Speed',
            'wind_dir':      'Wind Direction',
            'sky_temp':      'Sky Temperature',
            'beam_solar':    'Beam Solar',
            'diffuse_solar': 'Diffuse Solar',
        }
        self.handles['weather'] = {}
        for key, ctrl_type in _weather_ctrl_types.items():
            self.handles['weather'][key] = exchange.get_actuator_handle(
                self.state, "Weather Data", ctrl_type, "Environment"
            )
        
        self.handles_initialized = True
        return True
    
    def apply_outdoor_co2_for_current_timestep(self):
        """Set outdoor CO2 schedule actuator for this timestep.

        Priority: outdoor_co2_override (programmatic) > CSV lookup > config value.
        Set env.outdoor_co2_override = <ppm> at any time to inject a custom value every step.
        """
        if not self.handles_initialized:
            return
        h = self.handles.get('outdoor_co2', -1)
        if h is None or h <= 0:
            return
        month = self.api.exchange.month(self.state)
        day = self.api.exchange.day_of_month(self.state)
        hour = self.api.exchange.hour(self.state)
        ppm = self.get_outdoor_co2_ppm(month, day, hour)
        self.api.exchange.set_actuator_value(self.state, h, ppm)

    def get_outdoor_co2_ppm(self, month, day, hour):
        """Return the outdoor CO2 value used by the controller for this timestep."""
        if self.outdoor_co2_override is not None:
            return float(self.outdoor_co2_override)
        if self.outdoor_co2_lookup:
            return float(self.outdoor_co2_lookup.get_ppm(month, day, hour))
        outdoor_co2 = self.hvac_config.config.get('simulation', {}).get('outdoor_co2_ppm')
        if outdoor_co2 is None:
            raise RuntimeError("Missing required simulation.outdoor_co2_ppm")
        return float(outdoor_co2)

    def apply_weather_override(self):
        """Override EPW weather values for this timestep from env.weather_override dict.

        Set env.weather_override = {'dry_bulb': 25.0, 'humidity': 60.0, ...} to inject custom
        weather each step. Only the keys you include are overridden; unset keys keep EPW values.

        Keys and units:
            dry_bulb      [°C]    outdoor dry-bulb temperature
            dew_point     [°C]    outdoor dew-point temperature
            humidity      [%]     outdoor relative humidity (0–100)
            wind_speed    [m/s]   wind speed
            wind_dir      [deg]   wind direction (0–360)
            sky_temp      [°C]    sky temperature
            beam_solar    [W/m²]  direct normal solar radiation
            diffuse_solar [W/m²]  diffuse horizontal solar radiation
        """
        if not self.handles_initialized or not self.weather_override:
            return
        exchange = self.api.exchange
        weather_handles = self.handles.get('weather', {})
        for key, value in self.weather_override.items():
            h = weather_handles.get(key, -1)
            if h and h > 0:
                exchange.set_actuator_value(self.state, h, float(value))
    
    _WEATHER_GETTERS = {
        'oat': ('outdoor_dry_bulb', ),
        'humidity': ('outdoor_relative_humidity', ),
        'cloud_cover': ('sky_temperature', ),  # cloud_cover feature is backed by sky temperature
    }

    def _weather_value_at_offset(self, base_hour, base_ts, offset_steps, field):
        """Return one weather value `offset_steps` zone timesteps ahead of
        (base_hour, base_ts), read directly from the EPW file via EnergyPlus's
        today/tomorrow weather API. This is a genuine forward look-ahead (perfect
        forecast), not a repeat of past/current readings.

        field: 'oat' | 'humidity' | 'cloud_cover'
        """
        exchange = self.api.exchange
        tsph = self.timesteps_per_hour
        total_index = base_hour * tsph + (base_ts - 1) + offset_steps
        day_offset, rem = divmod(total_index, tsph * 24)
        hour = rem // tsph
        ts = (rem % tsph) + 1

        if day_offset == 0:
            prefix = 'today'
        elif day_offset == 1:
            prefix = 'tomorrow'
        else:
            raise RuntimeError(
                f"Forecast offset of {offset_steps} timesteps is more than a day ahead; "
                "EnergyPlus only exposes today/tomorrow weather via this API."
            )
        suffix, = self._WEATHER_GETTERS[field]
        getter = getattr(exchange, f"{prefix}_weather_{suffix}_at_time")
        return getter(self.state, hour, ts)

    def get_current_state(self):
        """Get current environment state from EnergyPlus."""
        exchange = self.api.exchange
        
        # Get zone temperatures from config
        zone_count = self.hvac_config.config['state_space']['zone_temps']['count']
        zone_temps = []
        
        # Get actual zone temperatures from EnergyPlus
        zone_names = list(self.handles['zones'].keys())[:zone_count]
        for zone_name in zone_names:
            temp = exchange.get_variable_value(self.state, self.handles['zones'][zone_name]['temp'])
            zone_temps.append(temp)
        zone_temps = zone_temps[:zone_count]

        # Current weather from EnergyPlus weather API (actual EPW/override values, no fallback)
        current_hour_for_weather = exchange.hour(self.state)
        current_ts_for_weather = exchange.zone_time_step_number(self.state)
        oat = exchange.today_weather_outdoor_dry_bulb_at_time(
            self.state, current_hour_for_weather, current_ts_for_weather
        )
        humidity = exchange.today_weather_outdoor_relative_humidity_at_time(
            self.state, current_hour_for_weather, current_ts_for_weather
        )
        cloud_cover = exchange.today_weather_sky_temperature_at_time(
            self.state, current_hour_for_weather, current_ts_for_weather
        )
        
        current_weather = [oat, humidity, cloud_cover]

        # True forward-looking forecast: read ahead into the EPW file via EnergyPlus's
        # today/tomorrow weather API. Grouped per variable (all oat offsets, then all
        # humidity offsets, then all cloud_cover offsets) since each variable can have
        # its own independent list of timesteps-ahead. Gaussian noise (std growing
        # linearly with lead time) is added when state_space.weather_forecast.noise.enabled
        # is true, so the agent trains against realistic forecast error, not perfect foresight.
        noise_cfg = self.hvac_config.weather_forecast_noise
        forecast_weather = []
        for field in ('oat', 'humidity', 'cloud_cover'):
            for offset in self.weather_forecast_offsets[field]:
                value = self._weather_value_at_offset(current_hour_for_weather, current_ts_for_weather, offset, field)
                if noise_cfg['enabled'] and noise_cfg[field] > 0:
                    value += np.random.normal(0.0, noise_cfg[field] * offset)
                    if field == 'humidity':
                        value = float(np.clip(value, 0.0, 100.0))
                forecast_weather.append(value)

        # Optional past weather (lagged history) — only populated when
        # state_space.weather_history.enabled is true in config.
        past_weather = []
        if self.weather_history_enabled:
            # Update rolling history of *observed* weather (append happens after use so
            # lag 1 means "one step before now", not "now").
            keep = self.weather_history_horizon * 3
            for i in range(1, self.weather_history_horizon + 1):
                idx = len(self.weather_history) - i * 3
                if idx >= 0:
                    past_weather.extend(self.weather_history[idx:idx + 3])
                else:
                    past_weather.extend(current_weather)
            self.weather_history.extend(current_weather)
            if len(self.weather_history) > keep:
                self.weather_history = self.weather_history[-keep:]

        # Time features
        current_hour = exchange.hour(self.state)
        current_day = exchange.day_of_month(self.state)
        current_month = exchange.month(self.state)
        
        # Normalize to [0, 1] range
        time_features = [
            current_hour / 24.0,  # Hour of day
            0,  # Day of week (placeholder - would need proper calculation)
            current_month / 12.0  # Month of year
        ]
        
        # Previous action. The first timestep has no previous control action,
        # so that initial vector must be specified in the config.
        if self.current_action is not None:
            prev_action = self.current_action
        else:
            prev_cfg = self.hvac_config.config['state_space']['previous_actions']
            prev_action = prev_cfg.get('initial_value')
            if prev_action is None:
                raise RuntimeError(
                    "state_space.previous_actions.initial_value is required "
                    "for the first control timestep"
                )
            if len(prev_action) != prev_cfg['count']:
                raise RuntimeError(
                    "state_space.previous_actions.initial_value length must match "
                    "state_space.previous_actions.count"
                )
        
        # Outdoor CO2 (ppm): from override, CSV lookup, or explicit config constant
        outdoor_co2 = self.get_outdoor_co2_ppm(current_month, current_day, current_hour)
        co2_outdoor = [float(outdoor_co2)]
        
        # Zone CO2 (ppm) for the same zones as zone_temps
        zone_co2_list = []
        zone_co2_count = self.hvac_config.config['state_space'].get('zone_co2_ppm', {}).get('count', zone_count)
        for i, (zone_name, zhandles) in enumerate(self.handles['zones'].items()):
            if i >= zone_co2_count:
                break
            zone_co2_list.append(exchange.get_variable_value(self.state, zhandles['co2']))
        
        # Ensure all are lists (not numpy arrays) before concatenation
        zone_temps = list(zone_temps)
        current_weather = list(current_weather)
        forecast_weather = list(forecast_weather)
        past_weather = list(past_weather)
        time_features = list(time_features)
        prev_action = list(prev_action)

        # Combine all features (should match config state_size): temps, weather, forecast,
        # past weather (optional), time, prev_action, outdoor_co2, zone_co2
        all_features = zone_temps + current_weather + forecast_weather + past_weather + time_features + prev_action + co2_outdoor + zone_co2_list
        state = np.array(all_features[:self.state_size], dtype=np.float32)

        if self.hvac_config.is_normalization_enabled():
            mins, maxs = self.hvac_config.get_state_normalization_bounds()
            mins = mins[:self.state_size]
            maxs = maxs[:self.state_size]
            state = np.clip((state - mins) / (maxs - mins), 0.0, 1.0)

        return state
    
    def apply_action(self, action):
        """Apply action from RL agent to EnergyPlus model."""
        if not self.handles_initialized:
            return
        
        exchange = self.api.exchange
        
        # Map agent output [-1, 1] (tanh) to config bounds: scale = low + (a + 1) / 2 * (high - low)
        action_bounds = self.hvac_config.get_action_bounds()
        def scale(a, low, high):
            return low + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (high - low)
        sp_offset = scale(action[0], action_bounds['sp_offset'][0], action_bounds['sp_offset'][1])
        deadband = scale(action[1], action_bounds['deadband'][0], action_bounds['deadband'][1])
        airflow_mult = scale(action[2], action_bounds['airflow_multiplier'][0], action_bounds['airflow_multiplier'][1])
        
        # Calculate setpoints
        heating_sp = self.base_temp + sp_offset - deadband/2
        cooling_sp = self.base_temp + sp_offset + deadband/2
        
        # Apply to all zones
        for _, handles in self.handles['zones'].items():
            if handles['heating_sp'] > 0:
                exchange.set_actuator_value(self.state, handles['heating_sp'], heating_sp)
            if handles['cooling_sp'] > 0:
                exchange.set_actuator_value(self.state, handles['cooling_sp'], cooling_sp)
            if handles['airflow'] > 0:
                airflow = self.min_airflow + (self.max_airflow - self.min_airflow) * airflow_mult
                exchange.set_actuator_value(self.state, handles['airflow'], airflow)
        
        self.current_action = action
    
    def _compute_comfort_penalty(self, reward_config, zone_temps):
        """Compute thermal comfort penalty from config: either temperature-based or PPD-based."""
        model = reward_config.get('thermal_comfort_model', 'temperature')
        COMFORT_WEIGHT = reward_config['comfort_weight']
        if model == 'ppd':
            # PPD-based: penalty when zone PPD exceeds acceptable (requires Fanger output in IDF)
            ppd_max = reward_config.get('ppd_acceptable_max', 10.0)
            scale = reward_config.get('ppd_penalty_scale', 0.01)
            comfort_penalty = 0.0
            if self.handles_initialized:
                exchange = self.api.exchange
                for _, zhandles in self.handles['zones'].items():
                    h = zhandles.get('ppd', -1)
                    if h and h > 0:
                        ppd = exchange.get_variable_value(self.state, h)
                        comfort_penalty += max(0.0, ppd - ppd_max) * scale
            comfort_penalty *= COMFORT_WEIGHT
            return comfort_penalty
        # Temperature-based: penalty when zone temp deviates from base_temp beyond threshold
        COMFORT_THRESHOLD = reward_config['comfort_threshold']
        comfort_penalty = 0.0
        for temp in zone_temps:
            deviation = abs(temp - self.base_temp)
            if deviation > COMFORT_THRESHOLD:
                comfort_penalty += (deviation - COMFORT_THRESHOLD) * 0.5
        comfort_penalty *= COMFORT_WEIGHT
        return comfort_penalty
    
    def compute_reward_components(self, reward_config, zone_temps, action):
        """
        Compute reward and all components (energy cost, comfort, setpoint, demand, price used).
        Single place for reward logic; used by calculate_reward and for logging.
        Returns dict with: reward, energy_cost, comfort_penalty, setpoint_penalty, demand_penalty,
        total_cost, energy_price_used, energy_kwh, current_power.
        """
        ENERGY_WEIGHT = reward_config['energy_weight']
        SETPOINT_WEIGHT = reward_config['setpoint_weight']
        DEMAND_WEIGHT = reward_config['demand_weight']
        DEMAND_THRESHOLD = reward_config['demand_threshold']
        
        # Current time for real-time price
        hour = self.api.exchange.hour(self.state)
        day = self.api.exchange.day_of_month(self.state)
        month = self.api.exchange.month(self.state)
        energy_price_used = get_realtime_price(month, day, hour, reward_config)
        
        # Energy: use _cached_power_w updated by the power cache callback
        # (Facility Total Electricity Demand Rate resets before zone timestep callback fires;
        #  callback_after_predictor_after_hvac_managers captures it while it's still valid)
        dt_hours = 1.0 / self.timesteps_per_hour
        current_power = self._cached_power_w
        if current_power is None:
            raise RuntimeError(
                "Electric demand was not cached before reward calculation. "
                "Check callback_after_predictor_after_hvac_managers registration."
            )
        energy_kwh = (current_power / 1000.0) * dt_hours
        energy_cost = energy_kwh * energy_price_used * ENERGY_WEIGHT

        GAS_WEIGHT = reward_config.get('gas_weight', 0.3)
        GAS_PRICE = reward_config.get('gas_price_per_kwh', 0.017)
        gas_j = self.api.exchange.get_meter_value(self.state, self.handles['gas_meter'])
        gas_kwh = gas_j / 3_600_000.0
        current_gas_power = (gas_kwh / dt_hours) * 1000.0
        gas_cost = gas_kwh * GAS_PRICE * GAS_WEIGHT
        
        comfort_penalty = self._compute_comfort_penalty(reward_config, zone_temps)
        
        action_bounds = self.hvac_config.get_action_bounds()
        def _scale(a, lo, hi):
            return lo + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
        sp_offset = _scale(action[0], action_bounds['sp_offset'][0], action_bounds['sp_offset'][1])
        deadband = _scale(action[1], action_bounds['deadband'][0], action_bounds['deadband'][1])
        setpoint_penalty = 0.0
        if abs(sp_offset) > 2.0:
            setpoint_penalty += 0.1
        if deadband < 1.0:
            setpoint_penalty += 0.05
        setpoint_penalty *= SETPOINT_WEIGHT
        
        demand_penalty = 0.0
        current_power_kw = current_power / 1000.0  # convert W -> kW; DEMAND_THRESHOLD is in kW
        if current_power_kw > DEMAND_THRESHOLD:
            demand_penalty = (current_power_kw - DEMAND_THRESHOLD) * 0.2
        demand_penalty *= DEMAND_WEIGHT
        
        total_cost = energy_cost + gas_cost + comfort_penalty + setpoint_penalty + demand_penalty
        reward = -total_cost

        return {
            'reward': reward,
            'energy_cost': energy_cost,
            'gas_cost': gas_cost,
            'comfort_penalty': comfort_penalty,
            'setpoint_penalty': setpoint_penalty,
            'demand_penalty': demand_penalty,
            'total_cost': total_cost,
            'energy_price_used': energy_price_used,
            'energy_kwh': energy_kwh,
            'current_power': current_power,
            'gas_kwh': gas_kwh,
            'current_gas_power': current_gas_power,
        }
    
    def calculate_reward(self, action, curr_state):
        """Calculate reward based on energy efficiency and thermal comfort (uses compute_reward_components)."""
        reward_config = self.hvac_config.config['reward']
        zone_temps = curr_state[:5]
        components = self.compute_reward_components(reward_config, zone_temps, action)
        return components['reward']
    
    def step(self, action):
        """Execute one environment step."""
        prev_state = self.current_state.copy() if self.current_state is not None else None
        
        # Apply action
        self.apply_action(action)
        
        # Get new state
        self.current_state = self.get_current_state()
        
        # Calculate reward
        if prev_state is not None:
            reward = self.calculate_reward(action, self.current_state)
        else:
            reward = 0.0
        
        # Update tracking
        self.episode_reward += reward
        self.timestep_count += 1
        
        # Check if episode is done (after 1 hour of timesteps)
        current_day = self.api.exchange.day_of_month(self.state)
        
        # Episode end: use variable duration if set, else config default
        steps_this_episode = self._episode_duration_timesteps if self._episode_duration_timesteps is not None else self.episode_timesteps
        done = (self.timestep_count >= steps_this_episode) and current_day > 0 and not self.api.exchange.warmup_flag(self.state)
        
        return self.current_state, reward, done, {}
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_state = self.get_current_state()
        self.current_action = None
        self.episode_reward = 0.0
        self.timestep_count = 0
        self.weather_history = []
        return self.current_state


class RLHVACController:
    """Controller that uses RL agent for HVAC control."""
    
    def __init__(self, api, state, config_path=None, training_mode=False, live_plotter=None, loss_plotter=None,
                 output_dir=None, model_path=None, save_model=False, save_every=20):
        self.api = api
        self.state = state
        self.training_mode = training_mode
        self.live_plotter = live_plotter
        self.loss_plotter = loss_plotter
        self.output_dir = output_dir

        # Checkpointing: save_model gates both the final save and periodic
        # checkpoints every `save_every` episodes during training.
        self.save_model_enabled = save_model
        self.save_every = save_every

        # Initialize environment
        self.env = HVACEnvironment(api, state, config_path)

        # Initialize RL agent
        sac_config_path = project_root / "sac_config" / "sac_config.yaml"
        self.agent = SACAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            config_path=str(sac_config_path)
        )
        warmup_steps = self.agent.config.get('training', {}).get('warmup_steps', 0)
        self.training_start_memory = max(int(warmup_steps), int(self.agent.batch_size))
        if training_mode:
            print(
                "Training updates will start when replay memory reaches "
                f"{self.training_start_memory} transitions "
                f"(batch_size={self.agent.batch_size}, warmup_steps={warmup_steps})."
            )

        # Load a pre-trained model. --model overrides the load path explicitly (e.g. to
        # resume training or evaluate a checkpoint from a specific run's output dir);
        # otherwise eval mode falls back to models/sac_hvac_model.pth, and training mode
        # starts from scratch.
        if model_path:
            load_path = Path(model_path)
        elif not training_mode:
            load_path = project_root / "models" / "sac_hvac_model.pth"
        else:
            load_path = None
        loaded_episode_count = 0
        if load_path is not None:
            if load_path.exists():
                extra_state = self.agent.load(str(load_path))
                loaded_episode_count = extra_state.get('episode_count', 0)
                print(f"Loaded pre-trained model from {load_path}"
                      + (f" (resuming at episode {loaded_episode_count})" if loaded_episode_count else ""))
            else:
                print(f"No pre-trained model found at {load_path}, using untrained agent")

        # Episode tracking. Resumes from the loaded checkpoint's episode count (if any) so
        # continued training and plot/log episode numbers pick up where the checkpoint left off.
        self.episode_count = loaded_episode_count
        self.max_episodes = 99999  # run until end of training window (or override from run_simulation)
        self.max_episodes_reached = False
        self.training_window = self.env.hvac_config.get_training_window()
        self.episode_duration_range = self.env.hvac_config.get_episode_duration_range()
        self._episode_started_this_run = False  # true after we've called start_episode for current episode
        self._current_episode_duration_hours = None  # set at episode start for completion print
        self._power_handle_logged = False  # one-time debug for energy/demand handle
        # Random start within window: (month, day, hour) at which to start next episode; None = not yet sampled
        self._next_episode_start = None
        self._next_episode_duration_hours = None  # sampled duration for next episode
        self.training_update_count = 0
        
        # Control logging
        self.log_data = []
        self._co2_400_warned = False  # one-time diagnostic for CO2 stuck at 400
        self.fatal_error = None
        self._fatal_error_reported = False

    def _handle_callback_error(self, exc):
        """Record callback failures so run_simulation can fail cleanly."""
        self.fatal_error = exc
        if not self._fatal_error_reported:
            self._fatal_error_reported = True
            print(f"\nFATAL RL callback error: {exc}\n")
        self.max_episodes_reached = True
        self.api.runtime.stop_simulation(self.state)

    def power_cache_callback(self, state):
        """Cache electricity demand rate before it resets after zone reporting.
        Registered for callback_after_predictor_after_hvac_managers (fires during HVAC sub-iterations).
        """
        if self.api.exchange.warmup_flag(state):
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if not self.env.handles_initialized:
            return
        exchange = self.api.exchange
        if self.env.handles.get('total_power', -1) > 0:
            self.env._cached_power_w = exchange.get_variable_value(
                self.state, self.env.handles['total_power']
            )

    def pre_timestep_callback(self, state):
        """Apply CO2 and weather overrides before zone heat-balance calculations.
        Registered for callback_begin_zone_timestep_before_init_heat_balance so that overrides
        take effect for the current timestep's energy calculations.

        Usage — set on env before/during each step:
            controller.env.outdoor_co2_override = 500.0         # ppm
            controller.env.weather_override = {'dry_bulb': 28.0, 'humidity': 55.0}
        """
        if self.api.exchange.warmup_flag(state):
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if not self.env.handles_initialized:
            return
        self.env.apply_outdoor_co2_for_current_timestep()
        self.env.apply_weather_override()

    def timestep_callback(self, state):
        """Called at each EnergyPlus timestep."""
        try:
            self._timestep_callback_impl(state)
        except Exception as exc:
            self._handle_callback_error(exc)

    def _timestep_callback_impl(self, state):
        """Called at each EnergyPlus timestep, with errors surfaced by wrapper."""
        # Skip if in warmup
        if self.api.exchange.warmup_flag(state):
            return

        if not self.api.exchange.api_data_fully_ready(state):
            return

        # Skip if max episodes reached
        if self.max_episodes_reached:
            return

        # Training window: only run RL between start and end date/time
        current_month = self.api.exchange.month(self.state)
        current_day = self.api.exchange.day_of_month(self.state)
        current_hour = self.api.exchange.hour(self.state)
        if not _in_training_window(current_month, current_day, current_hour, self.training_window):
            return

        # Initialize environment on first real timestep inside window
        if not self.env.handles_initialized:
            if not self.env.initialize_handles():
                return
        
        # Sample random episode start (date/time) and duration on first use or when starting next episode
        if self._next_episode_start is None:
            self._next_episode_start = _sample_random_start_in_window(self.training_window)
            min_h, max_h = self.episode_duration_range
            self._next_episode_duration_hours = min_h if min_h == max_h else np.random.uniform(min_h, max_h)
        
        # Skip timesteps until we reach the randomly chosen episode start
        if (current_month, current_day, current_hour) < self._next_episode_start:
            return
        
        # Start of new episode: reset counters and set duration from sampled value
        if not self._episode_started_this_run:
            self.env.timestep_count = 0
            self.env.episode_reward = 0.0
            duration_hours = self._next_episode_duration_hours
            duration_timesteps = int(duration_hours * self.env.timesteps_per_hour)
            self.env.start_episode(duration_timesteps)
            self._episode_started_this_run = True
            self._current_episode_duration_hours = duration_hours
            nm, nd, nh = self._next_episode_start
            print(f"\n--- Episode {self.episode_count + 1} started at {nm}/{nd:02d} {nh:02d}:00 (duration: {duration_hours:.1f} h) ---\n")
        
        # Get current state
        current_state = self.env.get_current_state()
        
        # Select action
        if self.training_mode:
            action = self.agent.select_action(current_state, evaluate=False)
        else:
            action = self.agent.select_action(current_state, evaluate=True)
        
        # Execute action. next_state/current_state may be normalized for the agent;
        # reward/reporting below uses raw EnergyPlus values.
        next_state, env_step_reward, done, _ = self.env.step(action)
        
        # Reward components from single source (same as env.calculate_reward); includes real-time price
        reward_config = self.env.hvac_config.config['reward']
        zone_count_for_reward = self.env.hvac_config.config['state_space']['zone_temps']['count']
        raw_zone_temps = []
        for zone_name, zhandles in list(self.env.handles['zones'].items())[:zone_count_for_reward]:
            h_temp = zhandles.get('temp', -1)
            if h_temp and h_temp > 0:
                raw_zone_temps.append(self.api.exchange.get_variable_value(self.state, h_temp))
        if not raw_zone_temps:
            raw_zone_temps = current_state[:zone_count_for_reward]
        components = self.env.compute_reward_components(reward_config, raw_zone_temps, action)
        reward = components['reward']
        self.env.episode_reward += reward - env_step_reward
        energy_cost = components['energy_cost']
        gas_cost = components['gas_cost']
        comfort_penalty = components['comfort_penalty']
        setpoint_penalty = components['setpoint_penalty']
        demand_penalty = components['demand_penalty']
        total_cost = components['total_cost']
        current_power = components['current_power']
        energy_price_used = components['energy_price_used']
        energy_kwh = components['energy_kwh']
        gas_kwh = components['gas_kwh']
        current_gas_power = components['current_gas_power']
        
        # Store experience if training
        if self.training_mode:
            self.agent.store_transition(current_state, action, reward, next_state, float(done))
            
            # Update agent only after both the warmup threshold and batch size are satisfied.
            if len(self.agent.memory) >= self.training_start_memory:
                losses = self.agent.update_parameters()
                if losses is not None:
                    self.training_update_count += 1
                    if self.loss_plotter is not None:
                        self.loss_plotter.update(self.training_update_count, losses)
                    if self.training_update_count == 1 or self.training_update_count % 10 == 0:
                        print(
                            "   Training: "
                            f"update={self.training_update_count}  "
                            f"memory_len={len(self.agent.memory)}  "
                            f"actor_loss={losses['actor_loss']:.4f}  "
                            f"critic1_loss={losses['critic1_loss']:.4f}  "
                            f"critic2_loss={losses['critic2_loss']:.4f}"
                        )
        
        # Get time info
        current_hour = self.api.exchange.hour(self.state)
        current_day = self.api.exchange.day_of_month(self.state)
        current_month = self.api.exchange.month(self.state)
        
        # Current episode number (1-based)
        episode_no = self.episode_count + 1

        action_bounds = self.env.hvac_config.get_action_bounds()

        def _scale(a, lo, hi):
            return lo + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)

        sp_offset = _scale(action[0], action_bounds['sp_offset'][0], action_bounds['sp_offset'][1])
        deadband = _scale(action[1], action_bounds['deadband'][0], action_bounds['deadband'][1])
        airflow_mult = _scale(action[2], action_bounds['airflow_multiplier'][0], action_bounds['airflow_multiplier'][1])
        airflow_actual = self.env.min_airflow + (self.env.max_airflow - self.env.min_airflow) * airflow_mult
        htg_sp = self.env.base_temp + sp_offset - deadband / 2
        clg_sp = self.env.base_temp + sp_offset + deadband / 2
        outdoor_co2 = self.env.get_outdoor_co2_ppm(current_month, current_day, current_hour)
        raw_outdoor_temp = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(
            self.state,
            self.api.exchange.hour(self.state),
            self.api.exchange.zone_time_step_number(self.state),
        )
        
        # Zone temperature, CO2 (ppm), people, and thermal comfort PPD/PMV;
        # -1 or 0 handle means not available in this IDF.
        zone_temp = {}
        zone_co2 = {}
        zone_people = {}
        zone_ppd = {}
        zone_pmv = {}
        exchange = self.api.exchange
        for zone_name, zhandles in self.env.handles['zones'].items():
            h_temp = zhandles.get('temp', -1)
            if h_temp and h_temp > 0:
                zone_temp[f'temp_{zone_name}'] = exchange.get_variable_value(self.state, h_temp)
            else:
                zone_temp[f'temp_{zone_name}'] = np.nan
            h = zhandles.get('co2', -1)
            if h and h > 0:
                zone_co2[f'co2_{zone_name}'] = exchange.get_variable_value(self.state, h)
            else:
                zone_co2[f'co2_{zone_name}'] = np.nan
            h_people = zhandles.get('people', -1)
            if h_people and h_people > 0:
                zone_people[f'people_{zone_name}'] = exchange.get_variable_value(self.state, h_people)
            else:
                zone_people[f'people_{zone_name}'] = np.nan
            h_ppd = zhandles.get('ppd', -1)
            if h_ppd and h_ppd > 0:
                zone_ppd[f'ppd_{zone_name}'] = exchange.get_variable_value(self.state, h_ppd)
            else:
                zone_ppd[f'ppd_{zone_name}'] = np.nan
            h_pmv = zhandles.get('pmv', -1)
            if h_pmv and h_pmv > 0:
                zone_pmv[f'pmv_{zone_name}'] = exchange.get_variable_value(self.state, h_pmv)
            else:
                zone_pmv[f'pmv_{zone_name}'] = np.nan
        
        # One-time diagnostic: CO2 often stays 400 ppm during unoccupied hours (outdoor default)
        if not self._co2_400_warned and zone_co2:
            co2_vals = [v for v in zone_co2.values() if not np.isnan(v)]
            if co2_vals and all(abs(v - 400.0) < 1.0 for v in co2_vals):
                self._co2_400_warned = True
                print("\n  [CO2] All zone CO2 ≈ 400 ppm (outdoor default). "
                      "Occupancy in this model is 0 until 07:00; use --episodes 8 or more to include occupied hours (7–18) and see CO2 rise.\n")
        
        # Log data (including episode, KPIs, reward components, real-time price, zone CO2, PPD/PMV)
        log_entry = {
            'episode': episode_no,
            'timestep': self.env.timestep_count,
            'action': action.tolist(),
            'reward': reward,
            'energy_cost': energy_cost,
            'gas_cost': gas_cost,
            'comfort_penalty': comfort_penalty,
            'setpoint_penalty': setpoint_penalty,
            'demand_penalty': demand_penalty,
            'total_cost': total_cost,
            'energy_price_used': energy_price_used,
            'energy_kwh': energy_kwh,
            'current_power': current_power,
            'gas_kwh': gas_kwh,
            'current_gas_power': current_gas_power,
            'avg_zone_temp': np.mean(raw_zone_temps),
            'outdoor_temp': raw_outdoor_temp,
            'outdoor_co2': outdoor_co2,
            'heating_setpoint': htg_sp,
            'cooling_setpoint': clg_sp,
            'setpoint_offset': sp_offset,
            'deadband': deadband,
            'airflow': airflow_actual,
            'bottom_floor_airflow': airflow_actual,
            'mid_floor_airflow': airflow_actual,
            'top_floor_airflow': airflow_actual,
            'episode_reward': self.env.episode_reward,
            'hour': current_hour,
            'day': current_day,
            'month': current_month,
            'people': np.nansum(list(zone_people.values())) if zone_people else np.nan,
            **zone_temp,
            **zone_co2,
            **zone_people,
            **zone_ppd,
            **zone_pmv
        }
        self.log_data.append(log_entry)
        if self.live_plotter is not None:
            self.live_plotter.update(log_entry)
        
        # Print progress (use step-within-episode for display; variable-length episodes)
        timestep_minutes = 60 // self.env.timesteps_per_hour
        episode_step = self.env.timestep_count  # 0-based step within current episode (no modulo wrap)
        
        # Get actual time from EnergyPlus
        actual_hour = self.api.exchange.hour(self.state)
        actual_minute = int((self.api.exchange.minute_of_hour(self.state) if hasattr(self.api.exchange, 'minute_of_hour') 
                            else (episode_step % self.env.timesteps_per_hour) * timestep_minutes))
        # Note: gap between episodes is always one sim timestep (e.g. 15 min); "Step" is 0-based within episode
        actual_day = self.api.exchange.day_of_month(self.state)
        
        # Print every timestep with full details
        # Create datetime object
        datetime_str = f"2023-{current_month:02d}-{actual_day:02d} {actual_hour:02d}:{actual_minute:02d}"
        
        # Format control action with absolute units
    
        
        # Same [-1,1] -> bounds scaling as apply_action (for display)
        # States: zone temps (first 5), then outdoor temp
        zone_count = self.env.hvac_config.config['state_space']['zone_temps']['count']
        zone_temps = current_state[:zone_count]
        if len(current_state) <= zone_count:
            raise RuntimeError(
                f"State is missing outdoor temperature at index {zone_count}; "
                f"state length is {len(current_state)}"
            )
        outdoor_temp = current_state[zone_count]
        display_zone_temps = raw_zone_temps[:zone_count]
        zone_str = ", ".join([f"{t:5.1f}" for t in display_zone_temps])
        memory_len = len(self.agent.memory)
        memory_status = f"{memory_len}/{self.training_start_memory}" if self.training_mode else str(memory_len)
        
        # Print: time, step; actual actions (raw + scaled); states; reward; then blank line
        print(f"{datetime_str} | Ep {episode_no:3d} | Step {episode_step:3d}")
        action_items = [
            f"raw_sp={action[0]:+.2f}", f"raw_db={action[1]:+.2f}", f"raw_af={action[2]:+.2f}",
            f"sp_offset={sp_offset:+.2f}°C", f"htg_sp={htg_sp:.1f}°C",
            f"clg_sp={clg_sp:.1f}°C", f"deadband={deadband:.2f}°C", f"airflow={airflow_actual:.4f}m³/s",
        ]
        print("   Actions:")
        for i in range(0, len(action_items), 5):
            print("     " + "  ".join(action_items[i:i+5]))
        print(f"   States:  zone_temps=[{zone_str}]°C  outdoor_temp={raw_outdoor_temp:5.1f}°C  memory_len={memory_status}")
        print(f"   State vector [{len(current_state)}]:")
        for i in range(0, len(current_state), 10):
            chunk = current_state[i:i+10]
            print("     " + "  ".join([f"[{i+j:2d}]{v:+.3f}" for j, v in enumerate(chunk)]))
        print(f"   Reward: {reward:7.3f}  episode_total={self.env.episode_reward:7.3f}  |  elec_cost[$]={energy_cost:.4f}  gas_cost[$]={gas_cost:.4f}  comfort={comfort_penalty:.4f}  setpoint={setpoint_penalty:.4f}  demand_penalty[$]={demand_penalty:.4f}  total_cost={total_cost:.4f}  |  price[$/kWh]={energy_price_used:.3f}  elec[kWh]={energy_kwh:.4f}  elec[kW]={current_power/1000:.2f}  gas[kWh]={gas_kwh:.4f}  gas[kW]={current_gas_power/1000:.2f}")
        print()
        
        # Reset if episode done: sample next random start (after current time) and duration
        if done:
            self.episode_count += 1
            dur_h = getattr(self, '_current_episode_duration_hours', None)
            dur_str = f" (duration: {dur_h:.1f} h)" if dur_h is not None else ""
            print(f"\n--- Episode {self.episode_count} completed{dur_str} ---\n")

            # Periodic checkpoint every `save_every` episodes during training
            if (self.training_mode and self.save_model_enabled and self.save_every > 0
                    and self.episode_count % self.save_every == 0):
                ckpt_dir = os.path.join(self.output_dir, 'checkpoints') if self.output_dir else 'checkpoints'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'rl_hvac_model_ep{self.episode_count}.pth')
                self.save_model(ckpt_path)

            # Stop processing if reached max episodes
            if self.episode_count >= self.max_episodes:
                print(f"Reached maximum episodes ({self.max_episodes}), logging disabled")
                self.max_episodes_reached = True
                return
            self._episode_started_this_run = False
            # Next episode: random start date/time after current, random duration in [min, max]
            self._next_episode_start = _sample_random_start_in_window(
                self.training_window,
                after_month=current_month, after_day=current_day, after_hour=current_hour
            )
            min_h, max_h = self.episode_duration_range
            self._next_episode_duration_hours = min_h if min_h == max_h else np.random.uniform(min_h, max_h)
    
    def get_summary(self):
        """Return summary of RL control performance."""
        if not self.log_data:
            return "No control data logged"
        
        rewards = [d['reward'] for d in self.log_data]
        temps = [d['avg_zone_temp'] for d in self.log_data]
        
        summary = f"""
RL HVAC Control Summary:
  Total timesteps: {len(self.log_data)}
  Training mode: {self.training_mode}
  
  Performance:
    Total Reward: {self.env.episode_reward:.2f}
    Avg Reward/Step: {np.mean(rewards):.3f}
    Best Step Reward: {max(rewards):.3f}
    Worst Step Reward: {min(rewards):.3f}
    
  Temperature Control:
    Avg Zone Temp: {np.mean(temps):.2f}°C
    Temp Std Dev: {np.std(temps):.2f}°C
    Min Temp: {min(temps):.2f}°C
    Max Temp: {max(temps):.2f}°C
"""
        return summary
    
    def save_model(self, path):
        """Save trained RL model, including episode_count so a resumed run/plots can
        continue numbering from where this checkpoint left off."""
        self.agent.save(path, extra_state={'episode_count': self.episode_count})
        print(f"Model saved to {path}")
    
    def save_log_to_csv(self, filepath):
        """Save logged data to CSV file."""
        if not self.log_data:
            print("No data to save")
            return
        
        df = pd.DataFrame(self.log_data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.log_data)} timesteps to {filepath}")


def run_simulation(
    idf_path,
    epw_path,
    output_dir,
    config,
    max_episodes=None,
    training_mode=False,
    override_test=False,
    live_plot=False,
    loss_plot=False,
    live_plot_every=1,
    live_plot_hold=False,
    live_plot_scope="current",
    model_path=None,
    save_model=False,
    save_every=20,
):
    """Run EnergyPlus simulation with RL HVAC control.
    
    Simulation is driven by config training_window (start/end date and time) and
    episode_duration_hours [min, max]. A custom IDF is created when custom_period.enabled
    with RunPeriod = training_window start date to end date. RL runs only between
    start_date+start_hour and end_date+end_hour; each episode length is sampled from
    [episode_duration_hours min, max].
    """
    hvac_config = get_hvac_config(config)
    sim_cfg = hvac_config.config['simulation']
    window = sim_cfg.get('training_window', {})
    start_month = window.get('start_month', sim_cfg.get('start_month', 6))
    start_day = window.get('start_day', sim_cfg.get('start_day', 1))
    end_month = window.get('end_month', start_month)
    end_day = window.get('end_day', start_day)
    start_hour = window.get('start_hour', 0)
    end_hour = window.get('end_hour', 24)
    ep_range = sim_cfg.get('episode_duration_hours', [24, 24])
    if isinstance(ep_range, list):
        ep_min, ep_max = ep_range[0], ep_range[1]
    else:
        ep_min = ep_max = sim_cfg.get('episode_hours', 24)
    
    if max_episodes is not None:
        pass  # use CLI/config override
    else:
        # Run until end of window (controller uses large default); no fixed episode count
        max_episodes = 99999
    
    if sim_cfg.get('custom_period', {}).get('enabled', False):
        season_info = get_season_info(start_month)
        sim_days = calculate_simulation_days(start_month, start_day, end_month, end_day)
        print(f"\n" + "=" * 60)
        print("TRAINING WINDOW (from config)")
        print("=" * 60)
        print(f"  Date range: {start_month}/{start_day:02d} to {end_month}/{end_day:02d} ({sim_days} days)")
        print(f"  Time range: {start_hour}:00 to {end_hour}:00 (RL runs only in this window)")
        print(f"  Episode duration: [{ep_min}, {ep_max}] h (random per episode); start date/time: random in window")
        print(f"  Season: {season_info['name']} ({season_info['description']})")
        print("=" * 60)
        sim_cfg = hvac_config.config.get('simulation', {})
        outdoor_co2 = sim_cfg.get('outdoor_co2_ppm')
        outdoor_co2_csv = sim_cfg.get('outdoor_co2_csv_path')
        outdoor_co2_fallback = sim_cfg.get('outdoor_co2_fallback_ppm')
        idf_path = create_custom_idf(
            idf_path, start_month, start_day, end_month, end_day, output_dir,
            outdoor_co2_ppm=outdoor_co2,
            outdoor_co2_csv_path=outdoor_co2_csv,
            outdoor_co2_fallback_ppm=outdoor_co2_fallback,
        )
    
    live_plotter = None
    loss_plotter = None
    if live_plot:
        live_plotter = LiveRLPlotter(
            output_dir=output_dir,
            update_every=live_plot_every,
            episode_scope=live_plot_scope,
        )
        print(
            f"Live plot enabled (updates every {max(1, int(live_plot_every))} "
            f"timestep(s), scope: {live_plot_scope})"
        )
    if loss_plot:
        if training_mode:
            loss_plotter = LiveLossPlotter(
                output_dir=output_dir,
                update_every=live_plot_every,
            )
            print("SAC loss live plot enabled")
        else:
            print("Warning: --loss-plot requested without --training; no SAC losses will be plotted")

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    controller = RLHVACController(
        api,
        state,
        config,
        training_mode=training_mode,
        live_plotter=live_plotter,
        loss_plotter=loss_plotter,
        output_dir=output_dir,
        model_path=model_path,
        save_model=save_model,
        save_every=save_every,
    )
    controller.max_episodes = max_episodes

    if override_test:
        # Fixed values chosen to be obviously different from any real EPW/schedule value
        controller.env.weather_override = {
            'dry_bulb':      35.0,   # °C  — expect outdoor_temp=35.0 in every step
            'humidity':      80.0,   # %
            'wind_speed':    1.0,    # m/s
        }
        controller.env.outdoor_co2_override = 650.0  # ppm — expect outdoor CO2=650 in state
        print("\n" + "=" * 60)
        print("[Override Test] Active — injecting fixed values every timestep:")
        print("  weather_override : dry_bulb=35.0°C  humidity=80%  wind_speed=1.0 m/s")
        print("  outdoor_co2_override : 650 ppm")
        print("Verify: 'outdoor_temp' in step output should read 35.0 (not EPW value)")
        print("=" * 60 + "\n")

    # Register callbacks
    # 1. Pre-timestep: apply CO2 and weather overrides before zone heat-balance calculations
    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
        state, controller.pre_timestep_callback
    )
    # 2. Power cache: reads electricity/gas demand rates while still valid
    #    (HVAC demand rate variables reset before the zone timestep end callback fires)
    api.runtime.callback_after_predictor_after_hvac_managers(
        state, controller.power_cache_callback
    )
    # 3. Main RL loop: observe state, select action, compute reward, log
    api.runtime.callback_end_zone_timestep_after_zone_reporting(
        state, controller.timestep_callback
    )
    
    # Print configuration summary and named state vector columns
    cfg = controller.env.hvac_config.config
    ss = cfg['state_space']
    zone_names = list(controller.env.handles['zones'].keys()) if controller.env.handles_initialized else CONTROLLED_ZONE_NAMES
    zone_count = ss['zone_temps']['count']
    wf = controller.env.hvac_config.weather_forecast_offsets
    weather_history_cfg = ss.get('weather_history', {})
    past_horizon = weather_history_cfg.get('horizon', 0) if weather_history_cfg.get('enabled', False) else 0
    co2_count = ss.get('zone_co2_ppm', {}).get('count', zone_count)

    state_columns = []
    for z in zone_names[:zone_count]:
        state_columns.append(f"zone_temp_{z}")
    state_columns += ["oat", "humidity", "sky_temp"]
    for field, label in (('oat', 'oat'), ('humidity', 'humidity'), ('cloud_cover', 'sky_temp')):
        for t in wf[field]:
            state_columns.append(f"forecast_t{t}_{label}")
    for t in range(1, past_horizon + 1):
        state_columns += [f"past_t{t}_oat", f"past_t{t}_humidity", f"past_t{t}_sky_temp"]
    state_columns += ["hour_of_day", "day_of_week", "month_of_year"]
    state_columns += ["prev_sp_offset", "prev_deadband", "prev_airflow"]
    state_columns += ["outdoor_co2"]
    for z in zone_names[:co2_count]:
        state_columns.append(f"zone_co2_{z}")

    print("\n" + "=" * 60)
    print("RL HVAC Control Configuration")
    print("=" * 60)
    print(f"  Timesteps per hour: {controller.env.timesteps_per_hour}")
    print(f"  Training mode:      {controller.training_mode}")
    print(f"  Action size:        {controller.env.action_size}  [sp_offset, deadband, airflow_multiplier]")
    action_cfg = cfg['action_space']
    action_columns = [
        ("sp_offset",          action_cfg['sp_offset']['min'],          action_cfg['sp_offset']['max'],          "°C"),
        ("deadband",           action_cfg['deadband']['min'],           action_cfg['deadband']['max'],           "°C"),
        ("airflow_multiplier", action_cfg['airflow_multiplier']['min'], action_cfg['airflow_multiplier']['max'], "x"),
    ]

    print(f"  Action vector columns [{len(action_columns)}]:")
    for i, (name, lo, hi, unit) in enumerate(action_columns):
        print(f"    [{i}] {name:<22} range [{lo}, {hi}] {unit}  (agent output: tanh [-1, 1])")
    norm_enabled = controller.env.hvac_config.is_normalization_enabled()
    norm_label = "ON" if norm_enabled else "OFF"
    print(f"  State size:         {controller.env.state_size}  (normalization: {norm_label})")
    print(f"  State vector columns [{len(state_columns)}]:")
    max_name_len = max(len(n) for n in state_columns)
    for i in range(0, len(state_columns), 6):
        chunk = state_columns[i:i+6]
        print("    " + "  ".join([f"[{i+j:2d}] {name:<{max_name_len}}" for j, name in enumerate(chunk)]))
    print("=" * 60)

    try:
        input("\nPress Enter to start simulation, or Ctrl+C to abort: ")
    except KeyboardInterrupt:
        print("\nSimulation aborted.")
        return 1
    except EOFError:
        print("\nNo interactive stdin detected; starting simulation.")

    eplus_args = ['-w', epw_path, '-d', output_dir, idf_path]
    exit_code = api.runtime.run_energyplus(state, eplus_args)
    if controller.fatal_error is not None:
        print(f"RL controller failed: {controller.fatal_error}")
        exit_code = 1
    
    # Clean up
    if controller.training_mode and controller.save_model_enabled and controller.fatal_error is None:
        final_model_path = os.path.join(output_dir, 'rl_hvac_model.pth')
        controller.save_model(final_model_path)
    
    if controller.fatal_error is None:
        controller.save_log_to_csv(os.path.join(output_dir, 'rl_hvac_log.csv'))
        if live_plotter is not None:
            live_plotter.finish(output_dir, hold=live_plot_hold)
        if loss_plotter is not None:
            loss_plotter.finish(output_dir, hold=live_plot_hold)
    
    api.state_manager.delete_state(state)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("SIMULATION COMPLETED SUCCESSFULLY")
    else:
        print(f"SIMULATION FAILED (exit code: {exit_code})")
    print("=" * 60)
    
    print(controller.get_summary())
    
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description='Run EnergyPlus simulation with RL-based HVAC control.'
    )
    
    parser.add_argument('--idf', type=str, default=None,
                        help='Path to IDF file (default: from config simulation.idf_path)')
    parser.add_argument('--config', type=str, 
                        default='config/hvac_config.yaml',
                        help='Path to HVAC configuration file')
    parser.add_argument('--epw', type=str,
                        default='weather/chicago/TMY_lat41.88_lon-87.63.epw',
                        help='Path to weather file')
    parser.add_argument('--output', type=str,
                        default='outputs/rl_hvac_control',
                        help='Output directory')
    parser.add_argument('--training', action='store_true',
                        help='Enable training mode')
    parser.add_argument('--model', type=str,
                        help='Path to a pre-trained model to load at startup (overrides the default '
                             'models/sac_hvac_model.pth lookup used in eval mode; also works with --training '
                             'to resume from a checkpoint)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoints during training: every --save-every episodes, plus a '
                             'final save at the end, both under the run\'s --output directory')
    parser.add_argument('--save-every', type=int, default=20,
                        help='Episodes between periodic checkpoint saves during training (default: 20; '
                             'only used when --save-model is set)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (default: from config duration_hours / episode_hours)')
    parser.add_argument('--override-test', action='store_true',
                        help='Inject fixed weather (dry_bulb=35°C, humidity=80%%) and CO2 (650 ppm) every step to verify override pipeline')
    parser.add_argument('--live-plot', action='store_true',
                        help='Show a live matplotlib plot while the RL simulation is running')
    parser.add_argument('--loss-plot', action='store_true',
                        help='Show a separate live plot of SAC training losses; only active with --training')
    parser.add_argument('--live-plot-every', type=int, default=1,
                        help='Refresh the live plot every N logged timesteps (default: 1)')
    parser.add_argument('--live-plot-hold', action='store_true',
                        help='Keep the live plot window open after the simulation finishes')
    parser.add_argument('--live-plot-scope', choices=['current', 'all'], default='current',
                        help='Live plot scope: current resets at each episode; all overlays all episodes on one continuous timeline')
    
    args = parser.parse_args()
    
    # Load config first for IDF and paths (no hardcoded IDF)
    hvac_config = get_hvac_config(args.config)
    sim_cfg = hvac_config.config.get('simulation', {})
    idf_rel = args.idf or sim_cfg.get('idf_path', 'energyplus/control_models/MediumOffice_IAQ.idf')
    args.idf = str(project_root / idf_rel) if not os.path.isabs(idf_rel) else idf_rel
    if not os.path.isabs(args.epw):
        args.epw = str(project_root / args.epw)
    if not os.path.isabs(args.output):
        args.output = str(project_root / args.output)
    
    # Load weather file from config if not specified
    if args.epw == "energyplus/weather/USA_CO_Denver.Intl.AP.724650_TMY3.epw" and 'weather_files' in hvac_config.config:
        args.epw = str(project_root / hvac_config.config['weather_files']['summer'])
        print(f"Using weather file: {args.epw}")
    
    # Validate inputs
    if not os.path.exists(args.idf):
        print(f"Error: IDF file not found: {args.idf}")
        return 1
    if not os.path.exists(args.epw):
        print(f"Error: Weather file not found: {args.epw}")
        return 1
    
    # Run simulation: max_episodes from --episodes; --training enables storing transitions (memory grows)
    return run_simulation(
        args.idf, args.epw, args.output, args.config,
        max_episodes=args.episodes,
        training_mode=args.training,
        override_test=args.override_test,
        live_plot=args.live_plot,
        loss_plot=args.loss_plot,
        live_plot_every=args.live_plot_every,
        live_plot_hold=args.live_plot_hold,
        live_plot_scope=args.live_plot_scope,
        model_path=args.model,
        save_model=args.save_model,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    sys.exit(main())
