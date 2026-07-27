"""
RL-based HVAC Control with Setpoint and Airflow Management

This script uses an existing RL agent (SAC) to control:
- Heating and cooling setpoints with deadband
- AHU outdoor-air controller mass flow rates

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
from datetime import datetime, timedelta
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

FLOOR_ORDER = ["bottom", "mid", "top"]

# Threshold above which RTP energy price is shaded as "high" on the temperature panels.
HIGH_PRICE_THRESHOLD = 0.12
# ASHRAE-typical design PPD (%) shown as a horizontal guide on the People/PPD panel.
PPD_THRESHOLD = 10.0
PPD_LINE_COLOR = "tab:olive"
PPD_FILL_COLOR = "#c5d86d"  # lighter olive under the PPD curve

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

# Core zones (one per floor) and perimeter zones grouped by floor for zone_temps.mode=perimeter_core
CORE_ZONE_BY_FLOOR = {
    "bottom": "Core_bottom",
    "mid": "Core_mid",
    "top": "Core_top",
}

PERIMETER_ZONE_GROUPS = {
    "bottom": [
        "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2", "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
    ],
    "mid": [
        "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
    ],
    "top": [
        "Perimeter_top_ZN_1", "Perimeter_top_ZN_2", "Perimeter_top_ZN_3", "Perimeter_top_ZN_4",
    ],
}

# State vector column names when zone_temps.mode is "perimeter_core"
# Order: core_bottom, core_mid, core_top, perimeter_bottom, perimeter_mid, perimeter_top
PERIMETER_CORE_TEMP_COLUMNS = [
    "core_bottom", "core_mid", "core_top",
    "perimeter_bottom", "perimeter_mid", "perimeter_top",
]

# State vector column names when zone_temps.mode is "floor"
FLOOR_TEMP_COLUMNS = ["bottom", "mid", "top"]

AHU_OA_CONTROLS = {
    # Direct Controller:OutdoorAir actuator from eplusout.edd:
    # Component Type = "Outdoor Air Controller", Control Type = "Air Mass Flow Rate".
    # The same RL action is applied to all three floor PACUs.
    "bottom": {
        "airloop": "PACU_VAV_bot",
        "controller": "PACU_VAV_BOT_OA_CONTROLLER",
    },
    "mid": {
        "airloop": "PACU_VAV_mid",
        "controller": "PACU_VAV_MID_OA_CONTROLLER",
    },
    "top": {
        "airloop": "PACU_VAV_top",
        "controller": "PACU_VAV_TOP_OA_CONTROLLER",
    },
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
        # x-axis is elapsed hours since episode start (see update()); these track the
        # real clock hour the episode began at (for tick labels) and the last seen
        # calendar day (to draw a marker line whenever a new day starts).
        self._episode_start_hour = None
        self._last_month_day = None
        self.day_markers = []
        mpl_cache_dir = Path(output_dir) / ".matplotlib-cache"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
            from matplotlib.patches import Patch
        except ImportError as exc:
            raise RuntimeError(
                "Live plotting requires matplotlib. Install it in the active environment."
            ) from exc

        self.plt = plt
        self.FuncFormatter = FuncFormatter
        self.Patch = Patch
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
            "energy_price": [],
            "reward": [],
            "avg_co2": [],
            "bottom_floor_co2": [],
            "mid_floor_co2": [],
            "top_floor_co2": [],
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
            "commanded_oa_mass_flow": [],
            "bottom_floor_airflow_commanded": [],
            "mid_floor_airflow_commanded": [],
            "top_floor_airflow_commanded": [],
            "people": [],
            "avg_ppd": [],
            "energy_cost": [],
            "gas_cost": [],
            "comfort_penalty": [],
            "setpoint_penalty": [],
            "demand_penalty": [],
            "co2_penalty": [],
            "total_cost": [],
        }

        self.lines = {}
        self.fills = {}
        self.plot_axes = []
        self.temp_axes = []
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
            "energy_price": [],
            "reward": [],
            "avg_co2": [],
            "bottom_floor_co2": [],
            "mid_floor_co2": [],
            "top_floor_co2": [],
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
            "commanded_oa_mass_flow": [],
            "bottom_floor_airflow_commanded": [],
            "mid_floor_airflow_commanded": [],
            "top_floor_airflow_commanded": [],
            "people": [],
            "avg_ppd": [],
            "energy_cost": [],
            "gas_cost": [],
            "comfort_penalty": [],
            "setpoint_penalty": [],
            "demand_penalty": [],
            "co2_penalty": [],
            "total_cost": [],
        }

    def _reset_for_episode(self, episode):
        self.series = self._empty_series()
        self.current_episode = episode
        self.step_count = 0
        self._episode_start_hour = None
        self._last_month_day = None
        for marker in self.day_markers:
            marker.remove()
        self.day_markers = []
        for line in self.lines.values():
            line.set_data([], [])
        for fill in self.fills.values():
            fill.remove()
        self.fills = {}
        self.fig.suptitle(f"RL HVAC Simulation Live Plot - Episode {episode}")
        for ax in self.plot_axes:
            ax.relim()
            ax.autoscale_view()
        self.ax_co2.set_ylim(350, 900)

        self.fig.canvas.draw_idle()

    def _hour_of_day_formatter(self, x, pos=None):
        """Format an elapsed-hours x-value as the actual clock hour of day (HH:MM)."""
        if self.episode_scope != "current" or self._episode_start_hour is None:
            return f"{x:.0f}"
        clock_hour = (self._episode_start_hour + x) % 24
        hh = int(clock_hour)
        mm = int(round((clock_hour - hh) * 60)) % 60
        return f"{hh:02d}:{mm:02d}"

    def _add_day_marker(self, x, month, day):
        for ax in self.plot_axes:
            self.day_markers.append(
                ax.axvline(x, color="gray", linestyle=":", alpha=0.6, linewidth=1)
            )
        self.day_markers.append(
            self.plot_axes[0].text(
                x, 1.0, f"{month}/{day:02d}",
                transform=self.plot_axes[0].get_xaxis_transform(),
                va="bottom", ha="center", fontsize=7, color="gray",
            )
        )

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
        price_patch = self.Patch(
            color="orange", alpha=0.08, label=f"High RTP (>${HIGH_PRICE_THRESHOLD:.2f}/kWh)"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [price_patch], labels + [price_patch.get_label()], loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self._style_axis(ax)
        self.temp_axes.append(ax)

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
        self.lines[series_name], = ax.plot(
            [], [], label=f"{label} actual", color="tab:blue", linewidth=1.8
        )
        cmd_key = f"{series_name}_commanded"
        self.lines[cmd_key], = ax.plot(
            [], [], label="Commanded OA", color="tab:orange", linestyle="--",
            linewidth=1.5, alpha=0.9, drawstyle="steps-post",
        )
        if show_title:
            self._set_panel_title(ax, "AHU Outdoor Air (actual + commanded)")
        self._add_inside_ylabel(ax, "OA Flow (kg/s)")
        ax.legend(loc="lower right", fontsize=6)
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
        ax_top_airflow.set_xlabel("Hour of Day")
        self.plot_axes = [
            ax_bottom_temp, ax_mid_temp, ax_top_temp,
            ax_bottom_airflow, ax_mid_airflow, ax_top_airflow,
            ax_people, ax_co2, ax_electric, ax_kpi_cost, ax_reward,
        ]

        self.lines["people"], = ax_people.plot(
            [], [], label="People", color="black", linewidth=1.2
        )
        self._set_panel_title(ax_people, "People and PPD")
        self._add_inside_ylabel(ax_people, "People")
        ax_ppd = ax_people.twinx()
        self.ax_ppd = ax_ppd
        self.lines["avg_ppd"], = ax_ppd.plot(
            [], [], label="Avg PPD", color=PPD_LINE_COLOR, linewidth=1.8
        )
        self.ppd_threshold_line = ax_ppd.axhline(
            PPD_THRESHOLD,
            color="maroon",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=f"PPD threshold ({PPD_THRESHOLD:.0f}%)",
        )
        self._add_inside_ylabel(ax_ppd, "PPD (%)", side="right")
        people_lines = [
            self.lines["people"],
            self.lines["avg_ppd"],
            self.ppd_threshold_line,
        ]
        ax_people.legend(
            people_lines, [line.get_label() for line in people_lines],
            loc="lower right", fontsize=8,
        )
        ax_people.grid(True, alpha=0.3)
        self._style_axis(ax_people)
        self._style_axis(ax_ppd)
        self.plot_axes.append(ax_ppd)

        self.lines["bottom_floor_co2"], = ax_co2.plot(
            [], [], label="Bottom Floor Avg", color="tab:cyan", linewidth=2.0
        )
        self.lines["mid_floor_co2"], = ax_co2.plot(
            [], [], label="Mid Floor Avg", color="teal", linewidth=2.0
        )
        self.lines["top_floor_co2"], = ax_co2.plot(
            [], [], label="Top Floor Avg", color="darkcyan", linewidth=2.0
        )
        self.lines["outdoor_co2"], = ax_co2.plot(
            [], [], label="Outdoor CO2", color="tab:green", linestyle="--"
        )
        ax_co2.set_xlabel("Hour of Day")
        self._set_panel_title(ax_co2, "Floor Avg CO2 and Outdoor Temp")
        self._add_inside_ylabel(ax_co2, "CO2 (ppm)")
        ax_co2.set_ylim(350, 900)
        self.ax_co2 = ax_co2
        ax_oat = ax_co2.twinx()
        self.lines["outdoor_temp"], = ax_oat.plot([], [], label="Outdoor Temp", color="tab:orange")
        self._add_inside_ylabel(ax_oat, "Outdoor Temp (C)", side="right")
        co2_lines = [
            self.lines["bottom_floor_co2"],
            self.lines["mid_floor_co2"],
            self.lines["top_floor_co2"],
            self.lines["outdoor_co2"],
            self.lines["outdoor_temp"],
        ]
        ax_co2.legend(co2_lines, [line.get_label() for line in co2_lines], loc="lower right", fontsize=7)
        ax_co2.grid(True, alpha=0.3)
        self._style_axis(ax_co2)
        self._style_axis(ax_oat)
        self.plot_axes.append(ax_oat)

        self.lines["electric_kw"], = ax_electric.plot([], [], label="Electric", color="tab:red", alpha=0.85)
        self.lines["gas_kw"], = ax_electric.plot([], [], label="Gas", color="tab:green", alpha=0.85)
        ax_electric.set_xlabel("Hour of Day")
        self._set_panel_title(ax_electric, "Electric and Gas Power")
        self._add_inside_ylabel(ax_electric, "Power (kW)")
        ax_electric.grid(True, alpha=0.3)
        self._style_axis(ax_electric)
        ax_price = ax_electric.twinx()
        self.lines["energy_price"], = ax_price.plot(
            [], [], label="RTP Price", color="tab:purple", linestyle="--", linewidth=1.8, alpha=0.9
        )
        self._add_inside_ylabel(ax_price, "RTP ($/kWh)", side="right")
        power_lines = [
            self.lines["electric_kw"],
            self.lines["gas_kw"],
            self.lines["energy_price"],
        ]
        ax_electric.legend(power_lines, [line.get_label() for line in power_lines], loc="lower right", fontsize=8)
        self._style_axis(ax_price)
        self.plot_axes.append(ax_price)

        kpi_specs = [
            ("energy_cost", "Electric", "tab:red"),
            ("gas_cost", "Gas", "tab:green"),
            ("comfort_penalty", "Thermal $", "darkgoldenrod"),
            ("setpoint_penalty", "Setpoint", "steelblue"),
            ("demand_penalty", "Demand", "tab:orange"),
            ("co2_penalty", "IAQ $", "tab:cyan"),
            ("total_cost", "Total", "black"),
        ]
        for name, label, color in kpi_specs:
            linewidth = 2.0 if name == "total_cost" else 1.4
            alpha = 0.9 if name == "total_cost" else 0.75
            self.lines[name], = ax_kpi_cost.plot(
                [], [], label=label, color=color, linewidth=linewidth, alpha=alpha
            )
        ax_kpi_cost.set_xlabel("Hour of Day")
        self._set_panel_title(ax_kpi_cost, "Reward KPI Costs")
        self._add_inside_ylabel(ax_kpi_cost, "Cost / Penalty")
        ax_kpi_cost.legend(loc="lower right", fontsize=7, ncol=2)
        ax_kpi_cost.grid(True, alpha=0.3)
        self._style_axis(ax_kpi_cost)

        self.lines["reward"], = ax_reward.plot([], [], label="Step Reward", color="tab:purple")
        ax_reward.set_xlabel("Hour of Day")
        self._set_panel_title(ax_reward, "Step Reward")
        self._add_inside_ylabel(ax_reward, "Step Reward")
        ax_reward.legend(loc="lower right", fontsize=8)
        ax_reward.grid(True, alpha=0.3)
        self._style_axis(ax_reward)

        # x-axis ticks show actual clock hour-of-day (see _hour_of_day_formatter), not
        # raw elapsed hours from 0; only meaningful for episode_scope="current".
        formatter = self.FuncFormatter(self._hour_of_day_formatter)
        for ax in set(self.plot_axes):
            ax.xaxis.set_major_formatter(formatter)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

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

        # x-axis is elapsed hours since episode start (monotonic; never wraps at
        # midnight), so multi-day episodes plot as one continuous line. Tick labels
        # are converted back to actual clock hour-of-day by _hour_of_day_formatter.
        step = float(log_entry["elapsed_hours"]) if self.episode_scope == "current" else len(self.series["step"]) + 1
        self.series["step"].append(step)

        if self.episode_scope == "current":
            if self._episode_start_hour is None:
                self._episode_start_hour = int(log_entry["hour"])
            month_day = (int(log_entry["month"]), int(log_entry["day"]))
            if self._last_month_day is not None and month_day != self._last_month_day:
                self._add_day_marker(step, month_day[0], month_day[1])
            self._last_month_day = month_day
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
        self.series["energy_price"].append(float(log_entry["energy_price_used"]))
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
        cmd_oa = float(log_entry.get("commanded_oa_mass_flow", np.nan))
        self.series["commanded_oa_mass_flow"].append(cmd_oa)
        # Per-floor commanded overlays (same building-wide command)
        self.series["bottom_floor_airflow_commanded"].append(cmd_oa)
        self.series["mid_floor_airflow_commanded"].append(cmd_oa)
        self.series["top_floor_airflow_commanded"].append(cmd_oa)
        self.series["outdoor_co2"].append(float(log_entry["outdoor_co2"]))
        self.series["people"].append(float(log_entry.get("people", np.nan)))
        self.series["energy_cost"].append(float(log_entry["energy_cost"]))
        self.series["gas_cost"].append(float(log_entry["gas_cost"]))
        self.series["comfort_penalty"].append(float(log_entry["comfort_penalty"]))
        self.series["setpoint_penalty"].append(float(log_entry["setpoint_penalty"]))
        self.series["demand_penalty"].append(float(log_entry["demand_penalty"]))
        self.series["co2_penalty"].append(float(log_entry.get("co2_penalty", 0.0)))
        self.series["total_cost"].append(float(log_entry["total_cost"]))

        co2_values = [
            float(value)
            for key, value in log_entry.items()
            if key.startswith("co2_") and not np.isnan(value)
        ]
        if not co2_values:
            raise RuntimeError("Live plot requested floor CO2, but no zone CO2 values were logged")
        self.series["avg_co2"].append(float(np.mean(co2_values)))
        self.series["bottom_floor_co2"].append(self._avg_log_values(
            log_entry,
            [f"co2_{zone}" for zone in FLOOR_ZONE_GROUPS["bottom"]]
        ))
        self.series["mid_floor_co2"].append(self._avg_log_values(
            log_entry,
            [f"co2_{zone}" for zone in FLOOR_ZONE_GROUPS["mid"]]
        ))
        self.series["top_floor_co2"].append(self._avg_log_values(
            log_entry,
            [f"co2_{zone}" for zone in FLOOR_ZONE_GROUPS["top"]]
        ))

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
        self.fills["avg_ppd"] = self.ax_ppd.fill_between(
            x, self.series["avg_ppd"], color=PPD_FILL_COLOR, alpha=0.35
        )

        # Shade x-regions where RTP price is "high" on the floor temperature panels.
        # y-span (0, 1) is in axes-fraction via get_xaxis_transform(), so this tracks
        # the x-axis only and does not perturb the temperature y-autoscale.
        high_price = np.asarray(self.series["energy_price"]) > HIGH_PRICE_THRESHOLD
        for ax in self.temp_axes:
            fill_key = f"high_price_{id(ax)}"
            self.fills[fill_key] = ax.fill_between(
                x, 0, 1, where=high_price, transform=ax.get_xaxis_transform(),
                color="orange", alpha=0.08, step="post",
            )

        for ax in self.plot_axes:
            ax.relim()
            ax.autoscale_view()
        self.ax_co2.set_ylim(350, 900)

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
    """Live matplotlib dashboard for SAC losses and optional adaptive reward weights.

    When ``show_adaptive_weights`` is True, PPD/CO₂ scales and cost EMAs are drawn
    in the same window as the SAC losses (third row).
    """

    def __init__(self, output_dir, update_every=1, show_losses=True, show_adaptive_weights=False):
        if not show_losses and not show_adaptive_weights:
            raise ValueError("LiveLossPlotter needs show_losses and/or show_adaptive_weights")
        self.update_every = max(1, int(update_every))
        self.show_losses = bool(show_losses)
        self.show_adaptive_weights = bool(show_adaptive_weights)
        self.step_count = 0
        self.weight_step_count = 0
        self._warmup_line = None
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
        self.lines = {}
        self.loss_axes = []
        self.ax_scale = None
        self.ax_ema = None
        self.series = {
            "update": [],
            "actor_loss": [],
            "critic1_loss": [],
            "critic2_loss": [],
            "alpha_loss": [],
            "n_samples": [],
            "comfort_scale": [],
            "co2_scale": [],
            "ema_energy": [],
            "ema_comfort": [],
            "ema_co2": [],
        }
        self._setup_figure()

    def _setup_loss_axis(self, ax, series_name, label, color):
        self.lines[series_name], = ax.plot([], [], label=label, color=color, linewidth=1.8)
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Training Update", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        self.loss_axes.append(ax)

    def _setup_figure(self):
        title_bits = []
        if self.show_losses:
            title_bits.append("SAC Losses")
        if self.show_adaptive_weights:
            title_bits.append("Adaptive Weights")
        title = " / ".join(title_bits)

        if self.show_losses and self.show_adaptive_weights:
            self.fig = self.plt.figure(figsize=(12, 10), num="SAC Losses + Adaptive Weights")
            gs = self.fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.35, wspace=0.25)
            self._setup_loss_axis(self.fig.add_subplot(gs[0, 0]), "actor_loss", "Actor Loss", "tab:purple")
            self._setup_loss_axis(self.fig.add_subplot(gs[0, 1]), "critic1_loss", "Critic 1 Loss", "tab:red")
            self._setup_loss_axis(self.fig.add_subplot(gs[1, 0]), "critic2_loss", "Critic 2 Loss", "tab:orange")
            self._setup_loss_axis(self.fig.add_subplot(gs[1, 1]), "alpha_loss", "Alpha Loss", "tab:blue")
            self.ax_scale = self.fig.add_subplot(gs[2, 0])
            self.ax_ema = self.fig.add_subplot(gs[2, 1])
            self._setup_weight_axes()
        elif self.show_losses:
            self.fig, axes = self.plt.subplots(2, 2, figsize=(10, 7), num="SAC Training Losses")
            ax_actor, ax_critic1, ax_critic2, ax_alpha = axes.ravel()
            self._setup_loss_axis(ax_actor, "actor_loss", "Actor Loss", "tab:purple")
            self._setup_loss_axis(ax_critic1, "critic1_loss", "Critic 1 Loss", "tab:red")
            self._setup_loss_axis(ax_critic2, "critic2_loss", "Critic 2 Loss", "tab:orange")
            self._setup_loss_axis(ax_alpha, "alpha_loss", "Alpha Loss", "tab:blue")
            self.fig.tight_layout(pad=0.8, w_pad=0.5, h_pad=0.7)
        else:
            self.fig, (self.ax_scale, self.ax_ema) = self.plt.subplots(
                1, 2, figsize=(12, 4.5), num="Adaptive Reward Weights"
            )
            self._setup_weight_axes()
            self.fig.tight_layout(pad=0.9, w_pad=0.5)

        self.fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _setup_weight_axes(self):
        self.lines["comfort_scale"], = self.ax_scale.plot(
            [], [], label="PPD scale", color="darkgoldenrod", linewidth=2.0
        )
        self.lines["co2_scale"], = self.ax_scale.plot(
            [], [], label="CO₂ scale", color="tab:cyan", linewidth=2.0
        )
        self.ax_scale.set_ylabel("Adaptive scale")
        self.ax_scale.set_xlabel("Adaptive samples")
        self.ax_scale.set_ylim(-0.05, 1.05)
        self.ax_scale.legend(loc="best", fontsize=8)
        self.ax_scale.grid(True, alpha=0.3)
        self.ax_scale.set_title("PPD / CO₂ scales (converge after warmup)", fontsize=9, fontweight="bold")

        self.lines["ema_energy"], = self.ax_ema.plot(
            [], [], label="EMA(energy $)", color="tab:red", linewidth=1.8
        )
        self.lines["ema_comfort"], = self.ax_ema.plot(
            [], [], label="EMA(PPD raw $)", color="darkgoldenrod", linewidth=1.8, alpha=0.9
        )
        self.lines["ema_co2"], = self.ax_ema.plot(
            [], [], label="EMA(CO₂ raw $)", color="tab:cyan", linewidth=1.8, alpha=0.9
        )
        self.ax_ema.set_xlabel("Adaptive samples")
        self.ax_ema.set_ylabel("EMA cost ($)")
        self.ax_ema.legend(loc="best", fontsize=8)
        self.ax_ema.grid(True, alpha=0.3)
        self.ax_ema.set_title("Cost EMAs used for rebalancing", fontsize=9, fontweight="bold")

    def update(self, update_no, losses):
        """Append SAC training losses (called on each agent update)."""
        if not self.show_losses:
            return
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
        for name in ("actor_loss", "critic1_loss", "critic2_loss", "alpha_loss"):
            self.lines[name].set_data(x, self.series[name])
        for ax in self.loss_axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update_weights(self, log_entry):
        """Append adaptive PPD/CO₂ scales from a sim log entry."""
        if not self.show_adaptive_weights:
            return
        self.weight_step_count += 1
        n = int(log_entry.get("adaptive_n_samples", 0) or 0)
        x_val = n if n > 0 else self.weight_step_count
        self.series["n_samples"].append(x_val)
        self.series["comfort_scale"].append(float(log_entry.get("adaptive_comfort_scale", 1.0)))
        self.series["co2_scale"].append(float(log_entry.get("adaptive_co2_scale", 1.0)))
        ema_e = log_entry.get("adaptive_ema_energy")
        ema_c = log_entry.get("adaptive_ema_comfort")
        ema_co2 = log_entry.get("adaptive_ema_co2")
        self.series["ema_energy"].append(float(ema_e) if ema_e is not None else np.nan)
        self.series["ema_comfort"].append(float(ema_c) if ema_c is not None else np.nan)
        self.series["ema_co2"].append(float(ema_co2) if ema_co2 is not None else np.nan)

        active = bool(log_entry.get("adaptive_balancing_active", False))
        if self._warmup_line is None and active and n > 0:
            self._warmup_line = self.ax_scale.axvline(
                n, color="gray", linestyle="--", alpha=0.7, linewidth=1.2, label="Warmup end"
            )
            self.ax_ema.axvline(n, color="gray", linestyle="--", alpha=0.7, linewidth=1.2)
            self.ax_scale.legend(loc="best", fontsize=8)

        if self.weight_step_count % self.update_every != 0:
            return

        x = self.series["n_samples"]
        for name in ("comfort_scale", "co2_scale", "ema_energy", "ema_comfort", "ema_co2"):
            self.lines[name].set_data(x, self.series[name])
        self.ax_scale.relim()
        self.ax_scale.autoscale_view()
        self.ax_scale.set_ylim(-0.05, 1.05)
        self.ax_ema.relim()
        self.ax_ema.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def finish(self, output_dir, hold=False):
        if self.show_losses and self.show_adaptive_weights:
            name = "sac_losses_and_adaptive_weights.png"
        elif self.show_adaptive_weights:
            name = "adaptive_reward_weights.png"
        else:
            name = "sac_training_losses.png"
        path = Path(output_dir) / name
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot snapshot to {path}")
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class LiveEpisodeDollarKPIPlotter:
    """Per-episode dollar costs plus physical KPIs (kWh, PPD, CO₂, people).

    Dollar panel: absolute $ (converts from $/m² when needed).
    Physical panel: summed electric/gas kWh and episode-mean PPD, CO₂, people.
    """

    def __init__(self, output_dir, floor_area_m2=None):
        self.floor_area_m2 = float(floor_area_m2) if floor_area_m2 and floor_area_m2 > 0 else None
        self._episode_sums = None
        self._episode_phys = None
        self._current_episode = None
        self._norm_mode = None
        mpl_cache_dir = Path(output_dir) / ".matplotlib-cache"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError as exc:
            raise RuntimeError(
                "Episode KPI plotting requires matplotlib. Install it in the active environment."
            ) from exc

        self.plt = plt
        self.mticker = mticker
        plt.ion()
        self.fig = plt.figure(figsize=(12, 11), num="Episode KPIs")
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.1], hspace=0.38, wspace=0.28)
        self.ax_bars = self.fig.add_subplot(gs[0, :])
        self.ax_lines = self.fig.add_subplot(gs[1, :])
        self.ax_energy = self.fig.add_subplot(gs[2, 0])
        self.ax_comfort = self.fig.add_subplot(gs[2, 1])
        self.ax_gas = self.ax_energy.twinx()
        self.ax_co2 = self.ax_comfort.twinx()
        self.fig.suptitle("Episode KPIs", fontsize=12, fontweight="bold")

        self.kpi_keys = (
            ("energy_cost", "Electric $", "tab:red"),
            ("gas_cost", "Gas $", "tab:green"),
            ("comfort_penalty", "Thermal $", "darkgoldenrod"),
            ("co2_penalty", "IAQ $", "tab:cyan"),
            ("total_cost", "Total $", "black"),
        )
        self.phys_keys = (
            "elec_kwh",
            "gas_kwh",
            "mean_ppd",
            "mean_co2",
            "mean_people",
        )
        self.series = {"episode": []}
        for key, _, _ in self.kpi_keys:
            self.series[key] = []
        for key in self.phys_keys:
            self.series[key] = []
        self.lines = {}
        self.phys_lines = {}
        self.bars = None
        self._setup_axes()

    def _setup_axes(self):
        self.ax_bars.set_ylabel("Episode cost ($)")
        self.ax_bars.set_title(
            "Dollar costs (stacked) — gas/IAQ $ often tiny vs electric/thermal",
            fontsize=9, fontweight="bold",
        )
        self.ax_bars.grid(True, axis="y", alpha=0.3)

        for key, label, color in self.kpi_keys:
            linewidth = 2.2 if key == "total_cost" else 1.6
            self.lines[key], = self.ax_lines.plot(
                [], [], label=label, color=color, linewidth=linewidth, marker="o", markersize=4
            )
        self.ax_lines.set_ylabel("Episode cost ($)")
        self.ax_lines.set_title("Dollar KPI trends", fontsize=9, fontweight="bold")
        self.ax_lines.legend(loc="best", fontsize=8, ncol=3)
        self.ax_lines.grid(True, alpha=0.3)
        self.ax_lines.xaxis.set_major_locator(self.mticker.MaxNLocator(integer=True))

        self.phys_lines["elec_kwh"], = self.ax_energy.plot(
            [], [], label="Electric kWh", color="tab:red", linewidth=2.0, marker="o", markersize=4
        )
        self.phys_lines["gas_kwh"], = self.ax_gas.plot(
            [], [], label="Gas kWh", color="tab:green", linewidth=2.0, marker="s", markersize=4
        )
        self.ax_energy.set_xlabel("Episode")
        self.ax_energy.set_ylabel("Electric (kWh)", color="tab:red")
        self.ax_gas.set_ylabel("Gas (kWh)", color="tab:green")
        self.ax_energy.set_title("Physical energy use", fontsize=9, fontweight="bold")
        self.ax_energy.grid(True, alpha=0.3)
        self.ax_energy.tick_params(axis="y", labelcolor="tab:red")
        self.ax_gas.tick_params(axis="y", labelcolor="tab:green")
        energy_handles = [self.phys_lines["elec_kwh"], self.phys_lines["gas_kwh"]]
        self.ax_energy.legend(energy_handles, [h.get_label() for h in energy_handles], loc="best", fontsize=8)

        self.phys_lines["mean_ppd"], = self.ax_comfort.plot(
            [], [], label="Mean PPD (%)", color="darkgoldenrod", linewidth=2.0, marker="o", markersize=4
        )
        self.phys_lines["mean_people"], = self.ax_comfort.plot(
            [], [], label="Mean people", color="tab:olive", linewidth=1.6, marker="^", markersize=4, alpha=0.85
        )
        self.phys_lines["mean_co2"], = self.ax_co2.plot(
            [], [], label="Mean zone CO₂ (ppm)", color="tab:cyan", linewidth=2.0, marker="s", markersize=4
        )
        self.ax_comfort.set_xlabel("Episode")
        self.ax_comfort.set_ylabel("PPD (%) / people")
        self.ax_co2.set_ylabel("CO₂ (ppm)", color="tab:cyan")
        self.ax_comfort.set_title("Physical comfort / IAQ / occupancy", fontsize=9, fontweight="bold")
        self.ax_comfort.grid(True, alpha=0.3)
        self.ax_co2.tick_params(axis="y", labelcolor="tab:cyan")
        comfort_handles = [
            self.phys_lines["mean_ppd"],
            self.phys_lines["mean_people"],
            self.phys_lines["mean_co2"],
        ]
        self.ax_comfort.legend(
            comfort_handles, [h.get_label() for h in comfort_handles], loc="best", fontsize=7
        )
        for ax in (self.ax_energy, self.ax_comfort):
            ax.xaxis.set_major_locator(self.mticker.MaxNLocator(integer=True))

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _to_absolute_dollars(self, log_entry, key):
        """Convert a logged cost term to absolute $ for the timestep."""
        value = float(log_entry.get(key, 0.0) or 0.0)
        mode = str(log_entry.get("cost_normalization_mode", "absolute")).lower()
        self._norm_mode = mode
        if mode == "per_m2":
            area = self.floor_area_m2
            if area is None:
                factor = log_entry.get("cost_normalization_factor")
                if factor:
                    area = 1.0 / float(factor)
            if area:
                return value * float(area)
        return value

    @staticmethod
    def _mean_log_prefix(log_entry, prefix):
        values = [
            float(v)
            for k, v in log_entry.items()
            if str(k).startswith(prefix) and v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        return float(np.mean(values)) if values else np.nan

    def _empty_phys(self):
        return {
            "elec_kwh": 0.0,
            "gas_kwh": 0.0,
            "ppd_sum": 0.0,
            "ppd_n": 0,
            "co2_sum": 0.0,
            "co2_n": 0,
            "people_sum": 0.0,
            "people_n": 0,
        }

    def _ensure_episode(self, episode):
        episode = int(episode)
        if self._current_episode is None:
            self._current_episode = episode
            self._episode_sums = {key: 0.0 for key, _, _ in self.kpi_keys}
            self._episode_phys = self._empty_phys()
            return
        if episode != self._current_episode:
            self._finalize_current_episode()
            self._current_episode = episode
            self._episode_sums = {key: 0.0 for key, _, _ in self.kpi_keys}
            self._episode_phys = self._empty_phys()

    def update_step(self, log_entry):
        """Accumulate dollar and physical KPIs for the current episode timestep."""
        episode = int(log_entry["episode"])
        self._ensure_episode(episode)
        for key, _, _ in self.kpi_keys:
            self._episode_sums[key] += self._to_absolute_dollars(log_entry, key)

        phys = self._episode_phys
        phys["elec_kwh"] += float(log_entry.get("energy_kwh", 0.0) or 0.0)
        phys["gas_kwh"] += float(log_entry.get("gas_kwh", 0.0) or 0.0)

        ppd = self._mean_log_prefix(log_entry, "ppd_")
        if not np.isnan(ppd):
            phys["ppd_sum"] += ppd
            phys["ppd_n"] += 1

        co2 = self._mean_log_prefix(log_entry, "co2_")
        if not np.isnan(co2):
            phys["co2_sum"] += co2
            phys["co2_n"] += 1

        people = log_entry.get("people", np.nan)
        if people is not None and not (isinstance(people, float) and np.isnan(people)):
            phys["people_sum"] += float(people)
            phys["people_n"] += 1

    def finalize_episode(self, episode=None):
        """Push the completed episode totals onto the chart."""
        if episode is not None and self._current_episode is None:
            self._current_episode = int(episode)
        self._finalize_current_episode()

    def _finalize_current_episode(self):
        if self._current_episode is None or self._episode_sums is None:
            return

        self.series["episode"].append(int(self._current_episode))
        for key, _, _ in self.kpi_keys:
            self.series[key].append(float(self._episode_sums[key]))

        phys = self._episode_phys or self._empty_phys()
        self.series["elec_kwh"].append(float(phys["elec_kwh"]))
        self.series["gas_kwh"].append(float(phys["gas_kwh"]))
        self.series["mean_ppd"].append(
            phys["ppd_sum"] / phys["ppd_n"] if phys["ppd_n"] else np.nan
        )
        self.series["mean_co2"].append(
            phys["co2_sum"] / phys["co2_n"] if phys["co2_n"] else np.nan
        )
        self.series["mean_people"].append(
            phys["people_sum"] / phys["people_n"] if phys["people_n"] else np.nan
        )

        self._redraw()
        self._current_episode = None
        self._episode_sums = None
        self._episode_phys = None

    def _redraw(self):
        episodes = self.series["episode"]
        if not episodes:
            return

        self.ax_bars.clear()
        self.ax_bars.set_ylabel("Episode cost ($)")
        self.ax_bars.set_title(
            "Dollar costs (stacked) — gas/IAQ $ often tiny vs electric/thermal",
            fontsize=9, fontweight="bold",
        )
        self.ax_bars.grid(True, axis="y", alpha=0.3)

        x = np.asarray(episodes, dtype=float)
        bottom = np.zeros(len(episodes), dtype=float)
        width = 0.65
        for key, label, color in self.kpi_keys:
            if key == "total_cost":
                continue
            vals = np.asarray(self.series[key], dtype=float)
            self.ax_bars.bar(
                x, vals, width=width, bottom=bottom, label=label, color=color, alpha=0.85
            )
            bottom = bottom + vals
        self.ax_bars.plot(
            x, self.series["total_cost"], color="black", marker="o",
            linestyle="None", markersize=5, label="Total $", zorder=5
        )
        self.ax_bars.legend(loc="best", fontsize=8, ncol=3)
        self.ax_bars.xaxis.set_major_locator(self.mticker.MaxNLocator(integer=True))

        for key, _, _ in self.kpi_keys:
            self.lines[key].set_data(episodes, self.series[key])
        self.ax_lines.relim()
        self.ax_lines.autoscale_view()
        self.ax_lines.xaxis.set_major_locator(self.mticker.MaxNLocator(integer=True))

        self.phys_lines["elec_kwh"].set_data(episodes, self.series["elec_kwh"])
        self.phys_lines["gas_kwh"].set_data(episodes, self.series["gas_kwh"])
        self.phys_lines["mean_ppd"].set_data(episodes, self.series["mean_ppd"])
        self.phys_lines["mean_people"].set_data(episodes, self.series["mean_people"])
        self.phys_lines["mean_co2"].set_data(episodes, self.series["mean_co2"])
        for ax in (self.ax_energy, self.ax_gas, self.ax_comfort, self.ax_co2):
            ax.relim()
            ax.autoscale_view()
        for ax in (self.ax_energy, self.ax_comfort):
            ax.xaxis.set_major_locator(self.mticker.MaxNLocator(integer=True))

        unit_note = ""
        if self._norm_mode == "per_m2" and self.floor_area_m2:
            unit_note = f" — $ from $/m² × {self.floor_area_m2:.0f} m²"
        self.fig.suptitle(f"Episode KPIs{unit_note}", fontsize=12, fontweight="bold")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def finish(self, output_dir, hold=False):
        if self._episode_sums is not None and self._current_episode is not None:
            self._finalize_current_episode()
        path = Path(output_dir) / "episode_dollar_kpis.png"
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved episode KPI plot to {path}")
        csv_path = Path(output_dir) / "episode_dollar_kpis.csv"
        rows = []
        for i, ep in enumerate(self.series["episode"]):
            row = {"episode": ep}
            for key, _, _ in self.kpi_keys:
                row[key] = self.series[key][i]
            for key in self.phys_keys:
                row[key] = self.series[key][i]
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved episode KPI table to {csv_path}")
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.pause(0.001)


class AdaptiveCostBalancer:
    """Online scales for thermal/IAQ $ so their running means track energy $.

    Tracks EMA of energy (electric+gas), raw PPD productivity $, and raw CO₂
    productivity $. After ``min_samples``, sets:

        scale_c = clip(comfort_target_ratio * ema_energy / ema_comfort, …)
        scale_co2 = clip(co2_target_ratio * ema_energy / ema_co2, …)

    With ``weight_max: 1.0`` (default) this only scales productivity *down* when
    it dominates energy — it never amplifies above the static ``comfort_weight`` /
    ``co2_weight``. Warmup leaves scales at ``initial_scale`` (default 1.0).
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get('enabled', False))
        self.ema_alpha = float(cfg.get('ema_alpha', 0.01))
        self.min_samples = int(cfg.get('min_samples', 100))
        self.comfort_target_ratio = float(cfg.get('comfort_target_ratio', 1.0))
        self.co2_target_ratio = float(cfg.get('co2_target_ratio', 1.0))
        self.weight_min = float(cfg.get('weight_min', 0.0))
        self.weight_max = float(cfg.get('weight_max', 1.0))
        self.initial_scale = float(cfg.get('initial_scale', 1.0))
        self.min_mean = float(cfg.get('min_mean', 1e-12))
        self.ema_energy = None
        self.ema_comfort = None
        self.ema_co2 = None
        self.n_samples = 0
        self.comfort_scale = self.initial_scale
        self.co2_scale = self.initial_scale

    @staticmethod
    def _ema(prev, value, alpha):
        if prev is None:
            return float(value)
        return (1.0 - alpha) * float(prev) + alpha * float(value)

    def update_and_get_scales(self, energy_ref, comfort_raw, co2_raw):
        """Update running means; return dict of scales and diagnostics."""
        if not self.enabled:
            return {
                'comfort': 1.0,
                'co2': 1.0,
                'ema_energy': None,
                'ema_comfort': None,
                'ema_co2': None,
                'n_samples': 0,
                'active': False,
            }

        energy_ref = max(0.0, float(energy_ref))
        comfort_raw = max(0.0, float(comfort_raw))
        co2_raw = max(0.0, float(co2_raw))

        self.ema_energy = self._ema(self.ema_energy, energy_ref, self.ema_alpha)
        self.ema_comfort = self._ema(self.ema_comfort, comfort_raw, self.ema_alpha)
        self.ema_co2 = self._ema(self.ema_co2, co2_raw, self.ema_alpha)
        self.n_samples += 1

        active = self.n_samples >= self.min_samples
        if active:
            if self.ema_comfort is not None and self.ema_comfort > self.min_mean:
                self.comfort_scale = float(np.clip(
                    self.comfort_target_ratio * self.ema_energy / self.ema_comfort,
                    self.weight_min,
                    self.weight_max,
                ))
            if self.ema_co2 is not None and self.ema_co2 > self.min_mean:
                self.co2_scale = float(np.clip(
                    self.co2_target_ratio * self.ema_energy / self.ema_co2,
                    self.weight_min,
                    self.weight_max,
                ))

        return {
            'comfort': self.comfort_scale if active else self.initial_scale,
            'co2': self.co2_scale if active else self.initial_scale,
            'ema_energy': self.ema_energy,
            'ema_comfort': self.ema_comfort,
            'ema_co2': self.ema_co2,
            'n_samples': self.n_samples,
            'active': active,
        }

    def state_dict(self):
        """Serialize running stats for checkpoint resume."""
        return {
            'ema_energy': self.ema_energy,
            'ema_comfort': self.ema_comfort,
            'ema_co2': self.ema_co2,
            'n_samples': int(self.n_samples),
            'comfort_scale': float(self.comfort_scale),
            'co2_scale': float(self.co2_scale),
        }

    def load_state_dict(self, state):
        """Restore running stats from a checkpoint (ignores None / empty)."""
        if not state:
            return False
        self.ema_energy = state.get('ema_energy', None)
        self.ema_comfort = state.get('ema_comfort', None)
        self.ema_co2 = state.get('ema_co2', None)
        self.n_samples = int(state.get('n_samples', 0) or 0)
        if 'comfort_scale' in state and state['comfort_scale'] is not None:
            self.comfort_scale = float(state['comfort_scale'])
        if 'co2_scale' in state and state['co2_scale'] is not None:
            self.co2_scale = float(state['co2_scale'])
        return True


class HVACEnvironment:
    """Environment wrapper for EnergyPlus HVAC control with RL agent."""
    
    def __init__(self, api, state, config_path=None, simulation_overrides=None):
        self.api = api
        self.state = state
        
        # Load HVAC configuration
        self.hvac_config = get_hvac_config(config_path)
        if simulation_overrides:
            sim_cfg = self.hvac_config.config.setdefault('simulation', {})
            for key, value in simulation_overrides.items():
                if value is not None:
                    sim_cfg[key] = value
        
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

        # Online PPD/CO2 reward scales vs observed energy $ (see AdaptiveCostBalancer)
        reward_cfg = self.hvac_config.config.get('reward') or {}
        self.adaptive_cost_balancer = AdaptiveCostBalancer(
            reward_cfg.get('adaptive_balancing')
        )
        self._last_reward_components = None

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

        # Occupancy events: config-driven per-zone occupancy spikes/drops applied on
        # top of the model's normal People schedule (see apply_occupancy_events_for_current_timestep).
        self.occupancy_events_cfg = self.hvac_config.config.get('occupancy_events', {}) or {}
        # zone_name -> calibrated design occupant count (Area/Person "full occupancy" level),
        # lazily learned at runtime from the natural (un-overridden) schedule-driven occupancy.
        self._occupancy_design_people = {}

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
            zone_handles['co2'] = exchange.get_variable_handle(
                self.state, "Zone Air CO2 Concentration", zone_name
            )
            zone_handles['people'] = exchange.get_variable_handle(
                self.state, "Zone People Occupant Count", zone_name
            )
            # Actuator to override occupant count for occupancy_events (component name
            # matches the People object's Name field, which is the zone name in this model).
            zone_handles['people_actuator'] = exchange.get_actuator_handle(
                self.state, "People", "Number of People", zone_name
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

        self.handles['ahu_oa'] = {}
        for floor, info in AHU_OA_CONTROLS.items():
            oa_handles = {}
            airloop_name = info['airloop']
            controller_name = info['controller']
            oa_handles['commanded_mass_flow'] = exchange.get_actuator_handle(
                self.state, "Outdoor Air Controller", "Air Mass Flow Rate", controller_name
            )
            oa_handles['mass_flow'] = exchange.get_variable_handle(
                self.state, "Air System Outdoor Air Mass Flow Rate", airloop_name
            )
            oa_handles['mech_vent_request'] = exchange.get_variable_handle(
                self.state,
                "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate",
                airloop_name,
            )
            if oa_handles['commanded_mass_flow'] <= 0:
                raise RuntimeError(
                    f"EnergyPlus handle invalid for AHU OA controller actuator '{controller_name}'. "
                    "Check eplusout.edd for Outdoor Air Controller / Air Mass Flow Rate actuator names."
                )
            if oa_handles['mass_flow'] <= 0:
                raise RuntimeError(
                    f"EnergyPlus handle invalid for AHU OA mass-flow variable on '{airloop_name}'. "
                    "Declare 'Air System Outdoor Air Mass Flow Rate' as an Output:Variable."
                )
            self.handles['ahu_oa'][floor] = oa_handles

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

        # Occupancy events: natural (schedule-driven) occupancy fraction, read independently
        # of any People actuator override — requires Output:Variable injected by create_custom_idf.
        occ_schedule_name = self.occupancy_events_cfg.get('schedule_name')
        self.handles['occupancy_schedule'] = (
            exchange.get_variable_handle(self.state, "Schedule Value", occ_schedule_name)
            if occ_schedule_name else -1
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

    def _resolve_occupancy_event_zones(self, zone_refs):
        """Expand an occupancy event's `zones` list. Entries matching a FLOOR_ZONE_GROUPS
        key ("bottom"/"mid"/"top") expand to that floor's zones; anything else is treated
        as an exact zone name. Unknown zone names are silently dropped."""
        resolved = []
        for ref in zone_refs:
            for zone_name in FLOOR_ZONE_GROUPS.get(ref, [ref]):
                if zone_name in self.handles.get('zones', {}):
                    resolved.append(zone_name)
        return resolved

    @staticmethod
    def _occupancy_event_in_date_range(event, month, day):
        """Optional start/end month-day restriction on an event; defaults to active on
        every simulated day when not given."""
        start_month = event.get('start_month')
        end_month = event.get('end_month')
        if start_month is None or end_month is None:
            return True
        start = (start_month, event.get('start_day') or 1)
        end = (end_month, event.get('end_day') or 31)
        return start <= (month, day) <= end

    def apply_occupancy_events_for_current_timestep(self):
        """Override per-zone People counts for configured occupancy spike/drop events
        (config: occupancy_events). Zones/hours with no active event have their
        actuator reset so EnergyPlus's normal People schedule (and CO2 generation)
        drives occupancy as usual.

        Each event's delta (e.g. +0.4, -0.3) scales the *natural* scheduled occupancy
        for that hour, read from occupancy_events.schedule_name's "Schedule Value"
        output — which reflects the real schedule regardless of any actuator override,
        so events compose correctly with the model's normal daily occupancy pattern.
        The absolute per-zone design occupant count (Area/Person "full occupancy"
        level) isn't directly queryable via the EnergyPlus Python API, so it's
        calibrated lazily from the natural (un-overridden) occupant count the first
        time each zone is observed with no active event, using zone_time_step_number
        > 1 to guarantee the previous timestep's report is from the same clock hour
        (same schedule fraction) and therefore comparable, not lagged into a
        different hour's occupancy level.
        """
        if not self.handles_initialized:
            return
        cfg = self.occupancy_events_cfg
        if not cfg.get('enabled', False):
            return
        events = cfg.get('events', [])
        if not events:
            return

        exchange = self.api.exchange
        sched_handle = self.handles.get('occupancy_schedule', -1)
        if sched_handle is None or sched_handle <= 0:
            return
        schedule_fraction = exchange.get_variable_value(self.state, sched_handle)

        hour = exchange.hour(self.state)
        month = exchange.month(self.state)
        day = exchange.day_of_month(self.state)
        zone_time_step = exchange.zone_time_step_number(self.state)

        active_multiplier = {}
        for event in events:
            if hour not in event.get('hours', []):
                continue
            if not self._occupancy_event_in_date_range(event, month, day):
                continue
            multiplier = 1.0 + float(event.get('delta', 0.0))
            for zone_name in self._resolve_occupancy_event_zones(event.get('zones', [])):
                active_multiplier[zone_name] = active_multiplier.get(zone_name, 1.0) * multiplier

        for zone_name, zhandles in self.handles['zones'].items():
            actuator_handle = zhandles.get('people_actuator', -1)
            if actuator_handle is None or actuator_handle <= 0:
                continue

            if zone_name in active_multiplier:
                design_people = self._occupancy_design_people.get(zone_name)
                if design_people is None:
                    # Not yet calibrated (e.g. event fires before any normal occupied
                    # hour has been observed) — leave the natural schedule in effect.
                    continue
                new_count = max(0.0, design_people * schedule_fraction * active_multiplier[zone_name])
                exchange.set_actuator_value(self.state, actuator_handle, new_count)
            else:
                exchange.reset_actuator(self.state, actuator_handle)
                if (
                    zone_name not in self._occupancy_design_people
                    and zone_time_step > 1
                    and schedule_fraction > 1e-3
                ):
                    natural_count = exchange.get_variable_value(self.state, zhandles['people'])
                    self._occupancy_design_people[zone_name] = natural_count / schedule_fraction

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

    def _zone_temp_value(self, zone_name):
        """Read one zone mean air temperature (°C) from EnergyPlus."""
        zhandles = self.handles['zones'].get(zone_name)
        if not zhandles or zhandles.get('temp', -1) <= 0:
            raise RuntimeError(f"Missing temperature handle for zone '{zone_name}'")
        return float(self.api.exchange.get_variable_value(self.state, zhandles['temp']))

    def _get_zone_temp_features(self):
        """Build zone-temperature state features from config zone_temps.mode.

        modes:
          zones          — one temp per controlled zone (15)
          perimeter_core — core_bottom/mid/top + mean perimeter temp per floor (6)
          floor          — mean temp of all zones on each floor: bottom, mid, top (3)
        """
        mode = self.hvac_config.config['state_space']['zone_temps'].get('mode', 'zones')
        expected = self.hvac_config.config['state_space']['zone_temps']['count']

        if mode == 'zones':
            zone_temps = [
                self._zone_temp_value(zone_name)
                for zone_name in list(self.handles['zones'].keys())[:expected]
            ]
        elif mode == 'perimeter_core':
            zone_temps = []
            for floor in FLOOR_ORDER:
                zone_temps.append(self._zone_temp_value(CORE_ZONE_BY_FLOOR[floor]))
            for floor in FLOOR_ORDER:
                perim_temps = [self._zone_temp_value(z) for z in PERIMETER_ZONE_GROUPS[floor]]
                zone_temps.append(float(np.mean(perim_temps)))
        elif mode == 'floor':
            zone_temps = []
            for floor in FLOOR_ORDER:
                floor_temps = [self._zone_temp_value(z) for z in FLOOR_ZONE_GROUPS[floor]]
                zone_temps.append(float(np.mean(floor_temps)))
        else:
            raise RuntimeError(f"Unsupported state_space.zone_temps.mode: {mode!r}")

        if len(zone_temps) != expected:
            raise RuntimeError(
                f"zone_temps mode={mode!r} produced {len(zone_temps)} features, "
                f"expected count={expected}"
            )
        return zone_temps

    def get_current_state(self):
        """Get current environment state from EnergyPlus."""
        exchange = self.api.exchange
        
        # Zone temperatures: per-zone, perimeter_core, or floor averages
        zone_temps = self._get_zone_temp_features()
        zone_count = len(zone_temps)

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
        current_year = 2023
        if hasattr(exchange, 'year'):
            year_val = exchange.year(self.state)
            if year_val and int(year_val) > 0:
                current_year = int(year_val)

        # Day of week: Monday=0 .. Sunday=6, normalized to [0, 1]
        day_of_week = datetime(current_year, current_month, current_day).weekday() / 6.0

        # Normalize to [0, 1] range
        time_features = [
            current_hour / 24.0,  # Hour of day
            day_of_week,         # Day of week
            current_month / 12.0  # Month of year
        ]
        
        # Optional previous action features. Set previous_actions.count: 0 to exclude
        # these from the state while still using the same action space.
        prev_cfg = self.hvac_config.config['state_space'].get('previous_actions', {})
        prev_count = prev_cfg.get('count', 0)
        if prev_count <= 0:
            prev_action = []
        elif self.current_action is not None:
            prev_action = list(self.current_action)[:prev_count]
        else:
            prev_action = prev_cfg.get('initial_value')
            if prev_action is None:
                raise RuntimeError(
                    "state_space.previous_actions.initial_value is required "
                    "when state_space.previous_actions.count is greater than zero"
                )
            if len(prev_action) != prev_count:
                raise RuntimeError(
                    "state_space.previous_actions.initial_value length must match "
                    "state_space.previous_actions.count"
                )
        
        # Outdoor CO2 (ppm): from override, CSV lookup, or explicit config constant
        outdoor_co2 = self.get_outdoor_co2_ppm(current_month, current_day, current_hour)
        co2_outdoor = [float(outdoor_co2)]
        
        # Average zone return-air CO2 (ppm) per floor
        floor_co2_list = []
        floor_co2_count = self.hvac_config.config['state_space'].get('floor_co2_ppm', {}).get('count', 0)
        for floor in FLOOR_ORDER[:floor_co2_count]:
            co2_values = []
            for zone_name in FLOOR_ZONE_GROUPS[floor]:
                zhandles = self.handles['zones'].get(zone_name)
                if zhandles and zhandles.get('co2', -1) > 0:
                    co2_values.append(exchange.get_variable_value(self.state, zhandles['co2']))
            floor_co2_list.append(float(np.mean(co2_values)) if co2_values else 0.0)

        # Electricity RTP ($/kWh): current hour + 1-hour-ahead (published hour-ahead price)
        rtp_price_list = []
        rtp_count = self.hvac_config.config['state_space'].get('rtp_price', {}).get('count', 0)
        if rtp_count > 0:
            reward_config = self.hvac_config.config['reward']
            price_now = get_realtime_price(current_month, current_day, current_hour, reward_config)
            rtp_price_list.append(float(price_now))
            if rtp_count > 1:
                ahead = datetime(current_year, current_month, current_day, current_hour) + timedelta(hours=1)
                price_ahead = get_realtime_price(ahead.month, ahead.day, ahead.hour, reward_config)
                rtp_price_list.append(float(price_ahead))
            while len(rtp_price_list) < rtp_count:
                rtp_price_list.append(rtp_price_list[-1] if rtp_price_list else 0.0)
            rtp_price_list = rtp_price_list[:rtp_count]
        
        # Ensure all are lists (not numpy arrays) before concatenation
        zone_temps = list(zone_temps)
        current_weather = list(current_weather)
        forecast_weather = list(forecast_weather)
        past_weather = list(past_weather)
        time_features = list(time_features)
        prev_action = list(prev_action)

        # Combine all features (should match config state_size): temps, weather, forecast,
        # past weather (optional), time, optional prev_action, outdoor_co2, floor_co2, rtp
        all_features = (
            zone_temps + current_weather + forecast_weather + past_weather
            + time_features + prev_action + co2_outdoor + floor_co2_list + rtp_price_list
        )
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
        # Bounds come from config. For economizer-scale control, the max should be
        # near the largest autosized OA controller maximum: about 4.94 m3/s * 1.2 kg/m3 ~= 6 kg/s.
        oa_mass_flow = max(
            0.0,
            scale(action[2], action_bounds['airflow_multiplier'][0], action_bounds['airflow_multiplier'][1]),
        )
        
        # Calculate setpoints
        heating_sp = self.base_temp + sp_offset - deadband/2
        cooling_sp = self.base_temp + sp_offset + deadband/2
        
        # Apply to all zones
        for _, handles in self.handles['zones'].items():
            if handles['heating_sp'] > 0:
                exchange.set_actuator_value(self.state, handles['heating_sp'], heating_sp)
            if handles['cooling_sp'] > 0:
                exchange.set_actuator_value(self.state, handles['cooling_sp'], cooling_sp)

        for handles in self.handles.get('ahu_oa', {}).values():
            if handles['commanded_mass_flow'] > 0:
                exchange.set_actuator_value(self.state, handles['commanded_mass_flow'], oa_mass_flow)
        
        self.current_action = action
    
    def _zone_people_and_ppd(self):
        """Return list of (people_count, ppd) for controlled zones with valid handles."""
        exchange = self.api.exchange
        rows = []
        for zone_name, zhandles in self.handles.get('zones', {}).items():
            people = 0.0
            h_people = zhandles.get('people', -1)
            if h_people and h_people > 0:
                people = float(exchange.get_variable_value(self.state, h_people))
            ppd = None
            h_ppd = zhandles.get('ppd', -1)
            if h_ppd and h_ppd > 0:
                ppd = float(exchange.get_variable_value(self.state, h_ppd))
            rows.append((people, ppd))
        return rows

    def _productivity_cost_from_loss(self, loss_frac, people, reward_config, dt_hours):
        """Convert fractional productivity loss to $ for this timestep."""
        wage = float(reward_config.get('labor_cost_per_person_hour', 40.0))
        return max(0.0, float(loss_frac)) * max(0.0, float(people)) * wage * dt_hours

    def _compute_comfort_penalty(self, reward_config, zone_temps, dt_hours=None, apply_static_weight=True):
        """Thermal penalty: temperature, soft PPD, or PPD→productivity $ cost.

        ppd_productivity sources (order-of-magnitude linear proxy):
          ASHRAE 55 (~10% PPD design); Kosonen & Tan (PMV/PPD–productivity);
          Lan, Wargocki & Lian (2011) Energy and Buildings; Seppänen/Fisk thermal
          performance reviews. Slope ppd_productivity_loss_per_percent is tunable.

        If apply_static_weight is False, return the raw cost before comfort_weight
        (used for adaptive balancing).
        """
        model = reward_config.get('thermal_comfort_model', 'temperature')
        COMFORT_WEIGHT = float(reward_config['comfort_weight'])
        if dt_hours is None:
            dt_hours = 1.0 / self.timesteps_per_hour

        if model == 'ppd_productivity':
            # Occupant-weighted PPD → productivity loss → $
            # loss_frac = clamp((PPD - ppd_ref) * loss_per_ppd_point / 100, 0, max_loss)
            if not self.handles_initialized:
                return 0.0
            ppd_ref = float(reward_config.get('ppd_reference', 10.0))
            loss_per_ppd = float(reward_config.get('ppd_productivity_loss_per_percent', 0.5))
            max_loss = float(reward_config.get('ppd_max_productivity_loss', 0.15))
            cost = 0.0
            for people, ppd in self._zone_people_and_ppd():
                if ppd is None or people <= 0.0:
                    continue
                loss_frac = np.clip((ppd - ppd_ref) * loss_per_ppd / 100.0, 0.0, max_loss)
                cost += self._productivity_cost_from_loss(loss_frac, people, reward_config, dt_hours)
            return cost * COMFORT_WEIGHT if apply_static_weight else cost

        if model == 'ppd':
            # Soft excess-PPD score (not dollars)
            ppd_max = reward_config.get('ppd_acceptable_max', 10.0)
            scale = reward_config.get('ppd_penalty_scale', 0.01)
            comfort_penalty = 0.0
            if self.handles_initialized:
                for people, ppd in self._zone_people_and_ppd():
                    if ppd is None:
                        continue
                    comfort_penalty += max(0.0, ppd - ppd_max) * scale
            return comfort_penalty * COMFORT_WEIGHT if apply_static_weight else comfort_penalty

        # Temperature-based: penalty when zone temp deviates from base_temp beyond threshold
        COMFORT_THRESHOLD = reward_config['comfort_threshold']
        comfort_penalty = 0.0
        for temp in zone_temps:
            deviation = abs(temp - self.base_temp)
            if deviation > COMFORT_THRESHOLD:
                comfort_penalty += (deviation - COMFORT_THRESHOLD) * 0.5
        return comfort_penalty * COMFORT_WEIGHT if apply_static_weight else comfort_penalty

    def _compute_co2_penalty(self, reward_config, dt_hours=None, apply_static_weight=True):
        """IAQ penalty: soft ppm threshold or CO2→productivity $ cost.

        productivity model sources (order-of-magnitude):
          Seppänen, Fisk & Lei (2006) Indoor Air — ventilation & office performance;
          Wargocki et al. (2000) Indoor Air — outdoor air, SBS, productivity;
          Wargocki et al. pooled CO2–performance analyses (~%/100 ppm in some fits);
          Allen et al. (2016) COGfx — cognitive scores vs CO2/ventilation.
          Default co2_productivity_loss_per_100ppm=0.01 is a tunable proxy.

        For each floor (occupant-weighted floor CO2):
          loss_frac = clamp((CO2 - co2_ref) / 100 * loss_per_100ppm, 0, max_loss)
          cost += loss_frac * people_on_floor * wage * dt

        If apply_static_weight is False, return raw cost before co2_weight.
        """
        co2_weight = float(reward_config.get('co2_weight', 0.0))
        if not self.handles_initialized:
            return 0.0
        if apply_static_weight and co2_weight <= 0.0:
            return 0.0
        if dt_hours is None:
            dt_hours = 1.0 / self.timesteps_per_hour

        model = reward_config.get('co2_model', 'threshold')
        exchange = self.api.exchange

        if model == 'productivity':
            co2_ref = float(reward_config.get('co2_reference_ppm', 800.0))
            loss_per_100 = float(reward_config.get('co2_productivity_loss_per_100ppm', 0.01))
            max_loss = float(reward_config.get('co2_max_productivity_loss', 0.15))
            cost = 0.0
            for floor in FLOOR_ORDER:
                co2_values = []
                people_sum = 0.0
                for zone_name in FLOOR_ZONE_GROUPS[floor]:
                    zhandles = self.handles['zones'].get(zone_name)
                    if not zhandles:
                        continue
                    if zhandles.get('co2', -1) > 0:
                        co2_values.append(exchange.get_variable_value(self.state, zhandles['co2']))
                    h_people = zhandles.get('people', -1)
                    if h_people and h_people > 0:
                        people_sum += float(exchange.get_variable_value(self.state, h_people))
                if not co2_values or people_sum <= 0.0:
                    continue
                floor_co2 = float(np.mean(co2_values))
                loss_frac = np.clip(
                    (floor_co2 - co2_ref) / 100.0 * loss_per_100, 0.0, max_loss
                )
                cost += self._productivity_cost_from_loss(
                    loss_frac, people_sum, reward_config, dt_hours
                )
            return cost * co2_weight if apply_static_weight else cost

        # Legacy soft threshold score (not dollars)
        threshold = float(reward_config.get('co2_threshold_ppm', 1000.0))
        scale = float(reward_config.get('co2_penalty_scale', 0.001))
        penalty = 0.0
        for floor in FLOOR_ORDER:
            co2_values = []
            for zone_name in FLOOR_ZONE_GROUPS[floor]:
                zhandles = self.handles['zones'].get(zone_name)
                if zhandles and zhandles.get('co2', -1) > 0:
                    co2_values.append(exchange.get_variable_value(self.state, zhandles['co2']))
            if not co2_values:
                continue
            floor_co2 = float(np.mean(co2_values))
            penalty += max(0.0, floor_co2 - threshold) * scale
        return penalty * co2_weight if apply_static_weight else penalty

    def _apply_cost_normalization(self, reward_config, components):
        """Optionally convert absolute $ costs to normalized $/m² for transferability.

        reward.cost_normalization.mode:
          absolute — leave components as building-total $ for the timestep
          per_m2   — divide dollar terms by floor_area_m2

        Non-dollar terms (legacy setpoint/demand scores) are left unchanged.
        """
        norm = reward_config.get('cost_normalization') or {}
        mode = str(norm.get('mode', 'absolute')).lower()
        if mode in ('none', 'absolute', ''):
            components['cost_normalization_mode'] = 'absolute'
            components['cost_normalization_factor'] = 1.0
            return components
        if mode != 'per_m2':
            raise ValueError(
                f"reward.cost_normalization.mode must be 'absolute' or 'per_m2', got {mode!r}"
            )
        area = float(norm.get('floor_area_m2', 0.0))
        if area <= 0.0:
            raise ValueError(
                "reward.cost_normalization.floor_area_m2 must be > 0 when mode is 'per_m2'"
            )
        factor = 1.0 / area
        for key in ('energy_cost', 'gas_cost', 'comfort_penalty', 'co2_penalty', 'comfort_raw', 'co2_raw'):
            if key in components:
                components[key] = float(components[key]) * factor
        # Rebuild total from normalized dollar terms + any non-normalized residuals
        components['total_cost'] = (
            components['energy_cost']
            + components['gas_cost']
            + components['comfort_penalty']
            + components['setpoint_penalty']
            + components['demand_penalty']
            + components['co2_penalty']
        )
        components['reward'] = -components['total_cost']
        components['cost_normalization_mode'] = 'per_m2'
        components['cost_normalization_factor'] = factor
        return components

    def compute_reward_components(self, reward_config, zone_temps, action):
        """
        Compute reward and all components (energy cost, comfort, setpoint, demand, price used).
        Single place for reward logic; used by calculate_reward and for logging.
        Returns dict with: reward, energy_cost, comfort_penalty, setpoint_penalty, demand_penalty,
        co2_penalty, total_cost, energy_price_used, energy_kwh, current_power.

        Thermal (ppd_productivity) and IAQ (co2_model=productivity) use literature-based
        order-of-magnitude productivity translations; see config reward comments for sources
        (ASHRAE 55; Kosonen & Tan; Lan/Wargocki/Lian 2011; Seppänen/Fisk ventilation &
        temperature-performance work; Wargocki et al. 2000; Allen et al. COGfx).
        Optional cost_normalization.mode=per_m2 converts $ → $/m².
        """
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
        energy_cost = energy_kwh * energy_price_used  # $ = kWh × RTP

        GAS_PRICE = reward_config.get('gas_price_per_kwh', 0.017)
        gas_j = self.api.exchange.get_meter_value(self.state, self.handles['gas_meter'])
        gas_kwh = gas_j / 3_600_000.0
        current_gas_power = (gas_kwh / dt_hours) * 1000.0
        gas_cost = gas_kwh * GAS_PRICE  # $ = kWh × gas price
        
        # Raw productivity $ (before static + adaptive weights) for online balancing
        comfort_raw = self._compute_comfort_penalty(
            reward_config, zone_temps, dt_hours=dt_hours, apply_static_weight=False
        )
        co2_raw = self._compute_co2_penalty(
            reward_config, dt_hours=dt_hours, apply_static_weight=False
        )
        energy_ref = energy_cost + gas_cost
        adaptive = self.adaptive_cost_balancer.update_and_get_scales(
            energy_ref, comfort_raw, co2_raw
        )
        comfort_weight = float(reward_config['comfort_weight'])
        co2_weight = float(reward_config.get('co2_weight', 0.0))
        comfort_penalty = comfort_raw * comfort_weight * float(adaptive['comfort'])
        co2_penalty = co2_raw * co2_weight * float(adaptive['co2'])
        
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
        if DEMAND_WEIGHT > 0.0 and current_power_kw > DEMAND_THRESHOLD:
            demand_penalty = (current_power_kw - DEMAND_THRESHOLD) * 0.2
        demand_penalty *= DEMAND_WEIGHT
        
        total_cost = (
            energy_cost + gas_cost + comfort_penalty + setpoint_penalty
            + demand_penalty + co2_penalty
        )
        reward = -total_cost

        components = {
            'reward': reward,
            'energy_cost': energy_cost,
            'gas_cost': gas_cost,
            'comfort_penalty': comfort_penalty,
            'setpoint_penalty': setpoint_penalty,
            'demand_penalty': demand_penalty,
            'co2_penalty': co2_penalty,
            'comfort_raw': comfort_raw,
            'co2_raw': co2_raw,
            'adaptive_comfort_scale': float(adaptive['comfort']),
            'adaptive_co2_scale': float(adaptive['co2']),
            'adaptive_balancing_active': bool(adaptive['active']),
            'adaptive_ema_energy': adaptive['ema_energy'],
            'adaptive_ema_comfort': adaptive['ema_comfort'],
            'adaptive_ema_co2': adaptive['ema_co2'],
            'adaptive_n_samples': int(adaptive['n_samples']),
            'total_cost': total_cost,
            'energy_price_used': energy_price_used,
            'energy_kwh': energy_kwh,
            'current_power': current_power,
            'gas_kwh': gas_kwh,
            'current_gas_power': current_gas_power,
        }
        return self._apply_cost_normalization(reward_config, components)
    
    def calculate_reward(self, action, curr_state):
        """Calculate reward based on energy efficiency and thermal comfort (uses compute_reward_components)."""
        reward_config = self.hvac_config.config['reward']
        # Prefer all controlled zone temps for comfort; fall back to observation features
        if self.handles_initialized and self.handles.get('zones'):
            zone_temps = [
                self._zone_temp_value(zone_name)
                for zone_name in self.handles['zones']
            ]
        else:
            zone_count = self.hvac_config.config['state_space']['zone_temps']['count']
            zone_temps = list(curr_state[:zone_count])
        components = self.compute_reward_components(reward_config, zone_temps, action)
        self._last_reward_components = components
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
            self._last_reward_components = None
        
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
                 episode_kpi_plotter=None, output_dir=None, model_path=None, save_model=False, save_every=20,
                 simulation_overrides=None):
        self.api = api
        self.state = state
        self.training_mode = training_mode
        self.live_plotter = live_plotter
        self.loss_plotter = loss_plotter
        self.episode_kpi_plotter = episode_kpi_plotter
        self.output_dir = output_dir

        # Checkpointing: save_model gates both the final save and periodic
        # checkpoints every `save_every` episodes during training.
        self.save_model_enabled = save_model
        self.save_every = save_every

        # Initialize environment
        self.env = HVACEnvironment(api, state, config_path, simulation_overrides=simulation_overrides)

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
                bal = getattr(self.env, 'adaptive_cost_balancer', None)
                if bal is not None and bal.load_state_dict(extra_state.get('adaptive_balancer')):
                    print(
                        "Adaptive balancer restored: "
                        f"n_samples={bal.n_samples}  "
                        f"ppd_scale={bal.comfort_scale:.4f}  "
                        f"co2_scale={bal.co2_scale:.4f}  "
                        f"active={bal.n_samples >= bal.min_samples}"
                    )
                elif training_mode and bal is not None and bal.enabled:
                    print(
                        "No adaptive balancer state in checkpoint — "
                        "scales restart at initial_scale until min_samples."
                    )
                if training_mode:
                    mem = len(self.agent.memory)
                    if mem >= self.training_start_memory:
                        print(
                            f"Replay buffer restored ({mem} transitions) — "
                            "training updates will start on the first new step."
                        )
                    else:
                        print(
                            f"Replay buffer has {mem}/{self.training_start_memory} transitions — "
                            "collecting more before training updates resume."
                        )
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
        self.env.apply_occupancy_events_for_current_timestep()

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
        # Zone temps for logging / display (and fallback reward recompute)
        raw_zone_temps = []
        for zone_name, zhandles in self.env.handles['zones'].items():
            h_temp = zhandles.get('temp', -1)
            if h_temp and h_temp > 0:
                raw_zone_temps.append(self.api.exchange.get_variable_value(self.state, h_temp))
        if not raw_zone_temps:
            zone_count_for_reward = self.env.hvac_config.config['state_space']['zone_temps']['count']
            raw_zone_temps = list(current_state[:zone_count_for_reward])
        # Prefer components already computed in env.step (avoids double-updating adaptive EMA)
        if self.env._last_reward_components is not None:
            components = self.env._last_reward_components
        else:
            components = self.env.compute_reward_components(reward_config, raw_zone_temps, action)
        reward = components['reward']
        self.env.episode_reward += reward - env_step_reward
        energy_cost = components['energy_cost']
        gas_cost = components['gas_cost']
        comfort_penalty = components['comfort_penalty']
        setpoint_penalty = components['setpoint_penalty']
        demand_penalty = components['demand_penalty']
        co2_penalty = components['co2_penalty']
        total_cost = components['total_cost']
        current_power = components['current_power']
        energy_price_used = components['energy_price_used']
        energy_kwh = components['energy_kwh']
        gas_kwh = components['gas_kwh']
        current_gas_power = components['current_gas_power']
        cost_norm_mode = components.get('cost_normalization_mode', 'absolute')
        adaptive_comfort_scale = components.get('adaptive_comfort_scale', 1.0)
        adaptive_co2_scale = components.get('adaptive_co2_scale', 1.0)
        adaptive_balancing_active = components.get('adaptive_balancing_active', False)
        adaptive_n_samples = components.get('adaptive_n_samples', 0)
        
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
        oa_mass_flow = max(
            0.0,
            _scale(action[2], action_bounds['airflow_multiplier'][0], action_bounds['airflow_multiplier'][1]),
        )
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

        ahu_oa_flows = {}
        for floor, handles in self.env.handles.get('ahu_oa', {}).items():
            h_oa = handles.get('mass_flow', -1)
            if h_oa and h_oa > 0:
                ahu_oa_flows[floor] = exchange.get_variable_value(self.state, h_oa)
            else:
                ahu_oa_flows[floor] = np.nan
        valid_ahu_oa_flows = [v for v in ahu_oa_flows.values() if not np.isnan(v)]
        
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
            'elapsed_hours': self.env.timestep_count / self.env.timesteps_per_hour,
            'action': action.tolist(),
            'reward': reward,
            'energy_cost': energy_cost,
            'gas_cost': gas_cost,
            'comfort_penalty': comfort_penalty,
            'setpoint_penalty': setpoint_penalty,
            'demand_penalty': demand_penalty,
            'co2_penalty': co2_penalty,
            'total_cost': total_cost,
            'cost_normalization_mode': cost_norm_mode,
            'cost_normalization_factor': components.get('cost_normalization_factor', 1.0),
            'adaptive_comfort_scale': adaptive_comfort_scale,
            'adaptive_co2_scale': adaptive_co2_scale,
            'adaptive_balancing_active': adaptive_balancing_active,
            'adaptive_n_samples': adaptive_n_samples,
            'adaptive_ema_energy': components.get('adaptive_ema_energy'),
            'adaptive_ema_comfort': components.get('adaptive_ema_comfort'),
            'adaptive_ema_co2': components.get('adaptive_ema_co2'),
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
            'commanded_oa_mass_flow': oa_mass_flow,
            'airflow': float(np.mean(valid_ahu_oa_flows)) if valid_ahu_oa_flows else np.nan,
            'bottom_floor_airflow': ahu_oa_flows.get('bottom', np.nan),
            'mid_floor_airflow': ahu_oa_flows.get('mid', np.nan),
            'top_floor_airflow': ahu_oa_flows.get('top', np.nan),
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
        if self.loss_plotter is not None:
            self.loss_plotter.update_weights(log_entry)
        if self.episode_kpi_plotter is not None:
            self.episode_kpi_plotter.update_step(log_entry)
        
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
        # States: print observation temp features (aggregated or per-zone)
        zone_count = self.env.hvac_config.config['state_space']['zone_temps']['count']
        zone_temp_mode = self.env.hvac_config.config['state_space']['zone_temps'].get('mode', 'zones')
        zone_temps = current_state[:zone_count]
        if len(current_state) <= zone_count:
            raise RuntimeError(
                f"State is missing outdoor temperature at index {zone_count}; "
                f"state length is {len(current_state)}"
            )
        outdoor_temp = current_state[zone_count]
        if zone_temp_mode == 'perimeter_core':
            display_temps = []
            for floor in FLOOR_ORDER:
                display_temps.append(self.env._zone_temp_value(CORE_ZONE_BY_FLOOR[floor]))
            for floor in FLOOR_ORDER:
                perim = [self.env._zone_temp_value(z) for z in PERIMETER_ZONE_GROUPS[floor]]
                display_temps.append(float(np.mean(perim)))
            labels = PERIMETER_CORE_TEMP_COLUMNS
            zone_str = ", ".join(
                [f"{labels[i]}={display_temps[i]:5.1f}" for i in range(len(display_temps))]
            )
        elif zone_temp_mode == 'floor':
            display_temps = []
            for floor in FLOOR_ORDER:
                floor_temps = [self.env._zone_temp_value(z) for z in FLOOR_ZONE_GROUPS[floor]]
                display_temps.append(float(np.mean(floor_temps)))
            zone_str = ", ".join(
                [f"{FLOOR_TEMP_COLUMNS[i]}={display_temps[i]:5.1f}" for i in range(len(display_temps))]
            )
        else:
            display_zone_temps = raw_zone_temps[:zone_count]
            zone_str = ", ".join([f"{t:5.1f}" for t in display_zone_temps])
        memory_len = len(self.agent.memory)
        memory_status = f"{memory_len}/{self.training_start_memory}" if self.training_mode else str(memory_len)
        
        # Print: time, step; actual actions (raw + scaled); states; reward; then blank line
        print(f"{datetime_str} | Ep {episode_no:3d} | Step {episode_step:3d}")
        action_items = [
            f"raw_sp={action[0]:+.2f}", f"raw_db={action[1]:+.2f}", f"raw_oa={action[2]:+.2f}",
            f"sp_offset={sp_offset:+.2f}°C", f"htg_sp={htg_sp:.1f}°C",
            f"clg_sp={clg_sp:.1f}°C", f"deadband={deadband:.2f}°C",
            f"oa_cmd={oa_mass_flow:.3f}kg/s",
            f"oa_flow_bot={ahu_oa_flows.get('bottom', np.nan):.3f}kg/s",
            f"oa_flow_mid={ahu_oa_flows.get('mid', np.nan):.3f}kg/s",
            f"oa_flow_top={ahu_oa_flows.get('top', np.nan):.3f}kg/s",
        ]
        print("   Actions:")
        for i in range(0, len(action_items), 5):
            print("     " + "  ".join(action_items[i:i+5]))
        print(f"   States:  zone_temps=[{zone_str}]°C  outdoor_temp={raw_outdoor_temp:5.1f}°C  memory_len={memory_status}")
        print(f"   State vector [{len(current_state)}]:")
        for i in range(0, len(current_state), 10):
            chunk = current_state[i:i+10]
            print("     " + "  ".join([f"[{i+j:2d}]{v:+.3f}" for j, v in enumerate(chunk)]))
        print(f"   Reward: {reward:7.3f}  episode_total={self.env.episode_reward:7.3f}  |  elec_cost[$]={energy_cost:.4f}  gas_cost[$]={gas_cost:.4f}  comfort={comfort_penalty:.4f}  setpoint={setpoint_penalty:.4f}  demand_penalty[$]={demand_penalty:.4f}  co2={co2_penalty:.4f}  total_cost={total_cost:.4f}  |  price[$/kWh]={energy_price_used:.3f}  elec[kWh]={energy_kwh:.4f}  elec[kW]={current_power/1000:.2f}  gas[kWh]={gas_kwh:.4f}  gas[kW]={current_gas_power/1000:.2f}")
        print()
        
        # Reset if episode done: sample next random start (after current time) and duration
        if done:
            self.episode_count += 1
            dur_h = getattr(self, '_current_episode_duration_hours', None)
            dur_str = f" (duration: {dur_h:.1f} h)" if dur_h is not None else ""
            print(f"\n--- Episode {self.episode_count} completed{dur_str} ---\n")
            if self.episode_kpi_plotter is not None:
                self.episode_kpi_plotter.finalize_episode(self.episode_count)

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
        """Save trained RL model, including episode_count and adaptive balancer state."""
        bal = getattr(self.env, 'adaptive_cost_balancer', None)
        extra = {'episode_count': self.episode_count}
        if bal is not None and bal.enabled:
            extra['adaptive_balancer'] = bal.state_dict()
        self.agent.save(path, extra_state=extra)
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
    adaptive_weight_plot=False,
    episode_kpi_plot=False,
    live_plot_every=1,
    live_plot_hold=False,
    live_plot_scope="current",
    model_path=None,
    save_model=False,
    save_every=20,
    simulation_overrides=None,
):
    """Run EnergyPlus simulation with RL HVAC control.
    
    Simulation is driven by config training_window (start/end date and time) and
    episode_duration_hours [min, max]. A custom IDF is created when custom_period.enabled
    with RunPeriod = training_window start date to end date. RL runs only between
    start_date+start_hour and end_date+end_hour; each episode length is sampled from
    [episode_duration_hours min, max].
    """
    hvac_config = get_hvac_config(config)
    if simulation_overrides:
        sim_override_cfg = hvac_config.config.setdefault('simulation', {})
        for key, value in simulation_overrides.items():
            if value is not None:
                sim_override_cfg[key] = value
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
        occupancy_schedule_name = hvac_config.config.get('occupancy_events', {}).get('schedule_name')
        idf_path = create_custom_idf(
            idf_path, start_month, start_day, end_month, end_day, output_dir,
            outdoor_co2_ppm=outdoor_co2,
            outdoor_co2_csv_path=outdoor_co2_csv,
            outdoor_co2_fallback_ppm=outdoor_co2_fallback,
            occupancy_schedule_name=occupancy_schedule_name,
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

    show_losses = bool(loss_plot and training_mode)
    show_weights = bool(adaptive_weight_plot)
    if loss_plot and not training_mode:
        print("Warning: --loss-plot requested without --training; SAC loss panels will be omitted")
    if adaptive_weight_plot:
        ab = (hvac_config.config.get("reward") or {}).get("adaptive_balancing") or {}
        if not ab.get("enabled", False):
            print(
                "Warning: --adaptive-weight-plot set but reward.adaptive_balancing.enabled "
                "is false; scales will stay at 1.0"
            )
    if show_losses or show_weights:
        loss_plotter = LiveLossPlotter(
            output_dir=output_dir,
            update_every=live_plot_every,
            show_losses=show_losses,
            show_adaptive_weights=show_weights,
        )
        parts = []
        if show_losses:
            parts.append("SAC losses")
        if show_weights:
            parts.append("adaptive PPD/CO₂ weights")
        print("Training plot enabled (" + " + ".join(parts) + ")")

    episode_kpi_plotter = None
    if episode_kpi_plot:
        reward_cfg = hvac_config.config.get("reward") or {}
        norm = reward_cfg.get("cost_normalization") or {}
        area = norm.get("floor_area_m2")
        episode_kpi_plotter = LiveEpisodeDollarKPIPlotter(
            output_dir=output_dir,
            floor_area_m2=area,
        )
        print("Episode dollar KPI plot enabled (absolute $ per episode)")

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    controller = RLHVACController(
        api,
        state,
        config,
        training_mode=training_mode,
        live_plotter=live_plotter,
        loss_plotter=loss_plotter,
        episode_kpi_plotter=episode_kpi_plotter,
        output_dir=output_dir,
        model_path=model_path,
        save_model=save_model,
        save_every=save_every,
        simulation_overrides=simulation_overrides,
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
    zone_temp_mode = ss['zone_temps'].get('mode', 'zones')
    wf = controller.env.hvac_config.weather_forecast_offsets
    weather_history_cfg = ss.get('weather_history', {})
    past_horizon = weather_history_cfg.get('horizon', 0) if weather_history_cfg.get('enabled', False) else 0
    floor_co2_count = ss.get('floor_co2_ppm', {}).get('count', 0)

    state_columns = []
    if zone_temp_mode == 'perimeter_core':
        state_columns += [f"zone_temp_{name}" for name in PERIMETER_CORE_TEMP_COLUMNS[:zone_count]]
    elif zone_temp_mode == 'floor':
        state_columns += [f"zone_temp_{name}" for name in FLOOR_TEMP_COLUMNS[:zone_count]]
    else:
        for z in zone_names[:zone_count]:
            state_columns.append(f"zone_temp_{z}")
    state_columns += ["oat", "humidity", "sky_temp"]
    for field, label in (('oat', 'oat'), ('humidity', 'humidity'), ('cloud_cover', 'sky_temp')):
        for t in wf[field]:
            state_columns.append(f"forecast_t{t}_{label}")
    for t in range(1, past_horizon + 1):
        state_columns += [f"past_t{t}_oat", f"past_t{t}_humidity", f"past_t{t}_sky_temp"]
    state_columns += ["hour_of_day", "day_of_week", "month_of_year"]
    prev_count = ss.get('previous_actions', {}).get('count', 0)
    previous_action_columns = ["prev_sp_offset", "prev_deadband", "prev_airflow"]
    state_columns += previous_action_columns[:prev_count]
    state_columns += ["outdoor_co2"]
    for floor in FLOOR_ORDER[:floor_co2_count]:
        state_columns.append(f"floor_co2_{floor}")
    rtp_count = ss.get('rtp_price', {}).get('count', 0)
    rtp_columns = ["rtp_price_current", "rtp_price_hour_ahead"]
    state_columns += rtp_columns[:rtp_count]

    print("\n" + "=" * 60)
    print("RL HVAC Control Configuration")
    print("=" * 60)
    print(f"  Timesteps per hour: {controller.env.timesteps_per_hour}")
    print(f"  Training mode:      {controller.training_mode}")
    print(f"  Action size:        {controller.env.action_size}  [sp_offset, deadband, oa_mass_flow]")
    action_cfg = cfg['action_space']
    action_columns = [
        ("sp_offset",          action_cfg['sp_offset']['min'],          action_cfg['sp_offset']['max'],          "°C"),
        ("deadband",           action_cfg['deadband']['min'],           action_cfg['deadband']['max'],           "°C"),
        ("oa_mass_flow", action_cfg['airflow_multiplier']['min'], action_cfg['airflow_multiplier']['max'], "kg/s"),
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
        if episode_kpi_plotter is not None:
            episode_kpi_plotter.finish(output_dir, hold=live_plot_hold)
    
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
    parser.add_argument('--adaptive-weight-plot', action='store_true',
                        help='Add adaptive PPD/CO2 productivity scales (and cost EMAs) to the '
                             'SAC loss plot window; creates that window if --loss-plot is not set')
    parser.add_argument('--episode-kpi-plot', action='store_true',
                        help='Show a live chart of absolute dollar KPIs (electric, gas, thermal, '
                             'IAQ, total) summed per completed episode')
    parser.add_argument('--live-plot-every', type=int, default=1,
                        help='Refresh the live plot every N logged timesteps (default: 1)')
    parser.add_argument('--live-plot-hold', action='store_true',
                        help='Keep the live plot window open after the simulation finishes')
    parser.add_argument('--live-plot-scope', choices=['current', 'all'], default='current',
                        help='Live plot scope: current resets at each episode; all overlays all episodes on one continuous timeline')
    parser.add_argument('--outdoor-co2-csv', type=str, default=None,
                        help='Path to outdoor CO2 CSV (overrides simulation.outdoor_co2_csv_path in config)')
    parser.add_argument('--outdoor-co2-fallback', type=float, default=None,
                        help='Fallback ppm for missing CSV hours (overrides simulation.outdoor_co2_fallback_ppm)')
    
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

    simulation_overrides = {}
    if args.outdoor_co2_csv:
        co2_csv = args.outdoor_co2_csv
        if not os.path.isabs(co2_csv):
            co2_csv = str(project_root / co2_csv)
        if not os.path.exists(co2_csv):
            print(f"Error: Outdoor CO2 CSV not found: {co2_csv}")
            return 1
        simulation_overrides['outdoor_co2_csv_path'] = co2_csv
        print(f"Using outdoor CO2 CSV: {co2_csv}")
    if args.outdoor_co2_fallback is not None:
        simulation_overrides['outdoor_co2_fallback_ppm'] = args.outdoor_co2_fallback
    
    # Run simulation: max_episodes from --episodes; --training enables storing transitions (memory grows)
    return run_simulation(
        args.idf, args.epw, args.output, args.config,
        max_episodes=args.episodes,
        training_mode=args.training,
        override_test=args.override_test,
        live_plot=args.live_plot,
        loss_plot=args.loss_plot,
        adaptive_weight_plot=args.adaptive_weight_plot,
        episode_kpi_plot=args.episode_kpi_plot,
        live_plot_every=args.live_plot_every,
        live_plot_hold=args.live_plot_hold,
        live_plot_scope=args.live_plot_scope,
        model_path=args.model,
        save_model=args.save_model,
        save_every=args.save_every,
        simulation_overrides=simulation_overrides or None,
    )


if __name__ == "__main__":
    sys.exit(main())
