# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Real-time SITL visualizer using matplotlib.

Reads a shared-memory-style state file written by sitl_wheeled.py,
or can be subclassed to receive state via a queue.

Usage (standalone, reads from state file):
    python tools/sitl/viz.py --state /tmp/sitl_state.json --scenario scenarios/patrol_abc.yaml

Usage (embedded — import and call from sitl_wheeled.py):
    from tools.sitl.viz import SITLViz
    viz = SITLViz(world, update_hz=10)
    viz.update(robot)   # call in async loop
    viz.show()
"""

import argparse
import json
import math
import os
import sys
import time

try:
    import matplotlib

    matplotlib.use("TkAgg")  # works headless-free on macOS
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrow

    _MPL = True
except ImportError:
    _MPL = False

import yaml

# ── Visualizer ────────────────────────────────────────────────────────────────


class SITLViz:
    """Real-time top-down 2D visualizer for the wheeled SITL."""

    def __init__(self, scenario: dict, update_hz: float = 10):
        if not _MPL:
            raise ImportError("matplotlib is required: pip install matplotlib")

        self.scenario = scenario
        self.interval = 1.0 / update_hz
        self._last_draw = 0.0

        width = scenario.get("width_mm", 10000)
        height = scenario.get("height_mm", 10000)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_title("Robot SITL — top view")
        self.ax.grid(True, alpha=0.3)

        # Draw obstacles
        for obs in scenario.get("obstacles", []):
            rect = patches.Rectangle(
                (obs["x"], obs["y"]),
                obs["w"],
                obs["h"],
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.6,
            )
            self.ax.add_patch(rect)

        # Draw waypoints
        for name, wp in scenario.get("waypoints", {}).items():
            self.ax.plot(wp["x_mm"], wp["y_mm"], "b^", markersize=10)
            self.ax.annotate(
                name,
                (wp["x_mm"], wp["y_mm"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color="blue",
            )

        # Robot marker (arrow)
        self._robot_patch = None
        self._range_line_f = None
        self._range_line_r = None
        self._trail_x: list[float] = []
        self._trail_y: list[float] = []
        (self._trail_line,) = self.ax.plot([], [], "g-", linewidth=1, alpha=0.5, label="trail")

        # Status text
        self._status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        plt.legend(loc="lower right")
        plt.ion()
        plt.show()

    def update(
        self,
        x: float,
        y: float,
        hdg_deg: float,
        range_front_mm: float,
        range_right_mm: float,
        battery_mv: float,
        speed_l: int,
        speed_r: int,
    ):
        """Update visualization with latest robot state."""
        now = time.monotonic()
        if now - self._last_draw < self.interval:
            return
        self._last_draw = now

        # Remove old robot marker
        if self._robot_patch:
            self._robot_patch.remove()
        if self._range_line_f:
            self._range_line_f.remove()
        if self._range_line_r:
            self._range_line_r.remove()

        # Draw robot body (circle)
        self._robot_patch = patches.Circle((x, y), radius=100, color="red", zorder=5)
        self.ax.add_patch(self._robot_patch)

        # Draw heading arrow
        dx = 150 * math.cos(math.radians(hdg_deg))
        dy = 150 * math.sin(math.radians(hdg_deg))
        self.ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        # Range sensor rays
        cap = 2000  # display cap at 2000 mm
        rf = min(range_front_mm, cap)
        rr = min(range_right_mm, cap)

        fx = x + rf * math.cos(math.radians(hdg_deg))
        fy = y + rf * math.sin(math.radians(hdg_deg))
        (self._range_line_f,) = self.ax.plot([x, fx], [y, fy], "y--", lw=1, alpha=0.7)

        rx = x + rr * math.cos(math.radians((hdg_deg - 90) % 360))
        ry = y + rr * math.sin(math.radians((hdg_deg - 90) % 360))
        (self._range_line_r,) = self.ax.plot([x, rx], [y, ry], "c--", lw=1, alpha=0.7)

        # Trail
        self._trail_x.append(x)
        self._trail_y.append(y)
        if len(self._trail_x) > 2000:
            self._trail_x = self._trail_x[-1000:]
            self._trail_y = self._trail_y[-1000:]
        self._trail_line.set_data(self._trail_x, self._trail_y)

        # Status text
        batt_pct = max(0, (battery_mv - 6000) / (8400 - 6000) * 100)
        status = (
            f"Pos: ({x:.0f}, {y:.0f}) mm\n"
            f"Hdg: {hdg_deg:.1f}°\n"
            f"Speed L/R: {speed_l}/{speed_r}%\n"
            f"Front: {range_front_mm:.0f} mm\n"
            f"Right: {range_right_mm:.0f} mm\n"
            f"Batt: {battery_mv:.0f} mV ({batt_pct:.0f}%)"
        )
        self._status_text.set_text(status)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ── Standalone mode: reads JSON state file ────────────────────────────────────


def watch_state_file(state_path: str, scenario: dict, hz: float = 10):
    viz = SITLViz(scenario, update_hz=hz)
    print(f"[VIZ] Watching {state_path} at {hz} Hz. Close window to exit.")

    while plt.get_fignums():
        try:
            with open(state_path) as f:
                state = json.load(f)
            viz.update(
                x=state.get("x", 0),
                y=state.get("y", 0),
                hdg_deg=state.get("hdg_deg", 0),
                range_front_mm=state.get("range_front_mm", 9999),
                range_right_mm=state.get("range_right_mm", 9999),
                battery_mv=state.get("battery_mv", 7400),
                speed_l=state.get("speed_l", 0),
                speed_r=state.get("speed_r", 0),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(1.0 / hz)


def main():
    ap = argparse.ArgumentParser(description="SITL Real-time Visualizer")
    ap.add_argument(
        "--state", default="/tmp/sitl_state.json", help="JSON state file written by sitl_wheeled.py"
    )
    ap.add_argument("--scenario", default="tools/sitl/scenarios/empty.yaml")
    ap.add_argument("--hz", type=float, default=10)
    args = ap.parse_args()

    if not _MPL:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    scenario = {}
    if os.path.exists(args.scenario):
        with open(args.scenario) as f:
            scenario = yaml.safe_load(f) or {}

    watch_state_file(args.state, scenario, args.hz)


if __name__ == "__main__":
    main()
