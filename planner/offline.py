# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Offline Autonomy Manager (E05).

Handles robot behavior when the brain server loses connection.
The brain pre-loads a fallback plan that the kernel can execute independently.

States:
  ONLINE    — brain connected, normal operation
  DEGRADED  — connection unstable (high latency / packet loss)
  OFFLINE   — brain disconnected, fallback plan active
  DOCKING   — autonomous return to dock (low battery + offline)

Transitions:
  ONLINE → DEGRADED:  3 consecutive sensor timeouts
  DEGRADED → OFFLINE: 5s without valid sensor packet
  OFFLINE → ONLINE:   robot reconnects
  OFFLINE → DOCKING:  battery < DOCK_BATTERY_THRESHOLD_PCT
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("brain.offline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

## Number of consecutive sensor timeouts before entering DEGRADED.
DEGRADED_THRESHOLD_TIMEOUTS = 3

## Seconds without a sensor packet before transitioning to OFFLINE.
OFFLINE_TIMEOUT_S = 5.0

## Battery percentage below which OFFLINE → DOCKING.
DOCK_BATTERY_THRESHOLD_PCT = 20

## Maximum number of fallback waypoints to pre-load.
MAX_FALLBACK_WAYPOINTS = 32

## Reconnect attempt interval in seconds.
RECONNECT_INTERVAL_S = 10.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class OfflineState:
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DOCKING = "docking"


@dataclass
class FallbackWaypoint:
    """A waypoint for the kernel's offline patrol."""

    lat: float = 0.0
    lon: float = 0.0
    action: str = "patrol"  # patrol, stop, dock


@dataclass
class OfflineStatus:
    """Current offline autonomy status."""

    state: str = OfflineState.ONLINE
    last_sensor_time: float = 0.0
    timeout_count: int = 0
    fallback_waypoints: list = field(default_factory=list)
    battery_pct: int = 100
    time_offline_s: float = 0.0
    offline_start: float = 0.0


# ---------------------------------------------------------------------------
# OfflineManager
# ---------------------------------------------------------------------------


class OfflineManager:
    """Manages transitions between online/offline states."""

    def __init__(self, config: dict | None = None):
        self._status = OfflineStatus()
        cfg = config or {}
        self._offline_timeout = cfg.get("offline_timeout_s", OFFLINE_TIMEOUT_S)
        self._degraded_threshold = cfg.get("degraded_threshold", DEGRADED_THRESHOLD_TIMEOUTS)
        self._dock_battery_pct = cfg.get("dock_battery_pct", DOCK_BATTERY_THRESHOLD_PCT)
        self._fallback_waypoints: list[FallbackWaypoint] = []

    @property
    def state(self) -> str:
        return self._status.state

    @property
    def is_online(self) -> bool:
        return self._status.state == OfflineState.ONLINE

    @property
    def status(self) -> OfflineStatus:
        return self._status

    def set_fallback_waypoints(self, waypoints: list[FallbackWaypoint]):
        """Pre-load fallback waypoints for offline patrol."""
        self._fallback_waypoints = waypoints[:MAX_FALLBACK_WAYPOINTS]
        self._status.fallback_waypoints = [
            {"lat": w.lat, "lon": w.lon, "action": w.action} for w in self._fallback_waypoints
        ]
        logger.info("Loaded %d fallback waypoints", len(self._fallback_waypoints))

    def on_sensor_received(self, battery_pct: int = 100):
        """Called when a valid sensor packet arrives from the robot."""
        now = time.time()
        self._status.last_sensor_time = now
        self._status.timeout_count = 0
        self._status.battery_pct = battery_pct

        prev_state = self._status.state

        if self._status.state in (
            OfflineState.OFFLINE,
            OfflineState.DEGRADED,
            OfflineState.DOCKING,
        ):
            self._status.state = OfflineState.ONLINE
            self._status.time_offline_s = now - self._status.offline_start
            logger.info("Robot reconnected after %.1fs offline", self._status.time_offline_s)

        if prev_state != self._status.state:
            logger.info("State: %s → %s", prev_state, self._status.state)

    def on_sensor_timeout(self):
        """Called when expected sensor packet doesn't arrive in time."""
        self._status.timeout_count += 1
        prev_state = self._status.state

        if self._status.state == OfflineState.ONLINE:
            if self._status.timeout_count >= self._degraded_threshold:
                self._status.state = OfflineState.DEGRADED
                logger.warning("Connection degraded (%d timeouts)", self._status.timeout_count)

        if prev_state != self._status.state:
            logger.info("State: %s → %s", prev_state, self._status.state)

    def tick(self) -> str:
        """Periodic check — call every second. Returns current state."""
        now = time.time()

        if self._status.state == OfflineState.DEGRADED:
            elapsed = now - self._status.last_sensor_time
            if elapsed > self._offline_timeout:
                self._status.state = OfflineState.OFFLINE
                self._status.offline_start = now
                logger.warning("Robot OFFLINE (%.1fs since last sensor)", elapsed)

        if self._status.state == OfflineState.OFFLINE:
            if self._status.battery_pct < self._dock_battery_pct:
                self._status.state = OfflineState.DOCKING
                logger.warning(
                    "Battery %d%% < %d%% — DOCKING mode",
                    self._status.battery_pct,
                    self._dock_battery_pct,
                )

        return self._status.state

    def get_fallback_plan(self) -> list[dict]:
        """Get the pre-loaded fallback plan for the kernel."""
        return self._status.fallback_waypoints

    def status_text(self) -> str:
        """Human-readable status for Telegram /offline command."""
        s = self._status
        lines = [
            f"State: {s.state}",
            (
                f"Last sensor: {time.time() - s.last_sensor_time:.1f}s ago"
                if s.last_sensor_time > 0
                else "Last sensor: never"
            ),
            f"Timeouts: {s.timeout_count}",
            f"Battery: {s.battery_pct}%",
            f"Fallback waypoints: {len(s.fallback_waypoints)}",
        ]
        if s.state == OfflineState.OFFLINE:
            lines.append(f"Offline for: {time.time() - s.offline_start:.0f}s")
        return "\n".join(lines)
