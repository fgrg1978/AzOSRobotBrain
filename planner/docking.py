# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Auto-docking manager — autonomous charging for 24/7 operation.

Monitors battery voltage, triggers return-to-dock when low, manages the
dock/undock state machine, and resumes patrol when charged.

State machine:
    PATROL → LOW_BATTERY → NAVIGATING → HOMING → DOCKED → CHARGING
         → CHARGED → UNDOCKING → PATROL

IR beacon homing:
    Dock has an IR LED (850nm, continuous). Robot's IR sensor (GPIO input)
    detects beacon for final approach (~50cm). Brain sends differential
    steering based on IR signal strength / position.

Battery monitoring:
    ADS1115 ADC on robot reads battery voltage via voltage divider.
    Sent in SensorPacket.battery_mv.  Brain tracks and decides thresholds.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

logger = logging.getLogger("brain.docking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATTERY_LOW_MV = 6800  # trigger return to dock (2S pack ~3.4V/cell)
BATTERY_CRITICAL_MV = 6400  # emergency dock, override current task
BATTERY_FULL_MV = 8200  # 2S fully charged (~4.1V/cell), resume patrol
BATTERY_HYSTERESIS_MV = 200  # prevent oscillation near threshold

DOCK_APPROACH_SPEED_PCT = 20  # slow final approach speed
DOCK_IR_HOMING_DISTANCE_MM = 500  # switch from nav to IR homing
DOCK_ALIGNMENT_TIMEOUT_S = 30  # max time for IR homing alignment
DOCK_CHARGE_CHECK_INTERVAL_S = 60  # how often to check battery while charging
DOCK_UNDOCK_DRIVE_S = 2.0  # seconds to drive forward off dock
DOCK_UNDOCK_SPEED_PCT = 30  # speed to drive off dock

BATTERY_CHECK_INTERVAL_S = 10.0  # how often to evaluate battery state
BATTERY_SAMPLES_FOR_DECISION = 3  # require N consecutive low readings


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DockState(enum.Enum):
    PATROL = "patrol"  # normal operation
    LOW_BATTERY = "low_battery"  # battery below threshold, preparing to dock
    NAVIGATING = "navigating"  # navigating to dock waypoint
    HOMING = "homing"  # IR beacon final approach
    DOCKED = "docked"  # physically on dock, waiting for charge
    CHARGING = "charging"  # charging in progress
    CHARGED = "charged"  # fully charged, ready to undock
    UNDOCKING = "undocking"  # driving off dock
    CRITICAL = "critical"  # emergency — battery critically low


@dataclass
class DockInfo:
    """Configuration for a charging dock."""

    dock_id: str = "dock_0"
    x_mm: int = 0
    y_mm: int = 0
    heading_cdeg: int = 0  # approach heading (face dock from this angle)


# ---------------------------------------------------------------------------
# DockManager
# ---------------------------------------------------------------------------


class DockManager:
    """Manages battery monitoring and autonomous docking/undocking."""

    def __init__(
        self,
        dock: DockInfo | None = None,
        on_dock_needed: Optional[Callable[[], Awaitable[None]]] = None,
        on_undock_ready: Optional[Callable[[], Awaitable[None]]] = None,
        on_critical: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self._dock = dock or DockInfo()
        self._on_dock_needed = on_dock_needed
        self._on_undock_ready = on_undock_ready
        self._on_critical = on_critical

        self.state = DockState.PATROL
        self._battery_mv: int = 0
        self._last_battery_check: float = 0.0
        self._low_count: int = 0  # consecutive low readings
        self._critical_count: int = 0
        self._charge_start_time: float = 0.0
        self._dock_start_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def battery_mv(self) -> int:
        return self._battery_mv

    @property
    def is_docked(self) -> bool:
        return self.state in (
            DockState.DOCKED,
            DockState.CHARGING,
            DockState.CHARGED,
        )

    @property
    def needs_charge(self) -> bool:
        return self.state in (
            DockState.LOW_BATTERY,
            DockState.NAVIGATING,
            DockState.HOMING,
            DockState.CRITICAL,
        )

    @property
    def dock_info(self) -> DockInfo:
        return self._dock

    def update_battery(self, battery_mv: int) -> DockState | None:
        """Feed battery reading from SensorPacket. Returns new state if changed.

        Call this on every sensor packet. The manager applies debounce
        (BATTERY_SAMPLES_FOR_DECISION consecutive readings) before
        triggering state transitions.
        """
        self._battery_mv = battery_mv
        now = time.monotonic()

        # Rate-limit checks
        if now - self._last_battery_check < BATTERY_CHECK_INTERVAL_S:
            return None
        self._last_battery_check = now

        prev_state = self.state

        if self.state == DockState.PATROL:
            self._check_patrol_battery(battery_mv)
        elif self.state == DockState.CHARGING:
            self._check_charging_battery(battery_mv)
        elif self.state == DockState.CHARGED:
            pass  # waiting for undock command

        if self.state != prev_state:
            logger.info(
                "[Dock] State: %s → %s (battery=%dmV)",
                prev_state.value,
                self.state.value,
                battery_mv,
            )
            return self.state
        return None

    def report_arrived_at_dock(self):
        """Called when robot reaches dock waypoint — switch to homing."""
        if self.state in (DockState.NAVIGATING, DockState.LOW_BATTERY):
            self.state = DockState.HOMING
            self._dock_start_time = time.monotonic()
            logger.info("[Dock] Arrived at dock area, starting IR homing")

    def report_docked(self):
        """Called when physical contact detected (charging current flowing)."""
        if self.state in (
            DockState.HOMING,
            DockState.NAVIGATING,
            DockState.LOW_BATTERY,
            DockState.CRITICAL,
        ):
            self.state = DockState.DOCKED
            logger.info("[Dock] Docked — waiting for charge to start")

    def report_charging(self):
        """Called when battery voltage starts rising (charge confirmed)."""
        if self.state == DockState.DOCKED:
            self.state = DockState.CHARGING
            self._charge_start_time = time.monotonic()
            logger.info("[Dock] Charging started")

    def report_undocked(self):
        """Called when robot drives off dock."""
        if self.state in (DockState.CHARGED, DockState.UNDOCKING):
            self.state = DockState.PATROL
            self._low_count = 0
            self._critical_count = 0
            logger.info("[Dock] Undocked — resuming patrol")

    def start_dock_sequence(self):
        """Manually trigger docking (e.g., from Telegram /dock command)."""
        if self.state == DockState.PATROL:
            self.state = DockState.LOW_BATTERY
            logger.info("[Dock] Manual dock request")

    def start_undock_sequence(self):
        """Trigger undocking from charged state."""
        if self.state in (DockState.CHARGED, DockState.CHARGING, DockState.DOCKED):
            self.state = DockState.UNDOCKING
            logger.info("[Dock] Undock sequence started")

    def abort(self):
        """Abort docking and return to patrol (e.g., if obstacle blocks dock)."""
        if self.state in (DockState.NAVIGATING, DockState.HOMING):
            self.state = DockState.PATROL
            self._low_count = 0
            logger.info("[Dock] Docking aborted")

    def check_homing_timeout(self) -> bool:
        """Check if IR homing has timed out. Returns True if timed out."""
        if self.state != DockState.HOMING:
            return False
        elapsed = time.monotonic() - self._dock_start_time
        if elapsed > DOCK_ALIGNMENT_TIMEOUT_S:
            logger.warning("[Dock] IR homing timeout after %.1fs", elapsed)
            return True
        return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _check_patrol_battery(self, battery_mv: int):
        """Check battery during patrol — trigger dock if low."""
        if battery_mv <= BATTERY_CRITICAL_MV:
            self._critical_count += 1
            if self._critical_count >= BATTERY_SAMPLES_FOR_DECISION:
                self.state = DockState.CRITICAL
                self._critical_count = 0
        elif battery_mv <= BATTERY_LOW_MV:
            self._low_count += 1
            self._critical_count = 0
            if self._low_count >= BATTERY_SAMPLES_FOR_DECISION:
                self.state = DockState.LOW_BATTERY
                self._low_count = 0
        else:
            self._low_count = 0
            self._critical_count = 0

    def _check_charging_battery(self, battery_mv: int):
        """Check battery during charging — transition to charged when full."""
        if battery_mv >= BATTERY_FULL_MV:
            self.state = DockState.CHARGED
            charge_duration = time.monotonic() - self._charge_start_time
            logger.info(
                "[Dock] Fully charged (%dmV) after %.0f min",
                battery_mv,
                charge_duration / 60,
            )

    # ── Status ────────────────────────────────────────────────────────────

    @property
    def charge_time_s(self) -> float:
        """Seconds since charging started (0 if not charging)."""
        if self.state != DockState.CHARGING or self._charge_start_time == 0:
            return 0.0
        return time.monotonic() - self._charge_start_time

    def __repr__(self) -> str:
        return (
            f"DockManager(state={self.state.value}, "
            f"battery={self._battery_mv}mV, "
            f"dock={self._dock.dock_id})"
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def dock_from_config(config: dict) -> DockInfo | None:
    """Parse dock config from config.yaml."""
    docks = config.get("docks", [])
    if not docks:
        return None
    d = docks[0]  # use first dock for single-robot setup
    wp = d.get("waypoint", {})
    return DockInfo(
        dock_id=d.get("id", "dock_0"),
        x_mm=wp.get("x_mm", 0),
        y_mm=wp.get("y_mm", 0),
        heading_cdeg=d.get("approach_heading_cdeg", 0),
    )
