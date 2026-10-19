# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/docking.py — auto-docking & charging manager."""

import time
import pytest

from planner.docking import (
    DockManager,
    DockState,
    DockInfo,
    dock_from_config,
    BATTERY_LOW_MV,
    BATTERY_CRITICAL_MV,
    BATTERY_FULL_MV,
    BATTERY_HYSTERESIS_MV,
    DOCK_APPROACH_SPEED_PCT,
    DOCK_IR_HOMING_DISTANCE_MM,
    DOCK_ALIGNMENT_TIMEOUT_S,
    DOCK_CHARGE_CHECK_INTERVAL_S,
    DOCK_UNDOCK_DRIVE_S,
    DOCK_UNDOCK_SPEED_PCT,
    BATTERY_CHECK_INTERVAL_S,
    BATTERY_SAMPLES_FOR_DECISION,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_battery_low_mv(self):
        assert BATTERY_LOW_MV > BATTERY_CRITICAL_MV

    def test_battery_critical_below_low(self):
        assert BATTERY_CRITICAL_MV < BATTERY_LOW_MV

    def test_battery_full_above_low(self):
        assert BATTERY_FULL_MV > BATTERY_LOW_MV

    def test_hysteresis_positive(self):
        assert BATTERY_HYSTERESIS_MV > 0

    def test_dock_approach_speed(self):
        assert 0 < DOCK_APPROACH_SPEED_PCT <= 100

    def test_dock_ir_homing_distance(self):
        assert DOCK_IR_HOMING_DISTANCE_MM > 0

    def test_dock_alignment_timeout(self):
        assert DOCK_ALIGNMENT_TIMEOUT_S > 0

    def test_charge_check_interval(self):
        assert DOCK_CHARGE_CHECK_INTERVAL_S > 0

    def test_undock_drive_time(self):
        assert DOCK_UNDOCK_DRIVE_S > 0

    def test_undock_speed(self):
        assert 0 < DOCK_UNDOCK_SPEED_PCT <= 100

    def test_battery_check_interval(self):
        assert BATTERY_CHECK_INTERVAL_S > 0

    def test_samples_for_decision(self):
        assert BATTERY_SAMPLES_FOR_DECISION >= 1


# ---------------------------------------------------------------------------
# DockInfo
# ---------------------------------------------------------------------------


class TestDockInfo:
    def test_defaults(self):
        d = DockInfo()
        assert d.dock_id == "dock_0"
        assert d.x_mm == 0
        assert d.y_mm == 0
        assert d.heading_cdeg == 0

    def test_custom(self):
        d = DockInfo(dock_id="dock_north", x_mm=5000, y_mm=3000, heading_cdeg=18000)
        assert d.dock_id == "dock_north"
        assert d.x_mm == 5000
        assert d.heading_cdeg == 18000


# ---------------------------------------------------------------------------
# DockManager — initial state
# ---------------------------------------------------------------------------


class TestDockManagerInit:
    def test_initial_state_patrol(self):
        dm = DockManager()
        assert dm.state == DockState.PATROL

    def test_initial_battery_zero(self):
        dm = DockManager()
        assert dm.battery_mv == 0

    def test_not_docked_initially(self):
        dm = DockManager()
        assert not dm.is_docked

    def test_no_charge_needed_initially(self):
        dm = DockManager()
        assert not dm.needs_charge

    def test_repr(self):
        dm = DockManager()
        r = repr(dm)
        assert "patrol" in r
        assert "dock_0" in r


# ---------------------------------------------------------------------------
# DockManager — battery monitoring
# ---------------------------------------------------------------------------


class TestDockManagerBattery:
    def _force_check(self, dm: DockManager, mv: int) -> DockState | None:
        """Force a battery check by resetting the timer."""
        dm._last_battery_check = 0
        return dm.update_battery(mv)

    def test_healthy_battery_stays_patrol(self):
        dm = DockManager()
        result = self._force_check(dm, 7500)
        assert dm.state == DockState.PATROL

    def test_single_low_reading_no_transition(self):
        """One low reading is not enough — need BATTERY_SAMPLES_FOR_DECISION."""
        dm = DockManager()
        self._force_check(dm, BATTERY_LOW_MV - 100)
        assert dm.state == DockState.PATROL

    def test_consecutive_low_triggers_low_battery(self):
        dm = DockManager()
        for _ in range(BATTERY_SAMPLES_FOR_DECISION):
            self._force_check(dm, BATTERY_LOW_MV - 100)
        assert dm.state == DockState.LOW_BATTERY

    def test_consecutive_critical_triggers_critical(self):
        dm = DockManager()
        for _ in range(BATTERY_SAMPLES_FOR_DECISION):
            self._force_check(dm, BATTERY_CRITICAL_MV - 100)
        assert dm.state == DockState.CRITICAL

    def test_recovery_resets_low_count(self):
        """If battery recovers between low readings, counter resets."""
        dm = DockManager()
        self._force_check(dm, BATTERY_LOW_MV - 100)
        self._force_check(dm, BATTERY_LOW_MV + 500)  # recovery
        self._force_check(dm, BATTERY_LOW_MV - 100)
        assert dm.state == DockState.PATROL  # not enough consecutive

    def test_charging_to_charged(self):
        dm = DockManager()
        dm.state = DockState.CHARGING
        dm._charge_start_time = time.monotonic() - 60
        result = self._force_check(dm, BATTERY_FULL_MV)
        assert dm.state == DockState.CHARGED

    def test_charging_below_full_stays(self):
        dm = DockManager()
        dm.state = DockState.CHARGING
        dm._charge_start_time = time.monotonic()
        self._force_check(dm, BATTERY_FULL_MV - 200)
        assert dm.state == DockState.CHARGING

    def test_rate_limiting(self):
        """Checks within BATTERY_CHECK_INTERVAL_S are skipped."""
        dm = DockManager()
        dm._last_battery_check = time.monotonic()  # just checked
        result = dm.update_battery(BATTERY_LOW_MV - 500)
        assert result is None
        assert dm.state == DockState.PATROL


# ---------------------------------------------------------------------------
# DockManager — state machine transitions
# ---------------------------------------------------------------------------


class TestDockManagerStateMachine:
    def test_report_arrived_at_dock(self):
        dm = DockManager()
        dm.state = DockState.NAVIGATING
        dm.report_arrived_at_dock()
        assert dm.state == DockState.HOMING

    def test_report_arrived_from_low_battery(self):
        dm = DockManager()
        dm.state = DockState.LOW_BATTERY
        dm.report_arrived_at_dock()
        assert dm.state == DockState.HOMING

    def test_report_arrived_wrong_state_noop(self):
        dm = DockManager()
        dm.state = DockState.PATROL
        dm.report_arrived_at_dock()
        assert dm.state == DockState.PATROL

    def test_report_docked(self):
        dm = DockManager()
        dm.state = DockState.HOMING
        dm.report_docked()
        assert dm.state == DockState.DOCKED
        assert dm.is_docked

    def test_report_docked_from_critical(self):
        dm = DockManager()
        dm.state = DockState.CRITICAL
        dm.report_docked()
        assert dm.state == DockState.DOCKED

    def test_report_charging(self):
        dm = DockManager()
        dm.state = DockState.DOCKED
        dm.report_charging()
        assert dm.state == DockState.CHARGING

    def test_report_charging_wrong_state(self):
        dm = DockManager()
        dm.state = DockState.PATROL
        dm.report_charging()
        assert dm.state == DockState.PATROL

    def test_report_undocked(self):
        dm = DockManager()
        dm.state = DockState.CHARGED
        dm.report_undocked()
        assert dm.state == DockState.PATROL
        assert not dm.is_docked

    def test_report_undocked_from_undocking(self):
        dm = DockManager()
        dm.state = DockState.UNDOCKING
        dm.report_undocked()
        assert dm.state == DockState.PATROL

    def test_report_undocked_resets_counters(self):
        dm = DockManager()
        dm._low_count = 5
        dm._critical_count = 3
        dm.state = DockState.CHARGED
        dm.report_undocked()
        assert dm._low_count == 0
        assert dm._critical_count == 0


# ---------------------------------------------------------------------------
# DockManager — manual commands
# ---------------------------------------------------------------------------


class TestDockManagerManual:
    def test_start_dock_sequence(self):
        dm = DockManager()
        dm.start_dock_sequence()
        assert dm.state == DockState.LOW_BATTERY

    def test_start_dock_wrong_state(self):
        dm = DockManager()
        dm.state = DockState.CHARGING
        dm.start_dock_sequence()
        assert dm.state == DockState.CHARGING  # no change

    def test_start_undock_from_charged(self):
        dm = DockManager()
        dm.state = DockState.CHARGED
        dm.start_undock_sequence()
        assert dm.state == DockState.UNDOCKING

    def test_start_undock_from_charging(self):
        dm = DockManager()
        dm.state = DockState.CHARGING
        dm.start_undock_sequence()
        assert dm.state == DockState.UNDOCKING

    def test_start_undock_from_docked(self):
        dm = DockManager()
        dm.state = DockState.DOCKED
        dm.start_undock_sequence()
        assert dm.state == DockState.UNDOCKING

    def test_abort_from_navigating(self):
        dm = DockManager()
        dm.state = DockState.NAVIGATING
        dm.abort()
        assert dm.state == DockState.PATROL

    def test_abort_from_homing(self):
        dm = DockManager()
        dm.state = DockState.HOMING
        dm.abort()
        assert dm.state == DockState.PATROL

    def test_abort_resets_low_count(self):
        dm = DockManager()
        dm._low_count = 5
        dm.state = DockState.NAVIGATING
        dm.abort()
        assert dm._low_count == 0


# ---------------------------------------------------------------------------
# DockManager — homing timeout
# ---------------------------------------------------------------------------


class TestDockManagerHomingTimeout:
    def test_no_timeout_when_not_homing(self):
        dm = DockManager()
        assert not dm.check_homing_timeout()

    def test_no_timeout_when_recent(self):
        dm = DockManager()
        dm.state = DockState.HOMING
        dm._dock_start_time = time.monotonic()
        assert not dm.check_homing_timeout()

    def test_timeout_when_expired(self):
        dm = DockManager()
        dm.state = DockState.HOMING
        dm._dock_start_time = time.monotonic() - DOCK_ALIGNMENT_TIMEOUT_S - 1
        assert dm.check_homing_timeout()


# ---------------------------------------------------------------------------
# DockManager — full cycle
# ---------------------------------------------------------------------------


class TestDockManagerFullCycle:
    def _force_check(self, dm, mv):
        dm._last_battery_check = 0
        return dm.update_battery(mv)

    def test_full_dock_undock_cycle(self):
        """Test complete: patrol → low → navigate → homing → docked → charging → charged → undock → patrol."""
        dm = DockManager()
        assert dm.state == DockState.PATROL

        # Battery drops
        for _ in range(BATTERY_SAMPLES_FOR_DECISION):
            self._force_check(dm, BATTERY_LOW_MV - 100)
        assert dm.state == DockState.LOW_BATTERY
        assert dm.needs_charge

        # Start navigating
        dm.state = DockState.NAVIGATING

        # Arrive at dock
        dm.report_arrived_at_dock()
        assert dm.state == DockState.HOMING

        # Dock contact
        dm.report_docked()
        assert dm.state == DockState.DOCKED
        assert dm.is_docked

        # Charging starts
        dm.report_charging()
        assert dm.state == DockState.CHARGING

        # Battery rises
        self._force_check(dm, BATTERY_FULL_MV)
        assert dm.state == DockState.CHARGED

        # Undock
        dm.start_undock_sequence()
        assert dm.state == DockState.UNDOCKING
        dm.report_undocked()
        assert dm.state == DockState.PATROL
        assert not dm.is_docked
        assert not dm.needs_charge


# ---------------------------------------------------------------------------
# DockManager — properties
# ---------------------------------------------------------------------------


class TestDockManagerProperties:
    def test_is_docked_states(self):
        dm = DockManager()
        for state in [DockState.DOCKED, DockState.CHARGING, DockState.CHARGED]:
            dm.state = state
            assert dm.is_docked

    def test_not_docked_states(self):
        dm = DockManager()
        for state in [
            DockState.PATROL,
            DockState.LOW_BATTERY,
            DockState.NAVIGATING,
            DockState.HOMING,
            DockState.UNDOCKING,
            DockState.CRITICAL,
        ]:
            dm.state = state
            assert not dm.is_docked

    def test_needs_charge_states(self):
        dm = DockManager()
        for state in [
            DockState.LOW_BATTERY,
            DockState.NAVIGATING,
            DockState.HOMING,
            DockState.CRITICAL,
        ]:
            dm.state = state
            assert dm.needs_charge

    def test_charge_time_zero_when_not_charging(self):
        dm = DockManager()
        assert dm.charge_time_s == 0.0

    def test_charge_time_positive_when_charging(self):
        dm = DockManager()
        dm.state = DockState.CHARGING
        dm._charge_start_time = time.monotonic() - 120
        assert dm.charge_time_s >= 119.0


# ---------------------------------------------------------------------------
# dock_from_config
# ---------------------------------------------------------------------------


class TestDockFromConfig:
    def test_empty_config(self):
        assert dock_from_config({}) is None

    def test_empty_docks_list(self):
        assert dock_from_config({"docks": []}) is None

    def test_single_dock(self):
        config = {
            "docks": [
                {
                    "id": "dock_north",
                    "waypoint": {"x_mm": 1000, "y_mm": 2000},
                    "approach_heading_cdeg": 18000,
                }
            ],
        }
        dock = dock_from_config(config)
        assert dock is not None
        assert dock.dock_id == "dock_north"
        assert dock.x_mm == 1000
        assert dock.y_mm == 2000
        assert dock.heading_cdeg == 18000

    def test_multiple_docks_uses_first(self):
        config = {
            "docks": [
                {"id": "dock_a", "waypoint": {"x_mm": 100, "y_mm": 200}},
                {"id": "dock_b", "waypoint": {"x_mm": 300, "y_mm": 400}},
            ],
        }
        dock = dock_from_config(config)
        assert dock.dock_id == "dock_a"

    def test_defaults_for_missing_fields(self):
        config = {"docks": [{"id": "d"}]}
        dock = dock_from_config(config)
        assert dock.x_mm == 0
        assert dock.y_mm == 0
        assert dock.heading_cdeg == 0
