# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/battery.py — battery monitoring."""

from planner.battery import (
    BatteryMonitor,
    BatteryState,
    BATTERY_NOMINAL_MAH,
    BATTERY_FULL_MV,
    BATTERY_EMPTY_MV,
    BATTERY_CELLS,
    FAILSAFE_WARNING_PCT,
    FAILSAFE_RTL_PCT,
    FAILSAFE_LAND_PCT,
    FAILSAFE_KILL_PCT,
    VOLTAGE_SAG_THRESHOLD_MV,
    HISTORY_MAX_SAMPLES,
)


class TestConstants:
    def test_nominal_mah_positive(self):
        assert BATTERY_NOMINAL_MAH > 0

    def test_full_above_empty(self):
        assert BATTERY_FULL_MV > BATTERY_EMPTY_MV

    def test_cells_positive(self):
        assert BATTERY_CELLS >= 1

    def test_failsafe_ordering(self):
        assert FAILSAFE_WARNING_PCT > FAILSAFE_RTL_PCT > FAILSAFE_LAND_PCT > FAILSAFE_KILL_PCT


class TestBatteryMonitor:
    def test_initial_state(self):
        m = BatteryMonitor()
        assert m.state.capacity_pct == 100

    def test_update_voltage(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400)
        assert m.state.voltage_mv == 7400
        assert m.state.per_cell_mv == 7400 / BATTERY_CELLS

    def test_update_current(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400, current_ma=800)
        assert m.state.current_ma == 800

    def test_capacity_from_kernel(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400, capacity_pct=72)
        assert m.state.capacity_pct == 72

    def test_capacity_from_voltage(self):
        m = BatteryMonitor()
        m.update(voltage_mv=BATTERY_FULL_MV)
        assert m.state.capacity_pct == 100
        m.update(voltage_mv=BATTERY_EMPTY_MV)
        assert m.state.capacity_pct == 0

    def test_autonomy_prediction(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400, current_ma=1000, mah_used=1800)
        assert m.state.autonomy_minutes > 0

    def test_sag_detection(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400)
        m.update(voltage_mv=7400 - VOLTAGE_SAG_THRESHOLD_MV - 100)
        assert m.state.sag_detected

    def test_no_sag_gradual(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7400)
        import time

        time.sleep(0.01)
        m.update(voltage_mv=7350)
        assert not m.state.sag_detected

    def test_status_text(self):
        m = BatteryMonitor()
        m.update(voltage_mv=7200, current_ma=500, mah_used=1200, capacity_pct=67)
        text = m.status_text()
        assert "67%" in text
        assert "7200" in text

    def test_history(self):
        m = BatteryMonitor()
        for i in range(5):
            m.update(voltage_mv=7400 - i * 100)
        assert len(m.get_history()) == 5

    def test_failsafe_in_status(self):
        m = BatteryMonitor()
        m.update(voltage_mv=6200, failsafe=3)
        assert "LAND" in m.status_text()
