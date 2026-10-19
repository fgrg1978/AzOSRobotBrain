# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/power.py — ECO/ALERT power mode manager."""

import asyncio
import time
import pytest

from planner.power import (
    PowerManager,
    PowerMode,
    PowerState,
    PATROL_SPEED_ECO_PCT,
    PATROL_SPEED_ALERT_PCT,
    LIDAR_HZ_ECO,
    LIDAR_HZ_ALERT,
    ALERT_TIMEOUT_S,
    ALERT_CLEAR_COUNT,
    POWER_CONFIG_KEY,
    CAMERA_POWER_KEY,
    WIFI_MODE_KEY,
    LIDAR_HZ_KEY,
    POWER_ECO,
    POWER_ALERT,
    CAMERA_ON,
    CAMERA_OFF,
    WIFI_BATCH,
    WIFI_CONTINUOUS,
)
from protocol import ConfigCmd, CONFIG_CMD


class TestPowerState:

    def test_defaults(self):
        s = PowerState()
        assert s.mode == PowerMode.ECO
        assert s.camera_on is False
        assert s.clear_count == 0


class TestPowerManager:

    def _make_manager(self):
        configs_sent = []

        async def mock_send(writer, pkt_type, payload):
            cmd = ConfigCmd.from_bytes(payload)
            configs_sent.append((cmd.config_key, cmd.value))

        mgr = PowerManager(mock_send)
        return mgr, configs_sent

    def test_initial_eco(self):
        mgr, _ = self._make_manager()
        assert mgr.is_eco
        assert not mgr.is_alert

    def test_patrol_speed_eco(self):
        mgr, _ = self._make_manager()
        assert mgr.patrol_speed == PATROL_SPEED_ECO_PCT

    def test_lidar_hz_eco(self):
        mgr, _ = self._make_manager()
        assert mgr.lidar_hz == LIDAR_HZ_ECO

    def test_vlm_inactive_in_eco(self):
        mgr, _ = self._make_manager()
        assert not mgr.vlm_active

    def test_trigger_alert(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("pir_motion", writer))

        assert mgr.is_alert
        assert mgr.patrol_speed == PATROL_SPEED_ALERT_PCT
        assert mgr.lidar_hz == LIDAR_HZ_ALERT
        assert mgr.vlm_active
        assert mgr.state.camera_on

        # should have sent: camera ON, wifi continuous, lidar hz, power mode
        keys_sent = [c[0] for c in configs]
        assert CAMERA_POWER_KEY in keys_sent
        assert WIFI_MODE_KEY in keys_sent
        assert LIDAR_HZ_KEY in keys_sent
        assert POWER_CONFIG_KEY in keys_sent

    def test_trigger_alert_idempotent(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("pir", writer))
        count_after_first = len(configs)
        asyncio.run(mgr.trigger_alert("pir_again", writer))
        # should not send more configs
        assert len(configs) == count_after_first

    def test_report_clear_deescalates(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))
        configs.clear()

        # report CLEAR N times
        for i in range(ALERT_CLEAR_COUNT):
            asyncio.run(mgr.report_clear(writer))

        assert mgr.is_eco
        assert not mgr.state.camera_on

        # should have sent deescalation configs
        keys_sent = [c[0] for c in configs]
        assert CAMERA_POWER_KEY in keys_sent
        assert POWER_CONFIG_KEY in keys_sent

    def test_report_clear_partial(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))

        # one clear is not enough
        asyncio.run(mgr.report_clear(writer))
        assert mgr.is_alert
        assert mgr.state.clear_count == 1

    def test_report_threat_resets_clears(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))
        asyncio.run(mgr.report_clear(writer))
        asyncio.run(mgr.report_clear(writer))
        assert mgr.state.clear_count == 2

        asyncio.run(mgr.report_threat(writer))
        assert mgr.state.clear_count == 0
        assert mgr.is_alert

    def test_report_threat_triggers_alert_from_eco(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.report_threat(writer))
        assert mgr.is_alert

    def test_timeout_deescalates(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))
        # fake the start time to simulate timeout
        mgr.state.alert_start_time = time.monotonic() - ALERT_TIMEOUT_S - 1

        result = asyncio.run(mgr.check_timeout(writer))
        assert result is True
        assert mgr.is_eco

    def test_timeout_no_deescalate_if_recent(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))
        # recent alert — should not timeout
        result = asyncio.run(mgr.check_timeout(writer))
        assert result is False
        assert mgr.is_alert

    def test_timeout_noop_in_eco(self):
        mgr, _ = self._make_manager()
        result = asyncio.run(mgr.check_timeout())
        assert result is False

    def test_force_eco(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.trigger_alert("test", writer))
        asyncio.run(mgr.force_eco(writer))
        assert mgr.is_eco

    def test_force_eco_noop_if_already(self):
        mgr, configs = self._make_manager()
        asyncio.run(mgr.force_eco())
        assert len(configs) == 0

    def test_no_writer_no_send(self):
        mgr, configs = self._make_manager()
        asyncio.run(mgr.trigger_alert("test"))  # no writer
        assert mgr.is_alert
        assert len(configs) == 0  # nothing sent

    def test_repr(self):
        mgr, _ = self._make_manager()
        r = repr(mgr)
        assert "PowerManager" in r
        assert "eco" in r
        assert "OFF" in r
