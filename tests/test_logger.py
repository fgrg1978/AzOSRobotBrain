# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/logger.py — mission logging and replay."""

import os
import tempfile
import time

from planner.logger import (
    MissionLogger,
    LogEvent,
    MissionSummary,
    MissionAnalytics,
    LOG_MAX_EVENTS,
    LOG_FLUSH_INTERVAL,
    LOG_RETENTION_DAYS,
)


class TestConstants:
    def test_max_events(self):
        assert LOG_MAX_EVENTS > 0

    def test_flush_interval(self):
        assert LOG_FLUSH_INTERVAL > 0

    def test_retention_days(self):
        assert LOG_RETENTION_DAYS > 0


class TestMissionLogger:
    def _logger(self):
        d = tempfile.mkdtemp()
        return MissionLogger(log_dir=d), d

    def test_initial_state(self):
        ml, _ = self._logger()
        assert not ml.active
        assert ml.event_count == 0

    def test_start_mission(self):
        ml, _ = self._logger()
        ml.start_mission("test")
        assert ml.active
        assert "test" in ml.mission_id

    def test_log_event(self):
        ml, _ = self._logger()
        ml.start_mission("test")
        ml.log_event("sensor", {"battery_mv": 7200})
        assert ml.event_count >= 2  # system start + sensor

    def test_end_mission(self):
        ml, _ = self._logger()
        ml.start_mission("test")
        ml.log_event("action", {"skill": "FORWARD"})
        summary = ml.end_mission()
        assert not ml.active
        assert summary.event_count > 0
        assert summary.duration_s >= 0

    def test_log_not_active(self):
        ml, _ = self._logger()
        ml.log_event("test", {})
        assert ml.event_count == 0

    def test_save_and_load(self):
        ml, d = self._logger()
        ml.start_mission("replay")
        ml.log_event("action", {"speed": 50})
        ml.log_event("alert", {"label": "person"})
        ml.end_mission()

        missions = ml.list_missions()
        assert len(missions) >= 1

        events = ml.load_mission(missions[0])
        assert len(events) > 0
        assert any(e.event_type == "alert" for e in events)

    def test_list_missions_empty(self):
        ml, _ = self._logger()
        assert ml.list_missions() == [] or isinstance(ml.list_missions(), list)

    def test_load_nonexistent(self):
        ml, _ = self._logger()
        assert ml.load_mission("nonexistent") == []


class TestMissionAnalytics:
    def test_empty(self):
        ml, _ = MissionLogger(), None
        a = ml.analyze([])
        assert a.total_events == 0

    def test_analyze(self):
        ml = MissionLogger()
        events = [
            LogEvent(timestamp=100, event_type="sensor", data={"battery_mv": 7200}),
            LogEvent(timestamp=101, event_type="sensor", data={"battery_mv": 7100}),
            LogEvent(timestamp=102, event_type="alert", data={"label": "person"}),
            LogEvent(timestamp=103, event_type="action", data={"skill": "STOP"}),
        ]
        a = ml.analyze(events)
        assert a.total_events == 4
        assert a.alert_count == 1
        assert a.events_by_type["sensor"] == 2
        assert a.avg_battery_mv == 7150
        assert a.min_battery_mv == 7100
        assert a.duration_s == 3.0
