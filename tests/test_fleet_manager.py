# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for fleet.py — per-connection FleetManager (E07).

Do not confuse with tests/test_fleet.py which covers planner/fleet.py
(FleetPlanner — zone dispatch).
"""

import asyncio
import time

import pytest

from fleet import (
    FleetManager,
    RobotRecord,
    FLEET_HEARTBEAT_TIMEOUT_S,
    FLEET_MAX_ROBOTS,
    FLEET_DEFAULT_ROBOT_TYPE,
    FLEET_DEFAULT_ROBOT_PORT,
    FLEET_UNSET_TIMESTAMP,
)

# ---------------------------------------------------------------------------
# Helpers — mock send_fn and mock writer
# ---------------------------------------------------------------------------


class _MockWriter:
    """Stand-in for asyncio.StreamWriter — records nothing, always succeeds."""

    def __init__(self, name: str = ""):
        self.name = name


class _CaptureSender:
    """Captures calls for later assertions."""

    def __init__(self):
        self.calls: list[tuple[object, int, bytes]] = []
        self.fail_for: set[object] = set()

    async def __call__(self, writer, pkt_type: int, payload: bytes) -> None:
        if writer in self.fail_for:
            raise ConnectionError("simulated send failure")
        self.calls.append((writer, pkt_type, payload))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_heartbeat_timeout_positive(self):
        assert FLEET_HEARTBEAT_TIMEOUT_S > 0

    def test_max_robots_positive(self):
        assert FLEET_MAX_ROBOTS > 0

    def test_default_robot_type_int(self):
        assert isinstance(FLEET_DEFAULT_ROBOT_TYPE, int)

    def test_default_port_positive(self):
        assert FLEET_DEFAULT_ROBOT_PORT > 0


# ---------------------------------------------------------------------------
# RobotRecord
# ---------------------------------------------------------------------------


class TestRobotRecord:
    def test_defaults(self):
        r = RobotRecord(robot_id="bot")
        assert r.robot_id == "bot"
        assert not r.online
        assert r.battery_mv == 0
        assert r.location == (0.0, 0.0)

    def test_to_dict_roundtrip_keys(self):
        r = RobotRecord(
            robot_id="bot", name="N", robot_type=1, battery_mv=7200, location=(1.0, 2.0)
        )
        d = r.to_dict()
        assert d["id"] == "bot"
        assert d["type"] == 1
        assert d["battery_mv"] == 7200
        assert d["location"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_adds_robot(self):
        fm = FleetManager()
        fm.register("r1", robot_type=0, name="R1", writer=_MockWriter("r1"))
        assert fm.count == 1
        rec = fm.get("r1")
        assert rec is not None
        assert rec.online
        assert rec.name == "R1"

    def test_register_without_writer_offline(self):
        fm = FleetManager()
        fm.register("r1")
        rec = fm.get("r1")
        assert rec is not None
        assert not rec.online
        assert rec.last_seen == FLEET_UNSET_TIMESTAMP

    def test_register_reconnect_updates_writer(self):
        fm = FleetManager()
        fm.register("r1")
        new_writer = _MockWriter("new")
        rec = fm.register("r1", writer=new_writer, name="Updated")
        assert rec.writer is new_writer
        assert rec.online
        assert rec.name == "Updated"

    def test_register_overflow_raises(self):
        fm = FleetManager(max_robots=2)
        fm.register("r1")
        fm.register("r2")
        with pytest.raises(RuntimeError):
            fm.register("r3")

    def test_unregister(self):
        fm = FleetManager()
        fm.register("r1")
        assert fm.unregister("r1") is True
        assert fm.count == 0

    def test_unregister_missing_returns_false(self):
        fm = FleetManager()
        assert fm.unregister("nope") is False

    def test_mark_disconnected(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        fm.mark_disconnected("r1")
        rec = fm.get("r1")
        assert rec is not None
        assert not rec.online
        assert rec.writer is None
        # still registered
        assert fm.count == 1


# ---------------------------------------------------------------------------
# Heartbeat & timeout
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_heartbeat_unknown_returns_false(self):
        fm = FleetManager()
        assert fm.heartbeat("nope") is False

    def test_heartbeat_updates_telemetry(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        ok = fm.heartbeat("r1", battery_mv=7400, location=(10.0, 20.0))
        assert ok
        rec = fm.get("r1")
        assert rec.battery_mv == 7400
        assert rec.location == (10.0, 20.0)
        assert rec.online

    def test_check_timeouts_no_timeouts(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        fm.heartbeat("r1")
        assert fm.check_timeouts() == []
        assert fm.get("r1").online

    def test_check_timeouts_marks_offline(self):
        fm = FleetManager(heartbeat_timeout_s=0.01)
        fm.register("r1", writer=_MockWriter())
        fm.heartbeat("r1")
        time.sleep(0.05)
        offline = fm.check_timeouts()
        assert "r1" in offline
        assert not fm.get("r1").online

    def test_check_timeouts_skips_never_seen(self):
        fm = FleetManager(heartbeat_timeout_s=0.01)
        fm.register("r1")  # no writer, last_seen == 0
        time.sleep(0.05)
        # still offline (never online) — not counted as "newly offline"
        assert fm.check_timeouts() == []

    def test_check_timeouts_idempotent(self):
        fm = FleetManager(heartbeat_timeout_s=0.01)
        fm.register("r1", writer=_MockWriter())
        fm.heartbeat("r1")
        time.sleep(0.05)
        first = fm.check_timeouts()
        second = fm.check_timeouts()
        assert "r1" in first
        assert second == []  # already offline, not "newly" offline


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_online_and_all(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        fm.register("r2")
        assert fm.count == 2
        assert fm.online_count == 1
        assert len(fm.all_robots()) == 2
        assert len(fm.online_robots()) == 1

    def test_get_fleet_status_shape(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter(), robot_type=1, name="Drone")
        fm.heartbeat("r1", battery_mv=7100)
        snap = fm.get_fleet_status()
        assert snap["total"] == 1
        assert snap["online"] == 1
        assert "r1" in snap["robots"]
        assert snap["robots"]["r1"]["battery_mv"] == 7100
        assert snap["robots"]["r1"]["type"] == 1

    def test_repr_has_counts(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        r = repr(fm)
        assert "total=1" in r
        assert "online=1" in r


# ---------------------------------------------------------------------------
# Targeted send
# ---------------------------------------------------------------------------


class TestSendTargeted:
    def test_send_to_missing_returns_false(self):
        fm = FleetManager(send_fn=_CaptureSender())
        ok = asyncio.run(fm.send_targeted("nope", 0x80, b""))
        assert not ok

    def test_send_to_offline_returns_false(self):
        fm = FleetManager(send_fn=_CaptureSender())
        fm.register("r1")  # no writer
        ok = asyncio.run(fm.send_targeted("r1", 0x80, b""))
        assert not ok

    def test_send_to_online_succeeds(self):
        sender = _CaptureSender()
        fm = FleetManager(send_fn=sender)
        w = _MockWriter("w1")
        fm.register("r1", writer=w)
        ok = asyncio.run(fm.send_targeted("r1", 0x80, b"\x01\x02"))
        assert ok
        assert len(sender.calls) == 1
        assert sender.calls[0] == (w, 0x80, b"\x01\x02")

    def test_send_no_send_fn_returns_false(self):
        fm = FleetManager()  # no send_fn
        fm.register("r1", writer=_MockWriter())
        ok = asyncio.run(fm.send_targeted("r1", 0x80, b""))
        assert not ok

    def test_send_on_error_marks_offline(self):
        sender = _CaptureSender()
        w = _MockWriter("w1")
        sender.fail_for.add(w)
        fm = FleetManager(send_fn=sender)
        fm.register("r1", writer=w)
        ok = asyncio.run(fm.send_targeted("r1", 0x80, b""))
        assert not ok
        assert not fm.get("r1").online


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_broadcast_all_online(self):
        sender = _CaptureSender()
        fm = FleetManager(send_fn=sender)
        w1, w2 = _MockWriter("w1"), _MockWriter("w2")
        fm.register("r1", writer=w1)
        fm.register("r2", writer=w2)
        results = asyncio.run(fm.broadcast(0x81, b"\xaa"))
        assert results == {"r1": True, "r2": True}
        assert len(sender.calls) == 2

    def test_broadcast_filters_offline(self):
        sender = _CaptureSender()
        fm = FleetManager(send_fn=sender)
        fm.register("r1", writer=_MockWriter("w1"))
        fm.register("r2")  # offline
        results = asyncio.run(fm.broadcast(0x81, b""))
        assert "r1" in results
        assert "r2" not in results
        assert len(sender.calls) == 1

    def test_broadcast_filters_by_type(self):
        sender = _CaptureSender()
        fm = FleetManager(send_fn=sender)
        fm.register("wheeled", writer=_MockWriter(), robot_type=0)
        fm.register("drone", writer=_MockWriter(), robot_type=1)
        results = asyncio.run(fm.broadcast(0x81, b"", robot_type=1))
        assert list(results.keys()) == ["drone"]

    def test_broadcast_partial_failure(self):
        sender = _CaptureSender()
        w_bad = _MockWriter("w_bad")
        w_ok = _MockWriter("w_ok")
        sender.fail_for.add(w_bad)
        fm = FleetManager(send_fn=sender)
        fm.register("bad", writer=w_bad)
        fm.register("ok", writer=w_ok)
        results = asyncio.run(fm.broadcast(0x81, b""))
        assert results["bad"] is False
        assert results["ok"] is True
        assert not fm.get("bad").online

    def test_broadcast_empty_fleet(self):
        fm = FleetManager(send_fn=_CaptureSender())
        results = asyncio.run(fm.broadcast(0x81, b""))
        assert results == {}

    def test_broadcast_no_send_fn(self):
        fm = FleetManager()
        fm.register("r1", writer=_MockWriter())
        results = asyncio.run(fm.broadcast(0x81, b""))
        assert results == {}


# ---------------------------------------------------------------------------
# Integration-lite: inject send_fn after construction
# ---------------------------------------------------------------------------


class TestInjectSendFn:
    def test_set_send_fn_after_init(self):
        fm = FleetManager()
        sender = _CaptureSender()
        fm.set_send_fn(sender)
        fm.register("r1", writer=_MockWriter())
        ok = asyncio.run(fm.send_targeted("r1", 0x80, b""))
        assert ok
