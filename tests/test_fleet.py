# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/fleet.py — fleet management."""

import time

from planner.fleet import (
    FleetPlanner,
    RobotEntry,
    DispatchResult,
    ROBOT_HEARTBEAT_TIMEOUT_S,
    FLEET_REBALANCE_INTERVAL_S,
    DISPATCH_COOLDOWN_S,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_heartbeat_timeout_positive(self):
        assert ROBOT_HEARTBEAT_TIMEOUT_S > 0

    def test_rebalance_interval_positive(self):
        assert FLEET_REBALANCE_INTERVAL_S > 0

    def test_dispatch_cooldown_positive(self):
        assert DISPATCH_COOLDOWN_S > 0


# ---------------------------------------------------------------------------
# RobotEntry
# ---------------------------------------------------------------------------


class TestRobotEntry:
    def test_defaults(self):
        r = RobotEntry(robot_id="bot1")
        assert r.robot_id == "bot1"
        assert not r.online
        assert not r.docked
        assert not r.busy
        assert r.zones == []


# ---------------------------------------------------------------------------
# FleetPlanner — registration
# ---------------------------------------------------------------------------


class TestFleetRegistration:
    def test_register(self):
        fp = FleetPlanner()
        fp.register("r1", port=9000)
        assert fp.robot_count == 1
        assert fp.get_robot("r1") is not None

    def test_register_with_zones(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north", "east"])
        assert fp.get_robot("r1").zones == ["north", "east"]

    def test_register_multiple(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.register("r2")
        fp.register("r3")
        assert fp.robot_count == 3

    def test_unregister(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.unregister("r1")
        assert fp.robot_count == 0

    def test_unregister_nonexistent(self):
        fp = FleetPlanner()
        fp.unregister("nope")  # no error
        assert fp.robot_count == 0

    def test_get_robot_not_found(self):
        fp = FleetPlanner()
        assert fp.get_robot("nope") is None


# ---------------------------------------------------------------------------
# FleetPlanner — heartbeat
# ---------------------------------------------------------------------------


class TestFleetHeartbeat:
    def test_heartbeat_sets_online(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1", x_mm=100, y_mm=200, battery_mv=7500)
        r = fp.get_robot("r1")
        assert r.online
        assert r.x_mm == 100
        assert r.y_mm == 200
        assert r.battery_mv == 7500

    def test_heartbeat_unknown_robot(self):
        fp = FleetPlanner()
        fp.heartbeat("unknown")  # no error

    def test_online_count(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.register("r2")
        assert fp.online_count == 0
        fp.heartbeat("r1")
        assert fp.online_count == 1
        fp.heartbeat("r2")
        assert fp.online_count == 2

    def test_check_timeouts_marks_offline(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        fp.get_robot("r1").last_heartbeat = time.monotonic() - ROBOT_HEARTBEAT_TIMEOUT_S - 1
        offline = fp.check_timeouts()
        assert "r1" in offline
        assert not fp.get_robot("r1").online

    def test_check_timeouts_online_stays(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        offline = fp.check_timeouts()
        assert offline == []
        assert fp.get_robot("r1").online


# ---------------------------------------------------------------------------
# FleetPlanner — zone assignment
# ---------------------------------------------------------------------------


class TestFleetZones:
    def test_assign_zone(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.assign_zone("r1", "north")
        assert "north" in fp.get_robot("r1").zones

    def test_assign_zone_no_duplicates(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.assign_zone("r1", "north")
        fp.assign_zone("r1", "north")
        assert fp.get_robot("r1").zones.count("north") == 1

    def test_unassign_zone(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north", "east"])
        fp.unassign_zone("r1", "north")
        assert "north" not in fp.get_robot("r1").zones
        assert "east" in fp.get_robot("r1").zones

    def test_get_zone_owner(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north"])
        fp.register("r2", zones=["south"])
        assert fp.get_zone_owner("north") == "r1"
        assert fp.get_zone_owner("south") == "r2"
        assert fp.get_zone_owner("west") is None

    def test_uncovered_zones(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north"])
        fp.heartbeat("r1")
        all_zones = ["north", "south", "east"]
        uncovered = fp.uncovered_zones(all_zones)
        assert "south" in uncovered
        assert "east" in uncovered
        assert "north" not in uncovered

    def test_uncovered_ignores_docked(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north"])
        fp.heartbeat("r1", docked=True)
        uncovered = fp.uncovered_zones(["north"])
        assert "north" in uncovered  # docked robot doesn't cover


# ---------------------------------------------------------------------------
# FleetPlanner — dispatch
# ---------------------------------------------------------------------------


class TestFleetDispatch:
    def test_dispatch_nearest(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.register("r2")
        fp.heartbeat("r1", x_mm=0, y_mm=0)
        fp.heartbeat("r2", x_mm=5000, y_mm=5000)
        result = fp.dispatch_nearest(100, 100)
        assert result is not None
        assert result.robot_id == "r1"
        assert result.distance_mm < 200

    def test_dispatch_no_available(self):
        fp = FleetPlanner()
        result = fp.dispatch_nearest(0, 0)
        assert result is None

    def test_dispatch_skips_offline(self):
        fp = FleetPlanner()
        fp.register("r1")  # offline
        result = fp.dispatch_nearest(0, 0)
        assert result is None

    def test_dispatch_skips_docked(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1", docked=True)
        result = fp.dispatch_nearest(0, 0)
        assert result is None

    def test_dispatch_skips_busy(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        fp.get_robot("r1").busy = True
        result = fp.dispatch_nearest(0, 0)
        assert result is None

    def test_dispatch_marks_busy(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        fp.dispatch_nearest(0, 0)
        assert fp.get_robot("r1").busy

    def test_dispatch_complete_clears_busy(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        fp.dispatch_nearest(0, 0)
        fp.report_dispatch_complete("r1")
        assert not fp.get_robot("r1").busy

    def test_dispatch_cooldown(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        fp.dispatch_nearest(0, 0)
        fp.report_dispatch_complete("r1")
        # Immediately after — should be on cooldown
        result = fp.dispatch_nearest(0, 0)
        assert result is None

    def test_dispatch_with_zone(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.heartbeat("r1")
        result = fp.dispatch_nearest(0, 0, zone="garden")
        assert result.zone == "garden"


# ---------------------------------------------------------------------------
# FleetPlanner — queries
# ---------------------------------------------------------------------------


class TestFleetQueries:
    def test_all_robots(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.register("r2")
        assert len(fp.all_robots()) == 2

    def test_online_robots(self):
        fp = FleetPlanner()
        fp.register("r1")
        fp.register("r2")
        fp.heartbeat("r1")
        assert len(fp.online_robots()) == 1

    def test_fleet_summary(self):
        fp = FleetPlanner()
        fp.register("r1", zones=["north"])
        fp.heartbeat("r1", battery_mv=7500)
        summary = fp.fleet_summary()
        assert summary["total"] == 1
        assert summary["online"] == 1
        assert summary["robots"][0]["id"] == "r1"
        assert summary["robots"][0]["battery_mv"] == 7500

    def test_repr(self):
        fp = FleetPlanner()
        fp.register("r1")
        r = repr(fp)
        assert "robots=1" in r


# ---------------------------------------------------------------------------
# FleetPlanner — config loading
# ---------------------------------------------------------------------------


class TestFleetConfig:
    def test_load_from_config(self):
        config = {
            "fleet": {
                "enabled": True,
                "robots": [
                    {"id": "robot_1", "port": 9000, "zones": ["north", "east"], "dock": "dock_n"},
                    {"id": "robot_2", "port": 9001, "zones": ["south"], "dock": "dock_s"},
                ],
            },
        }
        fp = FleetPlanner(config)
        assert fp.robot_count == 2
        r1 = fp.get_robot("robot_1")
        assert r1.port == 9000
        assert r1.zones == ["north", "east"]
        assert r1.dock_id == "dock_n"

    def test_disabled_fleet_no_robots(self):
        config = {"fleet": {"enabled": False, "robots": [{"id": "r1"}]}}
        fp = FleetPlanner(config)
        assert fp.robot_count == 0

    def test_no_fleet_config(self):
        fp = FleetPlanner({})
        assert fp.robot_count == 0

    def test_none_config(self):
        fp = FleetPlanner(None)
        assert fp.robot_count == 0
