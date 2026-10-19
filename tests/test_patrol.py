# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/patrol.py — patrol controller."""

import asyncio
import pytest

from perception.slam import SLAM, OccupancyGrid
from planner.path import PathPlanner
from planner.mapper import PerimeterMapper, Waypoint
from planner.patrol import (
    PatrolController,
    PatrolState,
    PATROL_WAYPOINT_REACH_MM,
    PATROL_DEFAULT_SPEED_PCT,
)
from policy.wheeled import WheeledPolicy
from protocol import ActuatorCmd


class TestPatrolController:

    def _make_patrol(self, waypoints=None):
        grid = OccupancyGrid(resolution_mm=50, size_cells=100)
        slam = SLAM(grid)
        planner = PathPlanner(grid)
        mapper = PerimeterMapper(slam, planner)

        if waypoints:
            mapper.waypoints = waypoints

        cmds_sent = []

        async def mock_send(cmd):
            cmds_sent.append(cmd)

        policy = WheeledPolicy(max_speed=80)
        patrol = PatrolController(
            slam=slam,
            path_planner=planner,
            mapper=mapper,
            send_cmd=mock_send,
            policy=policy,
        )
        return patrol, cmds_sent

    def test_initial_state(self):
        patrol, _ = self._make_patrol()
        assert patrol.state == PatrolState.IDLE

    def test_no_waypoints_error(self):
        patrol, _ = self._make_patrol(waypoints=[])

        async def run():
            await patrol.run_patrol(loop=False)
            return patrol.state

        state = asyncio.run(run())
        assert state == PatrolState.ERROR

    def test_stop_interrupts(self):
        wps = [
            Waypoint(0, 0, 0, "start"),
            Waypoint(5000, 0, 0, "far"),
        ]
        patrol, cmds = self._make_patrol(waypoints=wps)

        async def run():
            # stop immediately
            patrol.stop()
            await patrol.run_patrol(loop=False)
            return patrol.state

        state = asyncio.run(run())
        assert state == PatrolState.DONE

    def test_feed_sensors(self):
        patrol, _ = self._make_patrol()
        # should not crash
        patrol.feed_sensors(100, 0, 500, [(0, 2000)])

    def test_wrap_cdeg(self):
        assert PatrolController._wrap_cdeg(0) == 0
        assert PatrolController._wrap_cdeg(36000) == 0
        assert PatrolController._wrap_cdeg(-36000) == 0
        assert PatrolController._wrap_cdeg(27000) == -9000
        assert PatrolController._wrap_cdeg(-27000) == 9000

    def test_repr(self):
        patrol, _ = self._make_patrol()
        r = repr(patrol)
        assert "PatrolController" in r
        assert "idle" in r

    def test_patrol_count_starts_zero(self):
        patrol, _ = self._make_patrol()
        assert patrol.patrol_count == 0

    def test_current_waypoint_index(self):
        patrol, _ = self._make_patrol()
        assert patrol.current_waypoint_index == 0


class TestPatrolNavSkills:
    """Test that new patrol skills translate correctly in policy."""

    def test_map_perimeter_skill(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("MAP_PERIMETER")
        assert len(cmd.channels) == 2
        # should produce some forward motion
        assert cmd.channels[0] > 0

    def test_patrol_perimeter_skill(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("PATROL_PERIMETER")
        assert cmd.channels[0] > 0

    def test_explore_frontier_skill(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("EXPLORE_FRONTIER")
        assert cmd.channels[0] > 0

    def test_navigate_path_straight(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("NAVIGATE_PATH", {"speed": 40, "steer": 0})
        assert cmd.channels[0] == 40
        assert cmd.channels[1] == 40

    def test_navigate_path_steer_right(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("NAVIGATE_PATH", {"speed": 40, "steer": 20})
        assert cmd.channels[0] == 60  # left faster
        assert cmd.channels[1] == 20  # right slower

    def test_navigate_path_steer_left(self):
        p = WheeledPolicy(max_speed=80)
        cmd = p.translate("NAVIGATE_PATH", {"speed": 40, "steer": -20})
        assert cmd.channels[0] == 20
        assert cmd.channels[1] == 60
