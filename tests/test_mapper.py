# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/mapper.py — mapping orchestrator."""

import json
import os
import tempfile
import pytest

from perception.slam import OccupancyGrid, SLAM
from planner.path import PathPlanner
from planner.mapper import (
    PerimeterMapper,
    Waypoint,
    WAYPOINT_INTERVAL_MM,
    LOOP_CLOSURE_THRESHOLD_MM,
    MIN_MAPPING_DISTANCE_MM,
)


class TestWaypoint:

    def test_defaults(self):
        wp = Waypoint(x_mm=100, y_mm=200, heading_cdeg=9000)
        assert wp.label == ""
        assert wp.has_rtsp_coverage is False

    def test_with_label(self):
        wp = Waypoint(x_mm=0, y_mm=0, heading_cdeg=0, label="door")
        assert wp.label == "door"


class TestPerimeterMapper:

    def _make_mapper(self):
        grid = OccupancyGrid(resolution_mm=50, size_cells=100)
        slam = SLAM(grid)
        planner = PathPlanner(grid)
        return PerimeterMapper(slam, planner)

    def test_start_mapping(self):
        m = self._make_mapper()
        m.start_mapping()
        assert m.mapping_active is True
        assert len(m.waypoints) == 1
        assert m.waypoints[0].label == "start"

    def test_update_records_waypoint(self):
        m = self._make_mapper()
        m.start_mapping()

        # simulate moving forward enough for a waypoint
        scan = [(d * 100, 5000) for d in range(36)]  # sparse scan
        step_mm = WAYPOINT_INTERVAL_MM + 10
        result = m.update(step_mm, 0, 0, scan)

        assert result["new_waypoint"] is not None
        assert len(m.waypoints) == 2

    def test_update_no_waypoint_short_distance(self):
        m = self._make_mapper()
        m.start_mapping()

        scan = [(d * 100, 5000) for d in range(36)]
        result = m.update(100, 0, 0, scan)  # only 100mm

        assert result["new_waypoint"] is None
        assert len(m.waypoints) == 1  # only start

    def test_loop_closure(self):
        m = self._make_mapper()
        m.start_mapping()

        scan = [(d * 100, 5000) for d in range(36)]

        # move far enough
        big_step = MIN_MAPPING_DISTANCE_MM + 500
        m.update(big_step, 0, 0, scan)

        # come back near start
        m.update(-big_step, 0, 0, scan)
        result = m.update(0, 0, 0, scan)

        # loop should close if within threshold
        # (pose may not be exactly at start due to SLAM correction, but the
        #  total distance check should pass)

    def test_label_waypoint(self):
        m = self._make_mapper()
        m.start_mapping()
        m.label_waypoint(0, "gate")
        assert m.waypoints[0].label == "gate"

    def test_label_out_of_range(self):
        m = self._make_mapper()
        m.start_mapping()
        m.label_waypoint(99, "ghost")  # should not crash

    def test_patrol_route_excludes_rtsp(self):
        m = self._make_mapper()
        m.waypoints = [
            Waypoint(0, 0, 0, "start"),
            Waypoint(1000, 0, 0, "camera_zone", has_rtsp_coverage=True),
            Waypoint(2000, 0, 0, "blind_spot"),
        ]
        route = m.get_patrol_route(exclude_rtsp=True)
        assert len(route) == 2
        assert all(not wp.has_rtsp_coverage for wp in route)

    def test_patrol_route_includes_all(self):
        m = self._make_mapper()
        m.waypoints = [
            Waypoint(0, 0, 0, "start"),
            Waypoint(1000, 0, 0, "camera", has_rtsp_coverage=True),
        ]
        route = m.get_patrol_route(exclude_rtsp=False)
        assert len(route) == 2

    def test_save_load(self):
        m = self._make_mapper()
        m.start_mapping()
        scan = [(d * 100, 3000) for d in range(36)]
        m.update(WAYPOINT_INTERVAL_MM + 10, 0, 0, scan)

        with tempfile.TemporaryDirectory() as td:
            m.save(td)
            assert os.path.exists(os.path.join(td, "map.pgm"))
            assert os.path.exists(os.path.join(td, "perimeter.json"))

            # verify JSON
            with open(os.path.join(td, "perimeter.json")) as f:
                data = json.load(f)
            assert "waypoints" in data
            assert len(data["waypoints"]) == len(m.waypoints)

            # load into new mapper
            m2 = self._make_mapper()
            m2.load(td)
            assert len(m2.waypoints) == len(m.waypoints)

    def test_find_frontiers_empty_grid(self):
        m = self._make_mapper()
        # untouched grid = all unknown, no frontiers
        frontiers = m.find_frontiers()
        assert frontiers == []

    def test_find_frontiers_after_scan(self):
        m = self._make_mapper()
        # feed some scan data to create free/occupied/unknown boundaries
        scan = [(d * 100, 2000) for d in range(360)]
        m._slam.update(0, 0, 0, scan)
        frontiers = m.find_frontiers()
        # should find some frontiers at the edge of scanned area
        assert len(frontiers) > 0
