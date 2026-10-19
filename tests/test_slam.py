# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception/slam.py — occupancy grid, scan matching, SLAM."""

import math
import os
import tempfile
import pytest

from perception.slam import (
    OccupancyGrid,
    ScanMatcher,
    SLAM,
    _bresenham,
    SLAM_MAP_RESOLUTION_MM,
    SLAM_MAP_SIZE_CELLS,
    LOG_ODDS_OCCUPIED,
    LOG_ODDS_FREE,
    LOG_ODDS_PRIOR,
    OCCUPIED_THRESHOLD,
    FREE_THRESHOLD,
    SCAN_MAX_RANGE_MM,
    SCAN_MIN_RANGE_MM,
    CDEG_TO_RAD,
)

# ── OccupancyGrid ───────────────────────────────────────────────────────────


class TestOccupancyGrid:

    def test_defaults(self):
        g = OccupancyGrid()
        assert g.resolution_mm == SLAM_MAP_RESOLUTION_MM
        assert g.size_cells == SLAM_MAP_SIZE_CELLS
        assert len(g.cells) == SLAM_MAP_SIZE_CELLS**2

    def test_small_grid(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        assert len(g.cells) == 100

    def test_world_to_cell_origin(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        cell = g.world_to_cell(0, 0)
        assert cell == (5, 5)  # center of 10×10 grid

    def test_world_to_cell_offset(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        cell = g.world_to_cell(200, 100)
        assert cell == (7, 6)

    def test_world_to_cell_out_of_bounds(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        assert g.world_to_cell(99999, 0) is None

    def test_cell_to_world_roundtrip(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        cell = g.world_to_cell(200, 100)
        wx, wy = g.cell_to_world(*cell)
        assert wx == 200
        assert wy == 100

    def test_initial_state_unknown(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=10)
        assert not g.is_occupied(5, 5)
        assert not g.is_free(5, 5)
        assert g.is_unknown(5, 5)

    def test_update_from_scan_marks_occupied(self):
        g = OccupancyGrid(resolution_mm=50, size_cells=100)
        # Single scan point at 1000mm directly ahead (0 cdeg)
        g.update_from_scan(0, 0, 0, [(0, 1000)])
        # endpoint cell should be occupied
        end_cell = g.world_to_cell(1000, 0)
        assert end_cell is not None
        assert g.is_occupied(*end_cell)

    def test_update_from_scan_marks_free(self):
        g = OccupancyGrid(resolution_mm=50, size_cells=100)
        # run several scans to accumulate enough log-odds
        num_scans = 3
        for _ in range(num_scans):
            g.update_from_scan(0, 0, 0, [(0, 2000)])
        # cell well before endpoint should be free
        mid_cell = g.world_to_cell(500, 0)
        assert mid_cell is not None
        assert g.is_free(*mid_cell)

    def test_update_ignores_out_of_range(self):
        g = OccupancyGrid(resolution_mm=50, size_cells=10)
        # too close
        g.update_from_scan(0, 0, 0, [(0, 50)])
        # too far
        g.update_from_scan(0, 0, 0, [(0, SCAN_MAX_RANGE_MM + 100)])
        # all cells should remain unknown
        assert all(c == LOG_ODDS_PRIOR for c in g.cells)

    def test_save_load_pgm(self):
        g = OccupancyGrid(resolution_mm=100, size_cells=20)
        g.update_from_scan(0, 0, 0, [(0, 500)])

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.pgm")
            g.save_pgm(path)
            assert os.path.exists(path)

            g2 = OccupancyGrid(resolution_mm=100, size_cells=20)
            g2.load_pgm(path)
            # occupied cells should remain occupied
            end_cell = g.world_to_cell(500, 0)
            if end_cell:
                assert g2.is_occupied(*end_cell)


# ── Bresenham ────────────────────────────────────────────────────────────────


class TestBresenham:

    def test_horizontal(self):
        cells = _bresenham(0, 0, 5, 0)
        assert len(cells) == 6
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)

    def test_vertical(self):
        cells = _bresenham(0, 0, 0, 3)
        assert len(cells) == 4
        assert cells[-1] == (0, 3)

    def test_diagonal(self):
        cells = _bresenham(0, 0, 3, 3)
        assert cells[0] == (0, 0)
        assert cells[-1] == (3, 3)

    def test_single_point(self):
        cells = _bresenham(5, 5, 5, 5)
        assert cells == [(5, 5)]


# ── ScanMatcher ──────────────────────────────────────────────────────────────


class TestScanMatcher:

    def test_identity(self):
        m = ScanMatcher()
        pts = [(100, 0), (0, 100), (-100, 0), (0, -100)]
        dx, dy, dth = m.match(pts, pts)
        assert abs(dx) < 5
        assert abs(dy) < 5
        assert abs(dth) < 50  # centidegrees

    def test_small_translation(self):
        m = ScanMatcher()
        prev = [(1000, 0), (0, 1000), (-1000, 0), (0, -1000)]
        shift = 50  # mm
        curr = [(p[0] - shift, p[1]) for p in prev]
        dx, dy, dth = m.match(prev, curr)
        # should detect ~50mm shift in x
        assert abs(dx - shift) < 20

    def test_empty_scans_return_initial_guess(self):
        m = ScanMatcher()
        result = m.match([], [], (10, 20, 30))
        assert result == (10, 20, 30)


# ── SLAM ─────────────────────────────────────────────────────────────────────


class TestSLAM:

    def test_initial_pose(self):
        s = SLAM()
        assert s.get_pose() == (0.0, 0.0, 0.0)

    def test_update_returns_pose(self):
        s = SLAM(OccupancyGrid(resolution_mm=50, size_cells=100))
        scan = [(d * 100, 2000) for d in range(360)]
        pose = s.update(100, 0, 0, scan)
        assert len(pose) == 3
        # should have moved roughly 100mm forward
        assert pose[0] > 50

    def test_polar_to_cartesian(self):
        pts = SLAM.polar_to_cartesian([(0, 1000), (9000, 1000)])
        assert len(pts) == 2
        # 0 cdeg = forward (positive x)
        assert abs(pts[0][0] - 1000) < 1
        assert abs(pts[0][1]) < 1
        # 9000 cdeg = 90° = positive y
        assert abs(pts[1][0]) < 1
        assert abs(pts[1][1] - 1000) < 1

    def test_polar_to_cartesian_filters_range(self):
        pts = SLAM.polar_to_cartesian([(0, 50), (0, SCAN_MAX_RANGE_MM + 1)])
        assert len(pts) == 0

    def test_map_updated_after_scan(self):
        g = OccupancyGrid(resolution_mm=50, size_cells=100)
        s = SLAM(g)
        scan = [(0, 1000)]  # one point ahead
        s.update(0, 0, 0, scan)
        end_cell = g.world_to_cell(1000, 0)
        assert end_cell is not None
        assert g.is_occupied(*end_cell)
