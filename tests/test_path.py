# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/path.py — A* path planner."""

import pytest

from perception.slam import OccupancyGrid, LOG_ODDS_MAX, LOG_ODDS_MIN
from planner.path import PathPlanner, _octile_heuristic, PATH_PLAN_MARGIN_CELLS


class TestOctileHeuristic:

    def test_same_point(self):
        assert _octile_heuristic((5, 5), (5, 5)) == 0

    def test_straight(self):
        h = _octile_heuristic((0, 0), (3, 0))
        assert h == 30000  # 3 * STRAIGHT_COST

    def test_diagonal(self):
        h = _octile_heuristic((0, 0), (2, 2))
        assert h == 2 * 14142  # 2 * DIAGONAL_COST

    def test_mixed(self):
        h = _octile_heuristic((0, 0), (3, 1))
        # 1 diagonal + 2 straight
        assert h == 14142 + 20000


class TestPathPlanner:

    def _make_grid(self, size=50, resolution=100):
        g = OccupancyGrid(resolution_mm=resolution, size_cells=size)
        return g

    def _set_occupied(self, grid, x_mm, y_mm):
        cell = grid.world_to_cell(x_mm, y_mm)
        if cell:
            grid.cells[cell[1] * grid.size_cells + cell[0]] = LOG_ODDS_MAX

    def _set_free(self, grid, x_mm, y_mm):
        cell = grid.world_to_cell(x_mm, y_mm)
        if cell:
            grid.cells[cell[1] * grid.size_cells + cell[0]] = LOG_ODDS_MIN

    def test_straight_path(self):
        g = self._make_grid()
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 500, 0)
        assert len(path) >= 2
        # start and end should be close to requested
        assert abs(path[0][0]) <= 100
        assert abs(path[-1][0] - 500) <= 100

    def test_no_path_blocked_goal(self):
        g = self._make_grid()
        # block the goal area
        for dx in range(-300, 400, 100):
            for dy in range(-300, 400, 100):
                self._set_occupied(g, 500 + dx, 0 + dy)
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 500, 0)
        assert path == []

    def test_no_path_blocked_start(self):
        g = self._make_grid()
        self._set_occupied(g, 0, 0)
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 500, 0)
        assert path == []

    def test_path_around_obstacle(self):
        g = self._make_grid(size=50, resolution=100)
        # wall blocking direct path at x=300
        for y in range(-500, 600, 100):
            self._set_occupied(g, 300, y)
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 600, 0)
        # should find path going around
        if path:
            assert len(path) > 2
            # should reach goal area
            assert abs(path[-1][0] - 600) <= 100

    def test_out_of_bounds_returns_empty(self):
        g = self._make_grid(size=10, resolution=100)
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 99999, 99999)
        assert path == []

    def test_path_world_coords(self):
        g = self._make_grid()
        planner = PathPlanner(g)
        path = planner.plan(0, 0, 300, 300)
        # all waypoints should be tuples of floats
        for wp in path:
            assert len(wp) == 2
            assert isinstance(wp[0], (int, float))
