# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""A* path planner on a 2D occupancy grid."""

import heapq
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATH_PLAN_MARGIN_CELLS = 2  # inflate obstacles by this many cells
DIAGONAL_COST = 14142  # sqrt(2) * 10000, integer math
STRAIGHT_COST = 10000
MAX_PLAN_ITERATIONS = 50_000  # prevent runaway on large maps

# 8-connected neighbor offsets: (dc, dr, cost)
_NEIGHBORS = [
    (-1, -1, DIAGONAL_COST),
    (0, -1, STRAIGHT_COST),
    (1, -1, DIAGONAL_COST),
    (-1, 0, STRAIGHT_COST),
    (1, 0, STRAIGHT_COST),
    (-1, 1, DIAGONAL_COST),
    (0, 1, STRAIGHT_COST),
    (1, 1, DIAGONAL_COST),
]


def _octile_heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Octile distance heuristic for 8-connected grid (integer)."""
    dc = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    return STRAIGHT_COST * max(dc, dr) + (DIAGONAL_COST - STRAIGHT_COST) * min(dc, dr)


class PathPlanner:
    """A* path planner over an OccupancyGrid."""

    def __init__(self, occupancy_grid):
        self._grid = occupancy_grid

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def plan(
        self,
        start_x_mm: float,
        start_y_mm: float,
        goal_x_mm: float,
        goal_y_mm: float,
    ) -> list[tuple[float, float]]:
        """Plan a path from start to goal (world mm). Returns waypoints or []."""
        start_cell = self._grid.world_to_cell(start_x_mm, start_y_mm)
        goal_cell = self._grid.world_to_cell(goal_x_mm, goal_y_mm)

        if start_cell is None or goal_cell is None:
            return []

        blocked = self._inflate_obstacles()

        if start_cell in blocked or goal_cell in blocked:
            return []

        path_cells = self._a_star(start_cell, goal_cell, blocked)
        if not path_cells:
            return []

        smoothed = self._smooth_path(path_cells, blocked)
        return self._cells_to_world(smoothed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inflate_obstacles(self) -> set[tuple[int, int]]:
        """Return set of cells that are occupied or within margin."""
        blocked: set[tuple[int, int]] = set()
        size = self._grid.size_cells
        for r in range(size):
            for c in range(size):
                if self._grid.is_occupied(c, r):
                    for dr in range(-PATH_PLAN_MARGIN_CELLS, PATH_PLAN_MARGIN_CELLS + 1):
                        for dc in range(-PATH_PLAN_MARGIN_CELLS, PATH_PLAN_MARGIN_CELLS + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < size and 0 <= nc < size:
                                blocked.add((nc, nr))
        return blocked

    def _a_star(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Standard A* on 8-connected grid. Returns cell path or []."""
        size = self._grid.size_cells
        open_heap: list[tuple[int, int, tuple[int, int]]] = []  # (f, counter, cell)
        counter = 0
        heapq.heappush(open_heap, (_octile_heuristic(start, goal), counter, start))
        g_score: dict[tuple[int, int], int] = {start: 0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        iterations = 0

        while open_heap and iterations < MAX_PLAN_ITERATIONS:
            iterations += 1
            _f, _cnt, current = heapq.heappop(open_heap)

            if current == goal:
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cur_g = g_score[current]
            for dc, dr, cost in _NEIGHBORS:
                nc, nr = current[0] + dc, current[1] + dr
                neighbor = (nc, nr)
                if not (0 <= nc < size and 0 <= nr < size):
                    continue
                if neighbor in blocked:
                    continue
                tentative_g = cur_g + cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _octile_heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))
                    came_from[neighbor] = current

        return []

    def _smooth_path(
        self,
        path: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Remove collinear intermediate points."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev = smoothed[-1]
            nxt = path[i + 1]
            # keep point if direction changes
            d1 = (path[i][0] - prev[0], path[i][1] - prev[1])
            d2 = (nxt[0] - path[i][0], nxt[1] - path[i][1])
            if d1 != d2:
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def _cells_to_world(
        self,
        path_cells: list[tuple[int, int]],
    ) -> list[tuple[float, float]]:
        """Convert cell path to world coordinates (mm)."""
        return [self._grid.cell_to_world(c, r) for c, r in path_cells]
