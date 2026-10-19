# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""2D LiDAR SLAM — occupancy grid mapping + scan matching."""

import math
import struct
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLAM_MAP_RESOLUTION_MM = 50  # 5 cm per cell
SLAM_MAP_SIZE_CELLS = 1000  # 50 m × 50 m max

# Log-odds for Bayesian occupancy update
LOG_ODDS_FREE = -0.4
LOG_ODDS_OCCUPIED = 0.85
LOG_ODDS_PRIOR = 0.0
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0
OCCUPIED_THRESHOLD = 0.5
FREE_THRESHOLD = -0.5

# Scan matcher (ICP)
ICP_MAX_ITERATIONS = 20
ICP_CONVERGENCE_THRESHOLD_MM = 1.0
ICP_MAX_CORRESPONDENCE_MM = 500

# Scan limits
SCAN_MAX_RANGE_MM = 12000  # LD19 max range
SCAN_MIN_RANGE_MM = 100  # ignore very close returns

CDEG_TO_RAD = math.pi / 18000.0  # centidegrees → radians


# ---------------------------------------------------------------------------
# Occupancy Grid
# ---------------------------------------------------------------------------


class OccupancyGrid:
    """2D log-odds occupancy grid."""

    def __init__(
        self,
        resolution_mm: int = SLAM_MAP_RESOLUTION_MM,
        size_cells: int = SLAM_MAP_SIZE_CELLS,
    ):
        self.resolution_mm = resolution_mm
        self.size_cells = size_cells
        self._origin = size_cells // 2  # grid center = world (0, 0)
        self.cells: list[float] = [LOG_ODDS_PRIOR] * (size_cells * size_cells)

    # -- coordinate transforms ------------------------------------------------

    def world_to_cell(self, x_mm: float, y_mm: float):
        """World mm → (col, row) or None if out of bounds."""
        c = int(round(x_mm / self.resolution_mm)) + self._origin
        r = int(round(y_mm / self.resolution_mm)) + self._origin
        if 0 <= c < self.size_cells and 0 <= r < self.size_cells:
            return (c, r)
        return None

    def cell_to_world(self, col: int, row: int) -> tuple[float, float]:
        """(col, row) → world mm (cell center)."""
        x_mm = (col - self._origin) * self.resolution_mm
        y_mm = (row - self._origin) * self.resolution_mm
        return (x_mm, y_mm)

    # -- query ----------------------------------------------------------------

    def _idx(self, col: int, row: int) -> int:
        return row * self.size_cells + col

    def is_occupied(self, col: int, row: int) -> bool:
        if not (0 <= col < self.size_cells and 0 <= row < self.size_cells):
            return False
        return self.cells[self._idx(col, row)] >= OCCUPIED_THRESHOLD

    def is_free(self, col: int, row: int) -> bool:
        if not (0 <= col < self.size_cells and 0 <= row < self.size_cells):
            return False
        return self.cells[self._idx(col, row)] <= FREE_THRESHOLD

    def is_unknown(self, col: int, row: int) -> bool:
        if not (0 <= col < self.size_cells and 0 <= row < self.size_cells):
            return True
        v = self.cells[self._idx(col, row)]
        return FREE_THRESHOLD < v < OCCUPIED_THRESHOLD

    # -- update ---------------------------------------------------------------

    def update_from_scan(
        self,
        robot_x_mm: float,
        robot_y_mm: float,
        heading_cdeg: float,
        scan_points: list[tuple[int, int]],
    ):
        """Update grid from a LiDAR scan.

        scan_points: list of (angle_cdeg, distance_mm) relative to robot.
        """
        origin = self.world_to_cell(robot_x_mm, robot_y_mm)
        if origin is None:
            return

        heading_rad = heading_cdeg * CDEG_TO_RAD

        for angle_cdeg, dist_mm in scan_points:
            if dist_mm < SCAN_MIN_RANGE_MM or dist_mm > SCAN_MAX_RANGE_MM:
                continue

            abs_angle = heading_rad + angle_cdeg * CDEG_TO_RAD
            end_x = robot_x_mm + dist_mm * math.cos(abs_angle)
            end_y = robot_y_mm + dist_mm * math.sin(abs_angle)

            end_cell = self.world_to_cell(end_x, end_y)
            if end_cell is None:
                continue

            # trace ray — mark free
            for c, r in _bresenham(origin[0], origin[1], end_cell[0], end_cell[1]):
                if (c, r) == end_cell:
                    break
                if 0 <= c < self.size_cells and 0 <= r < self.size_cells:
                    idx = self._idx(c, r)
                    self.cells[idx] = max(LOG_ODDS_MIN, self.cells[idx] + LOG_ODDS_FREE)

            # mark endpoint occupied
            idx = self._idx(end_cell[0], end_cell[1])
            self.cells[idx] = min(LOG_ODDS_MAX, self.cells[idx] + LOG_ODDS_OCCUPIED)

    # -- persistence ----------------------------------------------------------

    def save_pgm(self, path: str):
        """Save as PGM P5 binary. occupied=0, free=255, unknown=128."""
        n = self.size_cells
        with open(path, "wb") as f:
            header = f"P5\n{n} {n}\n255\n".encode()
            f.write(header)
            for i in range(n * n):
                v = self.cells[i]
                if v >= OCCUPIED_THRESHOLD:
                    px = 0
                elif v <= FREE_THRESHOLD:
                    px = 255
                else:
                    px = 128
                f.write(struct.pack("B", px))

    def load_pgm(self, path: str):
        """Load from PGM P5 binary."""
        with open(path, "rb") as f:
            magic = f.readline().strip()
            assert magic == b"P5", f"Expected P5, got {magic}"
            # skip comments
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            w, h = map(int, line.split())
            maxval = int(f.readline().strip())
            assert w == h == self.size_cells, f"Size mismatch: {w}x{h} vs {self.size_cells}"
            data = f.read(w * h)
            for i, px in enumerate(data):
                if px < 64:
                    self.cells[i] = LOG_ODDS_MAX  # occupied
                elif px > 192:
                    self.cells[i] = LOG_ODDS_MIN  # free
                else:
                    self.cells[i] = LOG_ODDS_PRIOR  # unknown


# ---------------------------------------------------------------------------
# Scan Matcher (simple ICP)
# ---------------------------------------------------------------------------


class ScanMatcher:
    """Point-to-point ICP for 2D scan matching."""

    def match(
        self,
        prev_points: list[tuple[float, float]],
        curr_points: list[tuple[float, float]],
        initial_guess: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float, float]:
        """Match curr_points to prev_points.

        Returns (dx_mm, dy_mm, dtheta_cdeg) correction.
        """
        if len(prev_points) < 2 or len(curr_points) < 2:
            return initial_guess

        dx, dy, dtheta_cdeg = initial_guess
        dtheta_rad = dtheta_cdeg * CDEG_TO_RAD

        for _iteration in range(ICP_MAX_ITERATIONS):
            # transform curr_points by current estimate
            cos_t = math.cos(dtheta_rad)
            sin_t = math.sin(dtheta_rad)
            transformed = [
                (cos_t * px - sin_t * py + dx, sin_t * px + cos_t * py + dy)
                for px, py in curr_points
            ]

            # find correspondences (nearest neighbor)
            pairs = []
            for tx, ty in transformed:
                best_dist = ICP_MAX_CORRESPONDENCE_MM * ICP_MAX_CORRESPONDENCE_MM
                best_pt = None
                for px, py in prev_points:
                    d2 = (tx - px) ** 2 + (ty - py) ** 2
                    if d2 < best_dist:
                        best_dist = d2
                        best_pt = (px, py)
                if best_pt is not None:
                    pairs.append(((tx, ty), best_pt))

            if len(pairs) < 2:
                break

            # compute correction from correspondences
            sum_dx = sum(p[1][0] - p[0][0] for p in pairs)
            sum_dy = sum(p[1][1] - p[0][1] for p in pairs)
            n = len(pairs)
            corr_dx = sum_dx / n
            corr_dy = sum_dy / n

            # estimate rotation correction from cross-product of centroids
            cx_t = sum(p[0][0] for p in pairs) / n
            cy_t = sum(p[0][1] for p in pairs) / n
            cx_p = sum(p[1][0] for p in pairs) / n
            cy_p = sum(p[1][1] for p in pairs) / n

            num = sum(
                (p[0][0] - cx_t) * (p[1][1] - cy_p) - (p[0][1] - cy_t) * (p[1][0] - cx_p)
                for p in pairs
            )
            den = sum(
                (p[0][0] - cx_t) * (p[1][0] - cx_p) + (p[0][1] - cy_t) * (p[1][1] - cy_p)
                for p in pairs
            )
            corr_theta = math.atan2(num, den) if den != 0 else 0.0

            dx += corr_dx
            dy += corr_dy
            dtheta_rad += corr_theta

            if (
                abs(corr_dx) < ICP_CONVERGENCE_THRESHOLD_MM
                and abs(corr_dy) < ICP_CONVERGENCE_THRESHOLD_MM
                and abs(corr_theta) < ICP_CONVERGENCE_THRESHOLD_MM * CDEG_TO_RAD
            ):
                break

        return (dx, dy, dtheta_rad / CDEG_TO_RAD)


# ---------------------------------------------------------------------------
# SLAM (main interface)
# ---------------------------------------------------------------------------


class SLAM:
    """2D LiDAR SLAM: odometry + scan matching + occupancy grid."""

    def __init__(self, grid: OccupancyGrid | None = None):
        self.grid = grid or OccupancyGrid()
        self._matcher = ScanMatcher()
        self._x_mm: float = 0.0
        self._y_mm: float = 0.0
        self._heading_cdeg: float = 0.0
        self._prev_cartesian: list[tuple[float, float]] = []

    def get_pose(self) -> tuple[float, float, float]:
        return (self._x_mm, self._y_mm, self._heading_cdeg)

    def get_map(self) -> OccupancyGrid:
        return self.grid

    def update(
        self,
        odom_dx_mm: float,
        odom_dy_mm: float,
        odom_dtheta_cdeg: float,
        scan_points: list[tuple[int, int]],
    ) -> tuple[float, float, float]:
        """Process one SLAM step. Returns corrected (x_mm, y_mm, heading_cdeg)."""
        # predict from odometry
        heading_rad = self._heading_cdeg * CDEG_TO_RAD
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        pred_x = self._x_mm + odom_dx_mm * cos_h - odom_dy_mm * sin_h
        pred_y = self._y_mm + odom_dx_mm * sin_h + odom_dy_mm * cos_h
        pred_heading = self._heading_cdeg + odom_dtheta_cdeg

        # convert scan to local cartesian
        curr_cartesian = self.polar_to_cartesian(scan_points)

        # scan match against previous scan
        if self._prev_cartesian:
            dx, dy, dtheta = self._matcher.match(
                self._prev_cartesian,
                curr_cartesian,
                initial_guess=(0.0, 0.0, 0.0),
            )
            pred_x += dx
            pred_y += dy
            pred_heading += dtheta

        # update pose
        self._x_mm = pred_x
        self._y_mm = pred_y
        self._heading_cdeg = pred_heading

        # update map
        self.grid.update_from_scan(self._x_mm, self._y_mm, self._heading_cdeg, scan_points)

        # store for next match
        self._prev_cartesian = curr_cartesian

        return self.get_pose()

    @staticmethod
    def polar_to_cartesian(
        scan_points: list[tuple[int, int]],
    ) -> list[tuple[float, float]]:
        """Convert (angle_cdeg, distance_mm) to robot-local (x_mm, y_mm)."""
        result = []
        for angle_cdeg, dist_mm in scan_points:
            if dist_mm < SCAN_MIN_RANGE_MM or dist_mm > SCAN_MAX_RANGE_MM:
                continue
            rad = angle_cdeg * CDEG_TO_RAD
            result.append((dist_mm * math.cos(rad), dist_mm * math.sin(rad)))
        return result


# ---------------------------------------------------------------------------
# Bresenham line algorithm
# ---------------------------------------------------------------------------


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line from (x0,y0) to (x1,y1)."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells
