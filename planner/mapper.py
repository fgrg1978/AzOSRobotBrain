# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Mapping orchestrator — coordinates SLAM, path planning, and exploration."""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAYPOINT_INTERVAL_MM = 1000  # record waypoint every 1 m
LOOP_CLOSURE_THRESHOLD_MM = 500  # close loop when this close to start
MAP_SAVE_DIR = "data"
MAP_FILE = "map.pgm"
PERIMETER_FILE = "perimeter.json"
MIN_EXPLORATION_DISTANCE_MM = 300  # minimum distance to a frontier
FRONTIER_MIN_SIZE = 3  # minimum cells for a valid frontier cluster
MIN_MAPPING_DISTANCE_MM = 3000  # don't close loop until at least this far


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------


@dataclass
class Waypoint:
    x_mm: int
    y_mm: int
    heading_cdeg: int
    label: str = ""
    has_rtsp_coverage: bool = False
    zone_type: str = ""  # door, window, gate, open_area, etc.
    zone_priority: int = 0  # 0=normal, 1+=higher priority (scanned more often)
    last_state: str = ""  # last VLM description for change detection


# ---------------------------------------------------------------------------
# PerimeterMapper
# ---------------------------------------------------------------------------


class PerimeterMapper:
    """Orchestrates mapping: feeds SLAM, records waypoints, detects loop closure."""

    def __init__(self, slam, path_planner=None):
        self._slam = slam
        self._planner = path_planner
        self.waypoints: list[Waypoint] = []
        self.mapping_active: bool = False
        self._start_pose: tuple[float, float, float] | None = None
        self._total_distance_mm: float = 0.0
        self._since_last_wp_mm: float = 0.0

    def start_mapping(self):
        """Begin a new mapping run."""
        pose = self._slam.get_pose()
        self._start_pose = pose
        self.waypoints = [
            Waypoint(
                x_mm=int(pose[0]),
                y_mm=int(pose[1]),
                heading_cdeg=int(pose[2]),
                label="start",
            )
        ]
        self._total_distance_mm = 0.0
        self._since_last_wp_mm = 0.0
        self.mapping_active = True

    def update(
        self,
        odom_dx_mm: float,
        odom_dy_mm: float,
        odom_dtheta_cdeg: float,
        scan_points: list[tuple[int, int]],
    ) -> dict:
        """Feed odometry + scan to SLAM, track waypoints, check loop closure."""
        pose = self._slam.update(odom_dx_mm, odom_dy_mm, odom_dtheta_cdeg, scan_points)

        step_dist = math.hypot(odom_dx_mm, odom_dy_mm)
        self._total_distance_mm += step_dist
        self._since_last_wp_mm += step_dist

        new_wp = None
        loop_closed = False

        if self.mapping_active and self._since_last_wp_mm >= WAYPOINT_INTERVAL_MM:
            new_wp = Waypoint(
                x_mm=int(pose[0]),
                y_mm=int(pose[1]),
                heading_cdeg=int(pose[2]),
            )
            self.waypoints.append(new_wp)
            self._since_last_wp_mm = 0.0

        # check loop closure
        if (
            self.mapping_active
            and self._start_pose is not None
            and self._total_distance_mm > MIN_MAPPING_DISTANCE_MM
        ):
            dist_to_start = math.hypot(
                pose[0] - self._start_pose[0],
                pose[1] - self._start_pose[1],
            )
            if dist_to_start < LOOP_CLOSURE_THRESHOLD_MM:
                loop_closed = True
                self.mapping_active = False

        return {
            "pose": pose,
            "new_waypoint": new_wp,
            "loop_closed": loop_closed,
            "mapping_active": self.mapping_active,
        }

    def label_waypoint(self, index: int, label: str):
        """Set VLM label on a waypoint."""
        if 0 <= index < len(self.waypoints):
            self.waypoints[index].label = label

    def find_frontiers(self) -> list[tuple[float, float]]:
        """Find frontier cells (free adjacent to unknown), return cluster centroids."""
        grid = self._slam.get_map()
        frontier_cells: list[tuple[int, int]] = []

        for r in range(grid.size_cells):
            for c in range(grid.size_cells):
                if not grid.is_free(c, r):
                    continue
                # check 4-connected neighbors for unknown
                for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nc, nr = c + dc, r + dr
                    if grid.is_unknown(nc, nr):
                        frontier_cells.append((c, r))
                        break

        # cluster by flood fill (simple BFS)
        visited: set[tuple[int, int]] = set()
        frontier_set = set(frontier_cells)
        centroids: list[tuple[float, float]] = []

        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster = []
            stack = [cell]
            while stack:
                cur = stack.pop()
                if cur in visited or cur not in frontier_set:
                    continue
                visited.add(cur)
                cluster.append(cur)
                for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cur[0] + dc, cur[1] + dr))

            if len(cluster) >= FRONTIER_MIN_SIZE:
                avg_c = sum(p[0] for p in cluster) / len(cluster)
                avg_r = sum(p[1] for p in cluster) / len(cluster)
                wx, wy = grid.cell_to_world(int(avg_c), int(avg_r))
                centroids.append((wx, wy))

        return centroids

    def get_waypoint_by_label(self, label: str) -> Waypoint | None:
        """Find first waypoint matching label (case-insensitive)."""
        label_lower = label.lower()
        for wp in self.waypoints:
            if wp.label.lower() == label_lower:
                return wp
        return None

    def get_patrol_route(self, exclude_rtsp: bool = True) -> list[Waypoint]:
        """Return waypoints for patrol, optionally skipping RTSP-covered zones."""
        if exclude_rtsp:
            return [wp for wp in self.waypoints if not wp.has_rtsp_coverage]
        return list(self.waypoints)

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str = MAP_SAVE_DIR):
        """Save map.pgm + perimeter.json."""
        os.makedirs(directory, exist_ok=True)
        self._slam.get_map().save_pgm(os.path.join(directory, MAP_FILE))

        data = {
            "waypoints": [asdict(wp) for wp in self.waypoints],
            "total_distance_mm": int(self._total_distance_mm),
            "created": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(directory, PERIMETER_FILE), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, directory: str = MAP_SAVE_DIR):
        """Load map.pgm + perimeter.json."""
        map_path = os.path.join(directory, MAP_FILE)
        perim_path = os.path.join(directory, PERIMETER_FILE)

        if os.path.exists(map_path):
            self._slam.get_map().load_pgm(map_path)

        if os.path.exists(perim_path):
            with open(perim_path) as f:
                data = json.load(f)
            self.waypoints = [Waypoint(**wp) for wp in data.get("waypoints", [])]
            self._total_distance_mm = data.get("total_distance_mm", 0)
