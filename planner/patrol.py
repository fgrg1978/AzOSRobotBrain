# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Patrol controller — waypoint-based patrol with SLAM navigation and VLM scanning.

Orchestrates: SLAM localization → path planning → waypoint navigation → VLM scan.
Runs as an async loop, integrated with BrainServer.
"""

import asyncio
import enum
import logging
import math
import time
from typing import Callable, Awaitable, Optional

from perception.slam import SLAM, OccupancyGrid, CDEG_TO_RAD
from planner.path import PathPlanner
from planner.mapper import PerimeterMapper, Waypoint

logger = logging.getLogger("brain.patrol")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATROL_SCAN_DWELL_S = 3.0  # seconds to dwell at waypoint for VLM scan
PATROL_WAYPOINT_REACH_MM = 300  # close enough to consider waypoint reached
PATROL_HEADING_TOLERANCE_CDEG = 1500  # 15° tolerance for heading alignment
PATROL_NAV_STEP_HZ = 10  # navigation update rate
PATROL_DEFAULT_SPEED_PCT = 40  # default patrol motor speed
PATROL_TURN_SPEED_PCT = 25  # turn-in-place speed
PATROL_STEER_KP = 0.5  # proportional gain for heading correction
PATROL_MAX_STEER = 50  # max differential steer (-50..+50)


class PatrolState(enum.Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    SCANNING = "scanning"
    MAPPING = "mapping"
    DONE = "done"
    ERROR = "error"


class PatrolController:
    """Waypoint-based patrol with SLAM-aided navigation."""

    def __init__(
        self,
        slam: SLAM,
        path_planner: PathPlanner,
        mapper: PerimeterMapper,
        send_cmd: Callable,  # async (ActuatorCmd) -> None
        policy,  # WheeledPolicy
        on_waypoint_reached: Optional[Callable[[Waypoint, int], Awaitable[None]]] = None,
        on_detection: Optional[Callable[[str, bytes], Awaitable[None]]] = None,
        is_connected: Optional[Callable[[], bool]] = None,
    ):
        self._slam = slam
        self._planner = path_planner
        self._mapper = mapper
        self._send_cmd = send_cmd
        self._policy = policy
        self._on_waypoint_reached = on_waypoint_reached
        self._on_detection = on_detection
        # Optional liveness probe (BrainServer.state.connected). Without it
        # the navigation loop below has no way to notice the robot dropped
        # the TCP connection mid-nav and will otherwise spin forever calling
        # send_cmd() into the void. None keeps the class usable standalone
        # (e.g. in tests) — the check is simply skipped in that case.
        self._is_connected = is_connected

        self.state = PatrolState.IDLE
        self._stop_event = asyncio.Event()
        self._current_wp_idx = 0
        self._patrol_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        """Request patrol stop."""
        self._stop_event.set()

    async def run_patrol(self, loop: bool = True, exclude_rtsp: bool = True):
        """Run patrol loop over recorded waypoints.

        Args:
            loop: repeat patrol indefinitely
            exclude_rtsp: skip waypoints covered by RTSP cameras
        """
        route = self._mapper.get_patrol_route(exclude_rtsp=exclude_rtsp)
        if not route:
            logger.warning("[Patrol] No waypoints — run mapping first")
            self.state = PatrolState.ERROR
            return

        self.state = PatrolState.NAVIGATING
        self._stop_event.clear()
        self._current_wp_idx = 0
        self._patrol_count = 0

        logger.info("[Patrol] Starting patrol: %d waypoints, loop=%s", len(route), loop)

        while not self._stop_event.is_set():
            for i, wp in enumerate(route):
                if self._stop_event.is_set():
                    break

                self._current_wp_idx = i
                self.state = PatrolState.NAVIGATING

                # navigate to waypoint
                reached = await self._navigate_to(wp)
                if not reached:
                    continue  # skip if interrupted or unreachable

                # scan at waypoint
                self.state = PatrolState.SCANNING
                if self._on_waypoint_reached:
                    await self._on_waypoint_reached(wp, i)

                # dwell for VLM scan
                await self._interruptible_sleep(PATROL_SCAN_DWELL_S)

            self._patrol_count += 1
            logger.info("[Patrol] Lap %d complete", self._patrol_count)

            if not loop:
                break

        # stop motors
        cmd = self._policy.translate("STOP")
        await self._send_cmd(cmd)
        self.state = PatrolState.DONE
        logger.info("[Patrol] Stopped after %d laps", self._patrol_count)

    async def run_mapping(self):
        """Run autonomous mapping (frontier exploration)."""
        self.state = PatrolState.MAPPING
        self._stop_event.clear()
        self._mapper.start_mapping()

        logger.info("[Patrol] Mapping started")

        while not self._stop_event.is_set():
            # find frontiers
            frontiers = self._mapper.find_frontiers()
            if not frontiers:
                logger.info("[Patrol] No more frontiers — mapping complete")
                break

            # pick nearest frontier
            pose = self._slam.get_pose()
            nearest = min(
                frontiers,
                key=lambda f: math.hypot(f[0] - pose[0], f[1] - pose[1]),
            )

            # navigate to frontier
            wp = Waypoint(x_mm=int(nearest[0]), y_mm=int(nearest[1]), heading_cdeg=int(pose[2]))
            reached = await self._navigate_to(wp)
            if not reached:
                continue

            # do a 360 scan at frontier
            await self._rotate_360()

        # save map
        self._mapper.save()

        cmd = self._policy.translate("STOP")
        await self._send_cmd(cmd)
        self.state = PatrolState.DONE
        logger.info("[Patrol] Mapping complete, %d waypoints recorded", len(self._mapper.waypoints))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _should_continue(self) -> bool:
        """False once patrol should stop: interrupted, or (when wired) the
        robot's TCP connection dropped mid-navigation."""
        if self._stop_event.is_set():
            return False
        if self._is_connected is not None and not self._is_connected():
            return False
        return True

    async def _navigate_to(self, wp: Waypoint) -> bool:
        """Navigate to a waypoint using path planner. Returns True if reached.

        Always sends STOP on the way out (finally) — normal completion,
        early return (interrupted/disconnected/unreachable), or the caller
        cancelling us via `asyncio.wait_for(..., timeout=...)` must not
        leave the robot still executing the last NAVIGATE_PATH command.
        """
        try:
            pose = self._slam.get_pose()
            path = self._planner.plan(pose[0], pose[1], wp.x_mm, wp.y_mm)

            if not path:
                logger.warning("[Patrol] No path to (%d, %d)", wp.x_mm, wp.y_mm)
                return False

            interval = 1.0 / PATROL_NAV_STEP_HZ

            for target_x, target_y in path[1:]:  # skip start
                if not self._should_continue():
                    return False

                # drive toward target point
                while self._should_continue():
                    pose = self._slam.get_pose()
                    dx = target_x - pose[0]
                    dy = target_y - pose[1]
                    dist = math.hypot(dx, dy)

                    if dist < PATROL_WAYPOINT_REACH_MM:
                        break  # reached this path point

                    # desired heading
                    desired_cdeg = math.atan2(dy, dx) / CDEG_TO_RAD
                    heading_err = self._wrap_cdeg(desired_cdeg - pose[2])

                    # if heading error too large, turn in place first
                    if abs(heading_err) > PATROL_HEADING_TOLERANCE_CDEG:
                        steer = PATROL_TURN_SPEED_PCT if heading_err > 0 else -PATROL_TURN_SPEED_PCT
                        cmd = self._policy.translate(
                            "NAVIGATE_PATH",
                            {
                                "speed": 0,
                                "steer": steer,
                            },
                        )
                    else:
                        # proportional steering
                        steer = int(heading_err * PATROL_STEER_KP / 100)
                        steer = max(-PATROL_MAX_STEER, min(PATROL_MAX_STEER, steer))
                        cmd = self._policy.translate(
                            "NAVIGATE_PATH",
                            {
                                "speed": PATROL_DEFAULT_SPEED_PCT,
                                "steer": steer,
                            },
                        )

                    await self._send_cmd(cmd)
                    await asyncio.sleep(interval)
                else:
                    # Inner while exited via _should_continue() going False
                    # (interrupted/disconnected), not via the "reached" break.
                    return False

            return True
        finally:
            try:
                cmd = self._policy.translate("STOP")
                await self._send_cmd(cmd)
            except Exception:
                logger.exception("[Patrol] failed to send STOP in _navigate_to")

    async def _rotate_360(self):
        """Rotate 360° in place for scanning."""
        cmd = self._policy.translate("SCAN_360", {"speed": PATROL_TURN_SPEED_PCT})
        await self._send_cmd(cmd)
        # estimate time for 360° rotation
        scan_duration = 8.0  # seconds for full rotation at low speed
        await self._interruptible_sleep(scan_duration)
        # stop
        cmd = self._policy.translate("STOP")
        await self._send_cmd(cmd)

    # ------------------------------------------------------------------
    # Feed sensor data (called from server dispatch)
    # ------------------------------------------------------------------

    def feed_sensors(
        self,
        odom_dx_mm: float,
        odom_dy_mm: float,
        odom_dtheta_cdeg: float,
        scan_points: list[tuple[int, int]],
    ):
        """Feed odometry + LiDAR to SLAM (and mapper if mapping)."""
        self._slam.update(odom_dx_mm, odom_dy_mm, odom_dtheta_cdeg, scan_points)
        if self.state == PatrolState.MAPPING and self._mapper.mapping_active:
            self._mapper.update(odom_dx_mm, odom_dy_mm, odom_dtheta_cdeg, scan_points)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_cdeg(angle_cdeg: float) -> float:
        """Wrap angle to [-18000, 18000) centidegrees."""
        while angle_cdeg > 18000:
            angle_cdeg -= 36000
        while angle_cdeg <= -18000:
            angle_cdeg += 36000
        return angle_cdeg

    async def _interruptible_sleep(self, duration: float):
        poll = 0.05
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                break
            remaining = deadline - time.monotonic()
            await asyncio.sleep(min(poll, remaining))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def current_waypoint_index(self) -> int:
        return self._current_wp_idx

    @property
    def patrol_count(self) -> int:
        return self._patrol_count

    def __repr__(self) -> str:
        return (
            f"PatrolController(state={self.state.value}, "
            f"wp={self._current_wp_idx}, laps={self._patrol_count})"
        )
