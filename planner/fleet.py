# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Fleet management — multi-robot zone assignment and dispatch.

Orchestrates multiple robots: assigns patrol zones, dispatches nearest
robot to investigate threats, handles failover when robots go offline.

Single-robot setups work fine — FleetPlanner with 1 robot is a no-op wrapper.

Usage:
    fleet = FleetPlanner(config)
    fleet.register("robot_1", port=9000)
    fleet.assign_zone("robot_1", "zona_norte")
    nearest = fleet.dispatch_nearest(target_x, target_y)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("brain.fleet")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROBOT_HEARTBEAT_TIMEOUT_S = 30.0  # consider robot offline after no heartbeat
FLEET_REBALANCE_INTERVAL_S = 60.0  # recheck zone coverage periodically
DISPATCH_COOLDOWN_S = 10.0  # min time between dispatches to same zone


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RobotEntry:
    """Tracked state of one robot in the fleet."""

    robot_id: str
    port: int = 9000
    zones: list[str] = field(default_factory=list)
    dock_id: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    battery_mv: int = 0
    online: bool = False
    docked: bool = False
    busy: bool = False  # currently executing a dispatch
    last_heartbeat: float = 0.0
    last_dispatch: float = 0.0


@dataclass
class DispatchResult:
    """Result of dispatching a robot."""

    robot_id: str
    zone: str
    distance_mm: float
    success: bool = True


# ---------------------------------------------------------------------------
# FleetPlanner
# ---------------------------------------------------------------------------


class FleetPlanner:
    """Multi-robot fleet coordinator."""

    def __init__(self, config: dict | None = None):
        self._robots: dict[str, RobotEntry] = {}
        self._last_rebalance: float = 0.0

        if config:
            self._load_from_config(config)

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        robot_id: str,
        port: int = 9000,
        zones: list[str] | None = None,
        dock_id: str = "",
    ) -> RobotEntry:
        """Register a robot in the fleet."""
        entry = RobotEntry(
            robot_id=robot_id,
            port=port,
            zones=zones or [],
            dock_id=dock_id,
        )
        self._robots[robot_id] = entry
        logger.info("[Fleet] Registered robot '%s' on port %d", robot_id, port)
        return entry

    def unregister(self, robot_id: str):
        """Remove a robot from the fleet."""
        if robot_id in self._robots:
            del self._robots[robot_id]
            logger.info("[Fleet] Unregistered robot '%s'", robot_id)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def heartbeat(
        self,
        robot_id: str,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        battery_mv: int = 0,
        docked: bool = False,
    ):
        """Update robot state from heartbeat."""
        robot = self._robots.get(robot_id)
        if not robot:
            return
        robot.online = True
        robot.x_mm = x_mm
        robot.y_mm = y_mm
        robot.battery_mv = battery_mv
        robot.docked = docked
        robot.last_heartbeat = time.monotonic()

    def check_timeouts(self) -> list[str]:
        """Check for robots that missed heartbeat. Returns list of newly offline IDs."""
        now = time.monotonic()
        newly_offline = []
        for robot in self._robots.values():
            if robot.online and robot.last_heartbeat > 0:
                if now - robot.last_heartbeat > ROBOT_HEARTBEAT_TIMEOUT_S:
                    robot.online = False
                    newly_offline.append(robot.robot_id)
                    logger.warning(
                        "[Fleet] Robot '%s' went offline",
                        robot.robot_id,
                    )
        return newly_offline

    # ── Zone assignment ───────────────────────────────────────────────────

    def assign_zone(self, robot_id: str, zone: str):
        """Assign a zone to a robot."""
        robot = self._robots.get(robot_id)
        if robot and zone not in robot.zones:
            robot.zones.append(zone)

    def unassign_zone(self, robot_id: str, zone: str):
        """Remove a zone from a robot."""
        robot = self._robots.get(robot_id)
        if robot and zone in robot.zones:
            robot.zones.remove(zone)

    def get_zone_owner(self, zone: str) -> str | None:
        """Return robot_id that owns a zone, or None."""
        for robot in self._robots.values():
            if zone in robot.zones:
                return robot.robot_id
        return None

    def uncovered_zones(self, all_zones: list[str]) -> list[str]:
        """Return zones not assigned to any online robot."""
        covered = set()
        for robot in self._robots.values():
            if robot.online and not robot.docked:
                covered.update(robot.zones)
        return [z for z in all_zones if z not in covered]

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch_nearest(
        self,
        target_x_mm: float,
        target_y_mm: float,
        zone: str = "",
    ) -> DispatchResult | None:
        """Find the nearest available robot and dispatch it.

        Returns DispatchResult or None if no robot available.
        """
        now = time.monotonic()
        best: RobotEntry | None = None
        best_dist = float("inf")

        for robot in self._robots.values():
            if not robot.online or robot.docked or robot.busy:
                continue
            # Respect dispatch cooldown
            if now - robot.last_dispatch < DISPATCH_COOLDOWN_S:
                continue
            dist = math.hypot(
                target_x_mm - robot.x_mm,
                target_y_mm - robot.y_mm,
            )
            if dist < best_dist:
                best_dist = dist
                best = robot

        if best is None:
            logger.warning("[Fleet] No robot available for dispatch")
            return None

        best.busy = True
        best.last_dispatch = now
        logger.info(
            "[Fleet] Dispatching '%s' to (%.0f, %.0f) zone='%s' dist=%.0fmm",
            best.robot_id,
            target_x_mm,
            target_y_mm,
            zone,
            best_dist,
        )
        return DispatchResult(
            robot_id=best.robot_id,
            zone=zone,
            distance_mm=best_dist,
        )

    def report_dispatch_complete(self, robot_id: str):
        """Mark robot as no longer busy."""
        robot = self._robots.get(robot_id)
        if robot:
            robot.busy = False

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def robot_count(self) -> int:
        return len(self._robots)

    @property
    def online_count(self) -> int:
        return sum(1 for r in self._robots.values() if r.online)

    def get_robot(self, robot_id: str) -> RobotEntry | None:
        return self._robots.get(robot_id)

    def all_robots(self) -> list[RobotEntry]:
        return list(self._robots.values())

    def online_robots(self) -> list[RobotEntry]:
        return [r for r in self._robots.values() if r.online]

    def fleet_summary(self) -> dict:
        """Summary for Telegram /fleet command."""
        robots = []
        for r in self._robots.values():
            robots.append(
                {
                    "id": r.robot_id,
                    "online": r.online,
                    "battery_mv": r.battery_mv,
                    "zones": r.zones,
                    "docked": r.docked,
                    "busy": r.busy,
                }
            )
        return {
            "total": len(self._robots),
            "online": self.online_count,
            "robots": robots,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_from_config(self, config: dict):
        """Load fleet config from config.yaml."""
        fleet_cfg = config.get("fleet", {})
        if not fleet_cfg.get("enabled", False):
            return
        for entry in fleet_cfg.get("robots", []):
            self.register(
                robot_id=entry.get("id", f"robot_{self.robot_count}"),
                port=entry.get("port", 9000),
                zones=entry.get("zones", []),
                dock_id=entry.get("dock", ""),
            )

    def __repr__(self) -> str:
        return f"FleetPlanner(robots={self.robot_count}, " f"online={self.online_count})"
