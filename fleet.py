# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Fleet management — per-robot-id registry for multi-robot brain.

This is the connection-level fleet registry (E07). It is complementary to
`planner/fleet.py` (`FleetPlanner`) which handles zone assignment / nearest
dispatch. Here we track *connected* robots keyed by `robot_id`, forward
targeted or broadcast commands, and detect timeouts.

Typical usage:

    fleet = FleetManager()
    entry = fleet.register(robot_id="bot_1", robot_type=0, name="R1",
                            writer=writer)
    fleet.heartbeat("bot_1", battery_mv=7600)
    await fleet.send_targeted("bot_1", PKT_ACTUATOR, payload)
    await fleet.broadcast(PKT_ACTUATOR, payload)
    fleet.check_timeouts()            # mark silent robots offline
    status = fleet.get_fleet_status() # aggregated dict for REST API

`send_targeted` / `broadcast` accept an injectable `send_fn` so tests can
substitute a mock for the network layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Tuple, Union

import metrics as _metrics_mod

# ShardCoordinator is only needed for type annotations; import at runtime
# only when callers actually pass `shard=` so there is no hard dependency cycle.
if TYPE_CHECKING:
    from fleet_sharding import ShardCoordinator

logger = logging.getLogger("brain.fleet_mgr")


# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

# Consider robot offline after this many seconds without a heartbeat.
FLEET_HEARTBEAT_TIMEOUT_S: float = 30.0

# Meta keys that robots are NOT allowed to set via heartbeat or registration.
# Allowing robots to set ota_host/ota_port would let a rogue robot redirect
# the OTA socket to an attacker-controlled host (SSRF / supply-chain attack).
# The OTA endpoint is always derived from the brain's own peer address record.
_META_KEYS_FORBIDDEN: frozenset = frozenset({"ota_host", "ota_port"})

# Upper bound on registered robots to guard against accidental growth.
FLEET_MAX_ROBOTS: int = 64

# Default robot_type used when a connection does not declare one.
FLEET_DEFAULT_ROBOT_TYPE: int = 0  # wheeled

# Sentinels used when a field has never been populated.
FLEET_UNKNOWN_BATTERY_MV: int = 0
FLEET_UNSET_TIMESTAMP: float = 0.0

# Ports / roles defaults (only used by `register_from_config`).
FLEET_DEFAULT_ROBOT_PORT: int = 9000

# Safety cap for broadcast fan-out to avoid runaway parallelism.
FLEET_BROADCAST_MAX_CONCURRENCY: int = FLEET_MAX_ROBOTS


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RobotRecord:
    """Metadata + live state for one registered robot.

    `writer` is an `asyncio.StreamWriter` when the robot is connected via
    TCP; it may be None for offline / pre-registered robots. We type it as
    Any so unit tests can supply a mock.
    """

    robot_id: str
    robot_type: int = FLEET_DEFAULT_ROBOT_TYPE
    name: str = ""
    writer: Any = None
    online: bool = False
    registered_at: float = field(default_factory=time.time)
    last_seen: float = FLEET_UNSET_TIMESTAMP
    battery_mv: int = FLEET_UNKNOWN_BATTERY_MV
    location: tuple[float, float] = (0.0, 0.0)  # (x_mm, y_mm) or (lat, lon)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.robot_id,
            "type": self.robot_type,
            "name": self.name,
            "online": self.online,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "battery_mv": self.battery_mv,
            "location": list(self.location),
            "meta": self.meta,
        }


# Type alias for the send function. Signature mirrors
# `protocol.send_packet(writer, pkt_type, payload)`.
SendFn = Callable[[Any, int, bytes], Awaitable[None]]


# ---------------------------------------------------------------------------
# FleetManager
# ---------------------------------------------------------------------------


class FleetManager:
    """Registry of robots connected to this brain server.

    All public methods are safe to call from the main asyncio loop.
    """

    def __init__(
        self,
        send_fn: Optional[SendFn] = None,
        heartbeat_timeout_s: float = FLEET_HEARTBEAT_TIMEOUT_S,
        max_robots: int = FLEET_MAX_ROBOTS,
        shard: Optional["ShardCoordinator"] = None,
    ) -> None:
        self._robots: dict[str, RobotRecord] = {}
        self._send_fn = send_fn  # may be set later
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._max_robots = max_robots
        # Optional shard coordinator (S3).  When set, register() will reject
        # robots whose robot_id hashes to a different brain node and return a
        # (forwarded=True, target=<node_addr>) tuple instead of a RobotRecord.
        self._shard = shard

    @staticmethod
    def _sanitise_meta(meta: dict) -> dict:
        """Return a copy of `meta` with forbidden keys removed.

        Robots must not be able to override brain-side routing values such as
        ota_host / ota_port; if they could, a rogue robot could redirect OTA
        transfers to an attacker-controlled host (see _META_KEYS_FORBIDDEN).
        """
        forbidden = _META_KEYS_FORBIDDEN.intersection(meta)
        if forbidden:
            logger.warning(
                "[Fleet] Dropping forbidden meta keys from robot: %s",
                sorted(forbidden),
            )
        return {k: v for k, v in meta.items() if k not in _META_KEYS_FORBIDDEN}

    # ── Configuration ─────────────────────────────────────────────────────

    def set_send_fn(self, send_fn: SendFn) -> None:
        """Inject a send function (e.g. `protocol.send_packet`)."""
        self._send_fn = send_fn

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        robot_id: str,
        robot_type: int = FLEET_DEFAULT_ROBOT_TYPE,
        name: str = "",
        writer: Any = None,
        meta: Optional[dict] = None,
    ) -> Union[RobotRecord, Tuple[bool, str]]:
        """Register / update a robot. Raises RuntimeError on overflow.

        If the robot_id already exists, its writer/metadata are updated and
        `online` is set to True (useful for reconnects).

        Shard-aware mode (S3):
            When a ``ShardCoordinator`` was passed to ``__init__``, this method
            checks whether *robot_id* belongs to this brain instance.  If not,
            it returns the tuple ``(True, target_node_address)`` so the caller
            can forward the connection.  When shard is unset (default), the
            return type is always ``RobotRecord`` — behaviour is unchanged.
        """
        # ── Shard check (S3) ─────────────────────────────────────────────
        if self._shard is not None and not self._shard.is_local(robot_id):
            target = self._shard.forward_target(robot_id) or ""
            logger.info(
                "[Fleet] robot_id=%r not local — forward to %s",
                robot_id,
                target,
            )
            return (True, target)

        existing = self._robots.get(robot_id)
        if existing is not None:
            existing.robot_type = robot_type
            if name:
                existing.name = name
            if writer is not None:
                existing.writer = writer
            if meta:
                # Strip forbidden keys before merging robot-supplied metadata.
                existing.meta.update(self._sanitise_meta(meta))
            existing.online = True
            existing.last_seen = time.time()
            logger.info("[Fleet] Re-registered '%s' (type=%d)", robot_id, robot_type)
            return existing

        if len(self._robots) >= self._max_robots:
            raise RuntimeError(f"Fleet capacity exceeded " f"({self._max_robots} robots)")

        record = RobotRecord(
            robot_id=robot_id,
            robot_type=robot_type,
            name=name or robot_id,
            writer=writer,
            online=writer is not None,
            last_seen=time.time() if writer is not None else FLEET_UNSET_TIMESTAMP,
            # Strip forbidden keys on initial registration too.
            meta=self._sanitise_meta(meta) if meta else {},
        )
        self._robots[robot_id] = record
        _metrics_mod.M.fleet_size.set(float(len(self._robots)))
        if record.online:
            _metrics_mod.M.conn_active.inc()
            _metrics_mod.M.conn_accepted_total.inc()
        logger.info(
            "[Fleet] Registered '%s' (type=%d, total=%d)", robot_id, robot_type, len(self._robots)
        )
        return record

    def unregister(self, robot_id: str) -> bool:
        """Remove a robot. Returns True if it existed."""
        record = self._robots.pop(robot_id, None)
        if record is None:
            return False
        _metrics_mod.M.fleet_size.set(float(len(self._robots)))
        if record.online:
            _metrics_mod.M.conn_active.dec()
        logger.info("[Fleet] Unregistered '%s'", robot_id)
        return True

    def mark_disconnected(self, robot_id: str) -> None:
        """Mark a robot offline without removing it (graceful disconnect)."""
        record = self._robots.get(robot_id)
        if record:
            was_online = record.online
            record.online = False
            record.writer = None
            if was_online:
                _metrics_mod.M.conn_active.dec()
            logger.info("[Fleet] '%s' disconnected", robot_id)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def heartbeat(
        self,
        robot_id: str,
        battery_mv: Optional[int] = None,
        location: Optional[tuple[float, float]] = None,
        meta: Optional[dict] = None,
    ) -> bool:
        """Update last_seen + optional telemetry for a robot.

        Returns True if the robot is known, False if not registered.
        """
        record = self._robots.get(robot_id)
        if record is None:
            return False

        record.last_seen = time.time()
        record.online = True
        if battery_mv is not None:
            record.battery_mv = battery_mv
        if location is not None:
            record.location = location
        if meta:
            # Strip forbidden keys; robots must not redirect OTA endpoints.
            record.meta.update(self._sanitise_meta(meta))
        return True

    def check_timeouts(self) -> list[str]:
        """Mark robots offline if they missed heartbeat window.

        Returns list of robot_ids that transitioned from online->offline.
        """
        now = time.time()
        newly_offline: list[str] = []
        for record in self._robots.values():
            if not record.online:
                continue
            if record.last_seen == FLEET_UNSET_TIMESTAMP:
                continue
            if now - record.last_seen > self._heartbeat_timeout_s:
                record.online = False
                newly_offline.append(record.robot_id)
                logger.warning(
                    "[Fleet] '%s' timed out (%.1fs since last_seen)",
                    record.robot_id,
                    now - record.last_seen,
                )
        return newly_offline

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, robot_id: str) -> Optional[RobotRecord]:
        return self._robots.get(robot_id)

    def all_robots(self) -> list[RobotRecord]:
        return list(self._robots.values())

    def online_robots(self) -> list[RobotRecord]:
        return [r for r in self._robots.values() if r.online]

    @property
    def count(self) -> int:
        return len(self._robots)

    @property
    def online_count(self) -> int:
        return sum(1 for r in self._robots.values() if r.online)

    def get_fleet_status(self) -> dict:
        """Aggregated snapshot for the /fleet/robots REST endpoint."""
        return {
            "total": self.count,
            "online": self.online_count,
            "timeout_s": self._heartbeat_timeout_s,
            "robots": {
                rid: {
                    "online": r.online,
                    "type": r.robot_type,
                    "name": r.name,
                    "last_seen": r.last_seen,
                    "battery_mv": r.battery_mv,
                    "location": list(r.location),
                }
                for rid, r in self._robots.items()
            },
        }

    # ── Command dispatch ──────────────────────────────────────────────────

    async def send_targeted(
        self,
        robot_id: str,
        pkt_type: int,
        payload: bytes,
    ) -> bool:
        """Send a packet to a specific robot.

        Returns True on success, False if the robot isn't available.
        """
        record = self._robots.get(robot_id)
        if record is None or not record.online or record.writer is None:
            logger.warning("[Fleet] Cannot send to '%s' (not online)", robot_id)
            return False
        if self._send_fn is None:
            logger.error("[Fleet] send_fn not configured")
            return False
        try:
            await self._send_fn(record.writer, pkt_type, payload)
            return True
        except Exception as e:  # noqa: BLE001 - log and downgrade
            logger.error("[Fleet] send to '%s' failed: %s", robot_id, e)
            # Connection likely broken — mark offline.
            self.mark_disconnected(robot_id)
            return False

    async def broadcast(
        self,
        pkt_type: int,
        payload: bytes,
        robot_type: Optional[int] = None,
    ) -> dict[str, bool]:
        """Send the same packet to every online robot.

        If `robot_type` is provided, only robots of that type receive it.
        Returns a dict robot_id -> success flag.
        """
        if self._send_fn is None:
            logger.error("[Fleet] broadcast: send_fn not configured")
            return {}

        targets = [
            r
            for r in self._robots.values()
            if r.online
            and r.writer is not None
            and (robot_type is None or r.robot_type == robot_type)
        ]
        if not targets:
            return {}

        # Bound concurrency to avoid flooding.
        semaphore = asyncio.Semaphore(FLEET_BROADCAST_MAX_CONCURRENCY)

        async def _send_one(record: RobotRecord) -> tuple[str, bool]:
            async with semaphore:
                try:
                    await self._send_fn(record.writer, pkt_type, payload)
                    return record.robot_id, True
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "[Fleet] broadcast to '%s' failed: %s",
                        record.robot_id,
                        e,
                    )
                    self.mark_disconnected(record.robot_id)
                    return record.robot_id, False

        results = await asyncio.gather(*(_send_one(r) for r in targets))
        return dict(results)

    # ── Debug ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"FleetManager(total={self.count}, " f"online={self.online_count})"
