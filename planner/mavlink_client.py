# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""MAVLink bridge — control PX4/ArduPilot flight controllers from brain.

Translates brain skills (TAKEOFF, HOVER, NAVIGATE_TO, LAND) into MAVLink
COMMAND_LONG messages, and receives telemetry (GPS, attitude, battery)
back into brain state.

Coexists with custom protocol: custom → wheeled robots, MAVLink → drones.

Usage:
    client = MavlinkClient("/dev/ttyUSB0", baud=57600)
    await client.connect()
    await client.arm()
    await client.takeoff(alt_m=10)
    await client.goto(lat, lon, alt_m)
    await client.land()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("brain.mavlink")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAVLINK_SYSTEM_ID = 255  # GCS system ID
MAVLINK_COMPONENT_ID = 0  # GCS component
MAVLINK_TARGET_SYSTEM = 1  # autopilot system ID
MAVLINK_TARGET_COMPONENT = 1  # autopilot component

# MAVLink command IDs (subset)
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_DO_SET_MODE = 176

# MAVLink flight modes (PX4)
PX4_MODE_MANUAL = 0
PX4_MODE_STABILIZED = 2
PX4_MODE_OFFBOARD = 6
PX4_MODE_AUTO_LAND = 0x04 | (0x06 << 8)
PX4_MODE_AUTO_RTL = 0x04 | (0x05 << 8)

# Telemetry update rate
HEARTBEAT_INTERVAL_S = 1.0
TELEMETRY_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class MavTelemetry:
    """Latest telemetry from the autopilot."""

    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    heading_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    groundspeed_ms: float = 0.0
    battery_mv: int = 0
    battery_pct: int = 0
    armed: bool = False
    mode: str = ""
    gps_fix: int = 0
    satellites: int = 0
    last_heartbeat: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def connected(self) -> bool:
        if self.last_heartbeat == 0:
            return False
        return (time.time() - self.last_heartbeat) < TELEMETRY_TIMEOUT_S


# ---------------------------------------------------------------------------
# MavlinkClient
# ---------------------------------------------------------------------------
class MavlinkClient:
    """MAVLink bridge to PX4/ArduPilot autopilot."""

    def __init__(
        self,
        connection_string: str = "/dev/ttyUSB0",
        baud: int = 57600,
        source_system: int = MAVLINK_SYSTEM_ID,
    ):
        self._conn_str = connection_string
        self._baud = baud
        self._source_system = source_system
        self._mavlink = None  # pymavlink connection (lazy import)
        self._telemetry = MavTelemetry()
        self._connected = False
        self._recv_task: Optional[asyncio.Task] = None

    @property
    def telemetry(self) -> MavTelemetry:
        return self._telemetry

    @property
    def connected(self) -> bool:
        return self._connected and self._telemetry.connected

    # ── Connection ────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to autopilot via pymavlink."""
        try:
            from pymavlink import mavutil

            self._mavlink = mavutil.mavlink_connection(
                self._conn_str,
                baud=self._baud,
                source_system=self._source_system,
            )
            # Wait for heartbeat
            logger.info("[MAVLink] Waiting for heartbeat on %s...", self._conn_str)
            hb = await asyncio.to_thread(
                self._mavlink.wait_heartbeat,
                timeout=10,
            )
            if hb:
                self._connected = True
                self._telemetry.last_heartbeat = time.time()
                logger.info(
                    "[MAVLink] Connected to system %d component %d",
                    self._mavlink.target_system,
                    self._mavlink.target_component,
                )
                # Start telemetry receiver
                self._recv_task = asyncio.create_task(self._recv_loop())
                return True
            else:
                logger.error("[MAVLink] No heartbeat received")
                return False
        except ImportError:
            logger.error("[MAVLink] pymavlink not installed")
            return False
        except Exception as e:
            logger.error("[MAVLink] Connection failed: %s", e)
            return False

    async def disconnect(self):
        """Disconnect from autopilot."""
        if self._recv_task:
            self._recv_task.cancel()
        if self._mavlink:
            self._mavlink.close()
        self._connected = False
        logger.info("[MAVLink] Disconnected")

    # ── Commands ──────────────────────────────────────────────────────────

    async def arm(self) -> bool:
        """Arm motors."""
        return await self._send_command_long(
            MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1,
        )

    async def disarm(self) -> bool:
        """Disarm motors."""
        return await self._send_command_long(
            MAV_CMD_COMPONENT_ARM_DISARM,
            param1=0,
        )

    async def takeoff(self, alt_m: float = 10.0) -> bool:
        """Take off to altitude."""
        logger.info("[MAVLink] Takeoff to %.1fm", alt_m)
        return await self._send_command_long(
            MAV_CMD_NAV_TAKEOFF,
            param7=alt_m,
        )

    async def land(self) -> bool:
        """Land at current position."""
        logger.info("[MAVLink] Landing")
        return await self._send_command_long(MAV_CMD_NAV_LAND)

    async def rtl(self) -> bool:
        """Return to launch."""
        logger.info("[MAVLink] RTL")
        return await self._send_command_long(MAV_CMD_NAV_RETURN_TO_LAUNCH)

    async def goto(
        self,
        lat: float,
        lon: float,
        alt_m: float = 10.0,
    ) -> bool:
        """Navigate to GPS coordinate."""
        logger.info(
            "[MAVLink] GoTo (%.6f, %.6f) alt=%.1fm",
            lat,
            lon,
            alt_m,
        )
        return await self._send_command_long(
            MAV_CMD_NAV_WAYPOINT,
            param5=lat,
            param6=lon,
            param7=alt_m,
        )

    # ── Skill translation ─────────────────────────────────────────────────

    async def execute_skill(self, skill: str, args: dict | None = None):
        """Translate a brain skill to MAVLink command."""
        args = args or {}
        s = skill.strip().upper()

        if s == "TAKEOFF":
            alt = float(args.get("altitude_m", 10))
            await self.arm()
            await self.takeoff(alt)
        elif s == "LAND":
            await self.land()
        elif s == "RETURN_HOME":
            await self.rtl()
        elif s == "HOVER":
            pass  # PX4 holds position by default in LOITER
        elif s == "NAVIGATE_TO":
            lat = float(args.get("lat", 0))
            lon = float(args.get("lon", 0))
            alt = float(args.get("alt_m", 10))
            await self.goto(lat, lon, alt)
        elif s == "DISARM":
            await self.disarm()
        else:
            logger.warning("[MAVLink] Unknown skill: %s", skill)

    # ── Internal ──────────────────────────────────────────────────────────

    async def _send_command_long(
        self,
        command: int,
        param1: float = 0,
        param2: float = 0,
        param3: float = 0,
        param4: float = 0,
        param5: float = 0,
        param6: float = 0,
        param7: float = 0,
    ) -> bool:
        """Send a COMMAND_LONG message."""
        if not self._mavlink:
            return False
        try:
            await asyncio.to_thread(
                self._mavlink.mav.command_long_send,
                MAVLINK_TARGET_SYSTEM,
                MAVLINK_TARGET_COMPONENT,
                command,
                0,  # confirmation
                param1,
                param2,
                param3,
                param4,
                param5,
                param6,
                param7,
            )
            return True
        except Exception as e:
            logger.error("[MAVLink] Command %d failed: %s", command, e)
            return False

    async def _recv_loop(self):
        """Background task receiving MAVLink messages."""
        while self._connected:
            try:
                msg = await asyncio.to_thread(
                    self._mavlink.recv_match,
                    blocking=True,
                    timeout=1,
                )
                if msg is None:
                    continue
                msg_type = msg.get_type()
                self._process_message(msg_type, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("[MAVLink] Recv error: %s", e)

    def _process_message(self, msg_type: str, msg):
        """Process a received MAVLink message."""
        t = self._telemetry

        if msg_type == "HEARTBEAT":
            t.last_heartbeat = time.time()
            t.armed = (msg.base_mode & 128) != 0

        elif msg_type == "GLOBAL_POSITION_INT":
            t.lat = msg.lat / 1e7
            t.lon = msg.lon / 1e7
            t.alt_m = msg.alt / 1000.0
            t.heading_deg = msg.hdg / 100.0

        elif msg_type == "ATTITUDE":
            t.roll_deg = msg.roll * 57.2958
            t.pitch_deg = msg.pitch * 57.2958
            t.yaw_deg = msg.yaw * 57.2958

        elif msg_type == "SYS_STATUS":
            t.battery_mv = msg.voltage_battery
            t.battery_pct = msg.battery_remaining

        elif msg_type == "GPS_RAW_INT":
            t.gps_fix = msg.fix_type
            t.satellites = msg.satellites_visible

        t.timestamp = time.time()

    # ── Failsafe handling (E08) ──────────────────────────────────────────

    ## Brain heartbeat timeout — if brain stops commanding, trigger RTL.
    BRAIN_HEARTBEAT_TIMEOUT_S = 10.0
    ## Battery failsafe: RTL threshold percentage.
    BATTERY_RTL_PCT = 30
    ## Battery failsafe: LAND threshold percentage.
    BATTERY_LAND_PCT = 15

    async def check_failsafe(self) -> str | None:
        """Check failsafe conditions. Returns action taken or None.

        Call this periodically from the brain server loop.
        Failsafe priority: LAND > RTL > None.
        """
        t = self._telemetry
        if not t.connected:
            return None  # no telemetry = can't assess

        # Critical battery → immediate land
        if t.battery_pct > 0 and t.battery_pct < self.BATTERY_LAND_PCT:
            logger.critical("[MAVLink] CRITICAL battery %d%% — LAND", t.battery_pct)
            await self.land()
            return "land"

        # Low battery → RTL
        if t.battery_pct > 0 and t.battery_pct < self.BATTERY_RTL_PCT:
            logger.warning("[MAVLink] Low battery %d%% — RTL", t.battery_pct)
            await self.rtl()
            return "rtl"

        return None

    async def failsafe_on_disconnect(self):
        """Called when brain loses connection to the robot.

        Triggers RTL if the drone is armed and airborne.
        PX4 has its own RC-loss failsafe, but this covers brain-link loss.
        """
        t = self._telemetry
        if t.armed and t.alt_m > 2.0:
            logger.warning("[MAVLink] Brain disconnect — triggering RTL (alt=%.1fm)", t.alt_m)
            await self.rtl()
        elif t.armed:
            logger.warning("[MAVLink] Brain disconnect — disarming (on ground)")
            await self.disarm()
