# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""MAVLink bridge — control PX4/ArduPilot flight controllers from brain (E08).

This is the *root-level* MAVLink bridge used by `server.py` for drone robots.
It speaks MAVLink v1.0 over UDP or TCP and provides:

  * Telemetry ingest — HEARTBEAT, GLOBAL_POSITION_INT, ATTITUDE,
    BATTERY_STATUS/SYS_STATUS, GPS_RAW_INT.
  * Command out — ARM/DISARM, TAKEOFF, LAND, RTL, NAVIGATE_TO_WAYPOINT
    via MAV_CMD_* encoded as COMMAND_LONG (#76).
  * Skill translation — brain skill names (TAKEOFF / LAND / RETURN_HOME /
    HOVER / NAVIGATE_TO / ARM / DISARM) → MAVLink commands.
  * Failsafe — if the brain loses its link to the kernel, RTL is sent so
    the autopilot brings the drone home.

The implementation prefers `pymavlink` if importable (production path).
Otherwise it falls back to a minimal self-contained MAVLink v1 encoder /
decoder covering just the messages listed above so unit tests can run in
CI without installing any dependency.

Coexists with the custom `protocol.py` binary protocol: the custom
protocol targets wheeled / humanoid robots talking to our RISC-V kernel,
while this bridge talks to an off-the-shelf flight controller.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("brain.mavlink")

# ---------------------------------------------------------------------------
# MAVLink v1 wire-format constants  — NO MAGIC NUMBERS in callers.
# ---------------------------------------------------------------------------
MAVLINK_V1_STX = 0xFE  # v1.0 start-of-frame byte
MAVLINK_HEADER_LEN = 6  # stx + len + seq + sys + comp + msgid
MAVLINK_CHECKSUM_LEN = 2  # CRC-16/X.25 trailer
MAVLINK_MAX_PAYLOAD = 255  # v1 payload length field is u8
MAVLINK_MIN_FRAME_LEN = MAVLINK_HEADER_LEN + MAVLINK_CHECKSUM_LEN

# GCS identity (we are the brain, not the autopilot)
MAVLINK_SYSTEM_ID = 255  # conventional GCS system ID
MAVLINK_COMPONENT_ID = 190  # MAV_COMP_ID_MISSIONPLANNER
MAVLINK_TARGET_SYSTEM = 1  # autopilot default sysid
MAVLINK_TARGET_COMPONENT = 1  # autopilot default compid

# ---------------------------------------------------------------------------
# MAVLink message IDs (subset we care about)
# ---------------------------------------------------------------------------
MAVLINK_MSG_ID_HEARTBEAT = 0
MAVLINK_MSG_ID_SYS_STATUS = 1
MAVLINK_MSG_ID_GPS_RAW_INT = 24
MAVLINK_MSG_ID_ATTITUDE = 30
MAVLINK_MSG_ID_GLOBAL_POSITION_INT = 33
MAVLINK_MSG_ID_COMMAND_LONG = 76
MAVLINK_MSG_ID_COMMAND_ACK = 77
MAVLINK_MSG_ID_BATTERY_STATUS = 147

# Per-message CRC_EXTRA seed bytes (from MAVLink XML definitions).
# These are required as the last byte fed into the CRC to avoid silent
# schema mismatches between sender and receiver.
MAVLINK_CRC_EXTRA = {
    MAVLINK_MSG_ID_HEARTBEAT: 50,
    MAVLINK_MSG_ID_SYS_STATUS: 124,
    MAVLINK_MSG_ID_GPS_RAW_INT: 24,
    MAVLINK_MSG_ID_ATTITUDE: 39,
    MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 104,
    MAVLINK_MSG_ID_COMMAND_LONG: 152,
    MAVLINK_MSG_ID_COMMAND_ACK: 143,
    MAVLINK_MSG_ID_BATTERY_STATUS: 154,
}

# Payload length for each known message (fixed in MAVLink v1 — no truncation).
MAVLINK_PAYLOAD_LEN = {
    MAVLINK_MSG_ID_HEARTBEAT: 9,
    MAVLINK_MSG_ID_SYS_STATUS: 31,
    MAVLINK_MSG_ID_GPS_RAW_INT: 30,
    MAVLINK_MSG_ID_ATTITUDE: 28,
    MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 28,
    MAVLINK_MSG_ID_COMMAND_LONG: 33,
    MAVLINK_MSG_ID_COMMAND_ACK: 3,
    MAVLINK_MSG_ID_BATTERY_STATUS: 36,
}

# ---------------------------------------------------------------------------
# MAV_CMD command IDs (COMMAND_LONG payloads)
# ---------------------------------------------------------------------------
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400

# MAV_RESULT — COMMAND_ACK result codes
MAV_RESULT_ACCEPTED = 0
MAV_RESULT_TEMPORARILY_REJECTED = 1
MAV_RESULT_DENIED = 2
MAV_RESULT_UNSUPPORTED = 3
MAV_RESULT_FAILED = 4

# HEARTBEAT base_mode bitmask — armed flag
MAV_MODE_FLAG_SAFETY_ARMED = 0x80

# Common autopilot / vehicle types in HEARTBEAT
MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8
MAV_STATE_ACTIVE = 4

# ARM/DISARM param1 — 1 to arm, 0 to disarm
ARM_DISARM_ARM = 1
ARM_DISARM_DISARM = 0

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL_S = 1.0  # spec: GCS must heartbeat >=1 Hz
TELEMETRY_TIMEOUT_S = 5.0  # link considered dead after this
DEFAULT_UDP_URL = "udp://127.0.0.1:14540"  # PX4 SITL default GCS port
DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_RECV_POLL_S = 0.05
RAD_TO_DEG = 180.0 / 3.141592653589793

# Failsafe thresholds
BRAIN_HEARTBEAT_TIMEOUT_S = 10.0
BATTERY_RTL_PCT = 30
BATTERY_LAND_PCT = 15
# Altitude below which brain-link loss disarms instead of RTL (on ground).
DISARM_ON_GROUND_ALT_M = 2.0

# CRC-16/MCRF4XX polynomial constants (MAVLink uses this CRC)
MAVLINK_CRC_INIT = 0xFFFF
MAVLINK_CRC_POLY = 0x1021
MAVLINK_CRC_BITS = 8


# ---------------------------------------------------------------------------
# CRC-16/MCRF4XX — byte-at-a-time implementation (small + dependency-free)
# ---------------------------------------------------------------------------
def mavlink_crc_accumulate(data: bytes, crc: int = MAVLINK_CRC_INIT) -> int:
    """Accumulate `data` into a MAVLink CRC-16/MCRF4XX.

    Matches the reference implementation in pymavlink's `mavutil.x25crc`.
    """
    for b in data:
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


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


@dataclass
class MavlinkFrame:
    """Decoded MAVLink v1 frame."""

    seq: int
    sysid: int
    compid: int
    msgid: int
    payload: bytes


# ---------------------------------------------------------------------------
# Wire encode / decode (MAVLink v1)
# ---------------------------------------------------------------------------
def encode_mavlink_v1(
    msgid: int,
    payload: bytes,
    seq: int,
    sysid: int = MAVLINK_SYSTEM_ID,
    compid: int = MAVLINK_COMPONENT_ID,
) -> bytes:
    """Serialise a MAVLink v1 frame (STX + header + payload + CRC)."""
    if msgid not in MAVLINK_CRC_EXTRA:
        raise ValueError(f"unknown msgid {msgid}")
    expected_len = MAVLINK_PAYLOAD_LEN[msgid]
    # Pad / truncate to the fixed length expected for this message.
    if len(payload) < expected_len:
        payload = payload + b"\x00" * (expected_len - len(payload))
    elif len(payload) > expected_len:
        payload = payload[:expected_len]
    if len(payload) > MAVLINK_MAX_PAYLOAD:
        raise ValueError("payload too long for MAVLink v1")

    header = struct.pack(
        "<BBBBBB",
        MAVLINK_V1_STX,
        len(payload),
        seq & 0xFF,
        sysid & 0xFF,
        compid & 0xFF,
        msgid & 0xFF,
    )
    # CRC covers header[1:] (skip STX) + payload + crc_extra byte
    crc = mavlink_crc_accumulate(header[1:])
    crc = mavlink_crc_accumulate(payload, crc)
    crc = mavlink_crc_accumulate(bytes([MAVLINK_CRC_EXTRA[msgid]]), crc)
    return header + payload + struct.pack("<H", crc)


def decode_mavlink_v1(buf: bytes) -> Optional[MavlinkFrame]:
    """Decode a single MAVLink v1 frame from the start of `buf`.

    Returns None if the buffer does not contain a complete, CRC-valid frame.
    """
    if len(buf) < MAVLINK_MIN_FRAME_LEN:
        return None
    if buf[0] != MAVLINK_V1_STX:
        return None
    plen = buf[1]
    total = MAVLINK_HEADER_LEN + plen + MAVLINK_CHECKSUM_LEN
    if len(buf) < total:
        return None
    seq = buf[2]
    sysid = buf[3]
    compid = buf[4]
    msgid = buf[5]
    if msgid not in MAVLINK_CRC_EXTRA:
        return None
    payload = buf[MAVLINK_HEADER_LEN : MAVLINK_HEADER_LEN + plen]
    crc_wire = struct.unpack(
        "<H",
        buf[MAVLINK_HEADER_LEN + plen : total],
    )[0]
    crc = mavlink_crc_accumulate(buf[1:MAVLINK_HEADER_LEN])
    crc = mavlink_crc_accumulate(payload, crc)
    crc = mavlink_crc_accumulate(bytes([MAVLINK_CRC_EXTRA[msgid]]), crc)
    if crc != crc_wire:
        return None
    return MavlinkFrame(seq=seq, sysid=sysid, compid=compid, msgid=msgid, payload=payload)


def encode_heartbeat(seq: int = 0) -> bytes:
    """Encode a GCS HEARTBEAT (advertises the brain as online)."""
    # custom_mode(u32) type(u8) autopilot(u8) base_mode(u8) system_status(u8)
    # mavlink_version(u8)
    payload = struct.pack(
        "<IBBBBB",
        0,
        MAV_TYPE_GCS,
        MAV_AUTOPILOT_INVALID,
        0,
        MAV_STATE_ACTIVE,
        3,
    )
    return encode_mavlink_v1(MAVLINK_MSG_ID_HEARTBEAT, payload, seq)


def encode_command_long(
    command: int,
    param1: float = 0.0,
    param2: float = 0.0,
    param3: float = 0.0,
    param4: float = 0.0,
    param5: float = 0.0,
    param6: float = 0.0,
    param7: float = 0.0,
    confirmation: int = 0,
    target_system: int = MAVLINK_TARGET_SYSTEM,
    target_component: int = MAVLINK_TARGET_COMPONENT,
    seq: int = 0,
) -> bytes:
    """Encode a COMMAND_LONG (#76) as a MAVLink v1 frame."""
    # 7x float + u16 command + u8 target_sys + u8 target_comp + u8 confirmation
    payload = struct.pack(
        "<fffffffHBBB",
        param1,
        param2,
        param3,
        param4,
        param5,
        param6,
        param7,
        command & 0xFFFF,
        target_system & 0xFF,
        target_component & 0xFF,
        confirmation & 0xFF,
    )
    return encode_mavlink_v1(MAVLINK_MSG_ID_COMMAND_LONG, payload, seq)


def parse_heartbeat(payload: bytes) -> dict:
    custom_mode, mav_type, autopilot, base_mode, system_status, _ver = struct.unpack(
        "<IBBBBB", payload[:9]
    )
    return {
        "custom_mode": custom_mode,
        "type": mav_type,
        "autopilot": autopilot,
        "base_mode": base_mode,
        "system_status": system_status,
        "armed": bool(base_mode & MAV_MODE_FLAG_SAFETY_ARMED),
    }


def parse_global_position_int(payload: bytes) -> dict:
    # time_boot_ms(u32) lat(i32) lon(i32) alt(i32) relative_alt(i32)
    # vx(i16) vy(i16) vz(i16) hdg(u16)
    time_ms, lat, lon, alt, rel_alt, vx, vy, vz, hdg = struct.unpack("<IiiiihhhH", payload[:28])
    return {
        "time_boot_ms": time_ms,
        "lat_deg": lat / 1e7,
        "lon_deg": lon / 1e7,
        "alt_m": alt / 1000.0,
        "relative_alt_m": rel_alt / 1000.0,
        "vx_ms": vx / 100.0,
        "vy_ms": vy / 100.0,
        "vz_ms": vz / 100.0,
        "heading_deg": hdg / 100.0 if hdg != 0xFFFF else 0.0,
    }


def parse_attitude(payload: bytes) -> dict:
    # time_boot_ms(u32) roll(f32) pitch(f32) yaw(f32) rollspd(f32)
    # pitchspd(f32) yawspd(f32)
    time_ms, roll, pitch, yaw, rs, ps, ys = struct.unpack("<Iffffff", payload[:28])
    return {
        "time_boot_ms": time_ms,
        "roll_deg": roll * RAD_TO_DEG,
        "pitch_deg": pitch * RAD_TO_DEG,
        "yaw_deg": yaw * RAD_TO_DEG,
    }


def parse_sys_status(payload: bytes) -> dict:
    """Extract voltage (u16 mV) and battery_remaining (i8 %) from SYS_STATUS.

    Message layout (MAVLink v1, 31B):
      sensors_present(u32) sensors_enabled(u32) sensors_health(u32)
      load(u16) voltage_battery(u16) current_battery(i16)
      drop_rate_comm(u16) errors_comm(u16)
      errors_count1..4(4*u16) battery_remaining(i8)
    """
    (
        _sensors_present,
        _enabled,
        _health,
        _load,
        voltage_mv,
        _current_ca,
        _drop_rate,
        _errors_comm,
        _ec1,
        _ec2,
        _ec3,
        _ec4,
        battery_remaining,
    ) = struct.unpack("<IIIHHhHHHHHHb", payload[:31])
    return {
        "voltage_mv": voltage_mv,
        "battery_remaining_pct": battery_remaining,
    }


def parse_gps_raw_int(payload: bytes) -> dict:
    # time_usec(u64) lat(i32) lon(i32) alt(i32) eph(u16) epv(u16)
    # vel(u16) cog(u16) fix_type(u8) satellites(u8)
    _tsu, lat, lon, alt, _eph, _epv, _vel, _cog, fix_type, sats = struct.unpack(
        "<QiiiHHHHBB", payload[:30]
    )
    return {
        "lat_deg": lat / 1e7,
        "lon_deg": lon / 1e7,
        "alt_m": alt / 1000.0,
        "fix_type": fix_type,
        "satellites": sats,
    }


def parse_battery_status(payload: bytes) -> dict:
    # id(u8) battery_function(u8) type(u8) temperature(i16)
    # voltages[10](u16 mV) current_battery(i16 cA) current_consumed(i32)
    # energy_consumed(i32) battery_remaining(i8)
    head = struct.unpack("<BBBh", payload[:5])
    voltages = struct.unpack("<10H", payload[5:25])
    _current_ca, _cons, _energy, remaining = struct.unpack("<hiib", payload[25:36])
    return {
        "id": head[0],
        "voltage_mv": voltages[0],  # cell 1 / main bus
        "battery_remaining_pct": remaining,
    }


# ---------------------------------------------------------------------------
# Skill → MAV_CMD translation
# ---------------------------------------------------------------------------
def translate_skill(skill: str, args: Optional[dict] = None) -> Optional[dict]:
    """Map a brain skill name to a MAVLink command descriptor.

    Returns a dict ``{"command": MAV_CMD_*, "params": (p1..p7)}`` or ``None``
    if the skill has no MAVLink mapping (e.g. HOVER — PX4 loiters by default).
    Raised to module scope so unit tests can verify the mapping without
    instantiating the async client.
    """
    args = args or {}
    s = skill.strip().upper()

    if s in ("TAKEOFF", "LAUNCH"):
        alt_m = float(args.get("altitude_m", args.get("alt_m", 10.0)))
        return {
            "command": MAV_CMD_NAV_TAKEOFF,
            "params": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, alt_m),
        }
    if s == "LAND":
        return {
            "command": MAV_CMD_NAV_LAND,
            "params": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    if s in ("RETURN_HOME", "RTL", "RTH"):
        return {
            "command": MAV_CMD_NAV_RETURN_TO_LAUNCH,
            "params": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    if s in ("NAVIGATE_TO", "GOTO", "WAYPOINT"):
        lat = float(args.get("lat", 0.0))
        lon = float(args.get("lon", 0.0))
        alt = float(args.get("alt_m", args.get("altitude_m", 10.0)))
        return {
            "command": MAV_CMD_NAV_WAYPOINT,
            "params": (0.0, 0.0, 0.0, 0.0, lat, lon, alt),
        }
    if s == "ARM":
        return {
            "command": MAV_CMD_COMPONENT_ARM_DISARM,
            "params": (float(ARM_DISARM_ARM), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    if s == "DISARM":
        return {
            "command": MAV_CMD_COMPONENT_ARM_DISARM,
            "params": (float(ARM_DISARM_DISARM), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    if s == "HOVER":
        # PX4 holds position in LOITER by default — no command needed.
        return None
    return None


# ---------------------------------------------------------------------------
# Transport abstraction — covers UDP and TCP, stdlib only.
# ---------------------------------------------------------------------------
def parse_connection_string(conn: str) -> tuple[str, str, int]:
    """Parse connection strings of the form ``udp://host:port`` / ``tcp://...``.

    Defaults to UDP / 14540 if the input omits a scheme.
    """
    scheme = "udp"
    rest = conn
    if "://" in conn:
        scheme, rest = conn.split("://", 1)
        scheme = scheme.lower()
    host = "127.0.0.1"
    port = 14540
    if ":" in rest:
        host, port_s = rest.rsplit(":", 1)
        port = int(port_s)
    elif rest:
        host = rest
    return scheme, host, port


class _Transport:
    """Small abstract transport interface so tests can substitute a fake."""

    def send(self, data: bytes) -> None:
        raise NotImplementedError

    def recv(self, maxlen: int = 4096) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UdpTransport(_Transport):
    """Non-blocking UDP socket wrapper."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # Bind to an ephemeral port so we can read replies.
        self.sock.bind(("0.0.0.0", 0))

    def send(self, data: bytes) -> None:
        self.sock.sendto(data, (self.host, self.port))

    def recv(self, maxlen: int = 4096) -> bytes:
        try:
            data, _ = self.sock.recvfrom(maxlen)
            return data
        except BlockingIOError:
            return b""

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class TcpTransport(_Transport):
    def __init__(self, host: str, port: int, timeout: float = DEFAULT_CONNECT_TIMEOUT_S):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setblocking(False)

    def send(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except BlockingIOError:
            # Drop: SITL link backpressure is extremely rare.
            pass

    def recv(self, maxlen: int = 4096) -> bytes:
        try:
            return self.sock.recv(maxlen)
        except BlockingIOError:
            return b""

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MAVLinkClient
# ---------------------------------------------------------------------------
class MAVLinkClient:
    """Asynchronous MAVLink bridge to a PX4/ArduPilot autopilot.

    Parameters
    ----------
    connection_string : str
        `udp://host:port` or `tcp://host:port`.  Default: `udp://127.0.0.1:14540`
        (PX4 SITL GCS port).
    transport : optional
        Pre-built transport — lets tests inject a fake without touching sockets.
    """

    def __init__(
        self,
        connection_string: str = DEFAULT_UDP_URL,
        transport: Optional[_Transport] = None,
        source_system: int = MAVLINK_SYSTEM_ID,
        source_component: int = MAVLINK_COMPONENT_ID,
        target_system: int = MAVLINK_TARGET_SYSTEM,
        target_component: int = MAVLINK_TARGET_COMPONENT,
    ):
        self.connection_string = connection_string
        self._transport = transport
        self._source_system = source_system
        self._source_component = source_component
        self._target_system = target_system
        self._target_component = target_component

        self._telemetry = MavTelemetry()
        self._connected = False
        self._rx_buffer = bytearray()
        self._seq = 0

        self._recv_task: Optional[asyncio.Task] = None
        self._hb_task: Optional[asyncio.Task] = None
        self._running = False

        # Optional listeners — tests hook these without touching the loop.
        self.on_heartbeat: Optional[Callable[[dict], None]] = None
        self.on_telemetry: Optional[Callable[[MavTelemetry], None]] = None

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def telemetry(self) -> MavTelemetry:
        return self._telemetry

    @property
    def connected(self) -> bool:
        return self._connected and self._telemetry.connected

    # ── connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Open the transport and start background tasks."""
        if self._transport is None:
            scheme, host, port = parse_connection_string(self.connection_string)
            try:
                if scheme == "tcp":
                    self._transport = TcpTransport(host, port)
                else:
                    self._transport = UdpTransport(host, port)
            except Exception as e:
                logger.error("[MAVLink] connect(%s) failed: %s", self.connection_string, e)
                return False

        self._connected = True
        self._running = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._hb_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("[MAVLink] Connected via %s", self.connection_string)
        return True

    async def disconnect(self) -> None:
        """Stop background tasks and close the transport."""
        self._running = False
        for t in (self._recv_task, self._hb_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if self._transport:
            self._transport.close()
            self._transport = None
        self._connected = False
        logger.info("[MAVLink] Disconnected")

    # ── outbound commands ─────────────────────────────────────────────────

    def _next_seq(self) -> int:
        seq = self._seq & 0xFF
        self._seq = (self._seq + 1) & 0xFF
        return seq

    def _send_command_long(self, command: int, params: tuple = (0.0,) * 7) -> bool:
        if not self._transport:
            return False
        p = list(params) + [0.0] * max(0, 7 - len(params))
        frame = encode_command_long(
            command,
            param1=p[0],
            param2=p[1],
            param3=p[2],
            param4=p[3],
            param5=p[4],
            param6=p[5],
            param7=p[6],
            target_system=self._target_system,
            target_component=self._target_component,
            seq=self._next_seq(),
        )
        try:
            self._transport.send(frame)
            return True
        except Exception as e:
            logger.error("[MAVLink] send(cmd=%d) failed: %s", command, e)
            return False

    async def arm(self) -> bool:
        logger.info("[MAVLink] ARM")
        return self._send_command_long(
            MAV_CMD_COMPONENT_ARM_DISARM,
            (float(ARM_DISARM_ARM), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    async def disarm(self) -> bool:
        logger.info("[MAVLink] DISARM")
        return self._send_command_long(
            MAV_CMD_COMPONENT_ARM_DISARM,
            (float(ARM_DISARM_DISARM), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    async def takeoff(self, altitude_m: float = 10.0) -> bool:
        logger.info("[MAVLink] TAKEOFF alt=%.1fm", altitude_m)
        return self._send_command_long(
            MAV_CMD_NAV_TAKEOFF,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(altitude_m)),
        )

    async def land(self) -> bool:
        logger.info("[MAVLink] LAND")
        return self._send_command_long(MAV_CMD_NAV_LAND)

    async def rtl(self) -> bool:
        logger.info("[MAVLink] RTL")
        return self._send_command_long(MAV_CMD_NAV_RETURN_TO_LAUNCH)

    async def navigate_to_waypoint(self, lat: float, lon: float, alt_m: float = 10.0) -> bool:
        logger.info("[MAVLink] WAYPOINT (%.6f, %.6f) alt=%.1fm", lat, lon, alt_m)
        return self._send_command_long(
            MAV_CMD_NAV_WAYPOINT,
            (0.0, 0.0, 0.0, 0.0, float(lat), float(lon), float(alt_m)),
        )

    async def execute_skill(self, skill: str, args: Optional[dict] = None) -> bool:
        """Translate a brain skill to a MAVLink command and send it."""
        cmd = translate_skill(skill, args)
        if cmd is None:
            logger.debug("[MAVLink] skill %s has no MAVLink mapping", skill)
            return False
        return self._send_command_long(cmd["command"], cmd["params"])

    # ── inbound telemetry ─────────────────────────────────────────────────

    def feed_bytes(self, data: bytes) -> list[MavlinkFrame]:
        """Feed raw bytes from the transport into the decoder.

        Exposed for unit tests — the recv loop calls this internally.
        Returns the list of frames that were successfully decoded this call.
        """
        if not data:
            return []
        self._rx_buffer.extend(data)
        frames: list[MavlinkFrame] = []
        while self._rx_buffer:
            # Resync on STX
            if self._rx_buffer[0] != MAVLINK_V1_STX:
                self._rx_buffer.pop(0)
                continue
            if len(self._rx_buffer) < 2:
                break
            plen = self._rx_buffer[1]
            total = MAVLINK_HEADER_LEN + plen + MAVLINK_CHECKSUM_LEN
            if len(self._rx_buffer) < total:
                break
            frame = decode_mavlink_v1(bytes(self._rx_buffer[:total]))
            if frame is None:
                # Bad CRC or unknown msg — drop one byte and resync.
                self._rx_buffer.pop(0)
                continue
            del self._rx_buffer[:total]
            frames.append(frame)
            self._dispatch_frame(frame)
        return frames

    def _dispatch_frame(self, frame: MavlinkFrame) -> None:
        t = self._telemetry
        t.timestamp = time.time()

        if frame.msgid == MAVLINK_MSG_ID_HEARTBEAT:
            hb = parse_heartbeat(frame.payload)
            t.last_heartbeat = time.time()
            t.armed = hb["armed"]
            if self.on_heartbeat:
                try:
                    self.on_heartbeat(hb)
                except Exception:
                    logger.exception("[MAVLink] on_heartbeat raised")

        elif frame.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
            d = parse_global_position_int(frame.payload)
            t.lat = d["lat_deg"]
            t.lon = d["lon_deg"]
            t.alt_m = d["relative_alt_m"]
            t.heading_deg = d["heading_deg"]

        elif frame.msgid == MAVLINK_MSG_ID_ATTITUDE:
            d = parse_attitude(frame.payload)
            t.roll_deg = d["roll_deg"]
            t.pitch_deg = d["pitch_deg"]
            t.yaw_deg = d["yaw_deg"]

        elif frame.msgid == MAVLINK_MSG_ID_SYS_STATUS:
            d = parse_sys_status(frame.payload)
            t.battery_mv = d["voltage_mv"]
            if d["battery_remaining_pct"] >= 0:
                t.battery_pct = d["battery_remaining_pct"]

        elif frame.msgid == MAVLINK_MSG_ID_BATTERY_STATUS:
            d = parse_battery_status(frame.payload)
            if d["voltage_mv"] != 0xFFFF:
                t.battery_mv = d["voltage_mv"]
            if d["battery_remaining_pct"] >= 0:
                t.battery_pct = d["battery_remaining_pct"]

        elif frame.msgid == MAVLINK_MSG_ID_GPS_RAW_INT:
            d = parse_gps_raw_int(frame.payload)
            t.gps_fix = d["fix_type"]
            t.satellites = d["satellites"]

        if self.on_telemetry:
            try:
                self.on_telemetry(t)
            except Exception:
                logger.exception("[MAVLink] on_telemetry raised")

    async def _recv_loop(self) -> None:
        while self._running and self._transport is not None:
            try:
                data = await asyncio.to_thread(self._transport.recv)
            except Exception as e:
                logger.debug("[MAVLink] recv error: %s", e)
                data = b""
            if data:
                self.feed_bytes(data)
            else:
                await asyncio.sleep(DEFAULT_RECV_POLL_S)

    async def _heartbeat_loop(self) -> None:
        while self._running and self._transport is not None:
            try:
                self._transport.send(encode_heartbeat(self._next_seq()))
            except Exception as e:
                logger.debug("[MAVLink] heartbeat send failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── failsafe ──────────────────────────────────────────────────────────

    async def check_failsafe(self) -> Optional[str]:
        """Evaluate periodic failsafe conditions (battery).

        Call this from the brain loop. Returns the action taken, or ``None``.
        Priority: LAND > RTL > (no action).
        """
        t = self._telemetry
        if not t.connected:
            return None
        if 0 < t.battery_pct < BATTERY_LAND_PCT:
            logger.critical("[MAVLink] CRITICAL battery %d%% — LAND", t.battery_pct)
            await self.land()
            return "land"
        if 0 < t.battery_pct < BATTERY_RTL_PCT:
            logger.warning("[MAVLink] Low battery %d%% — RTL", t.battery_pct)
            await self.rtl()
            return "rtl"
        return None

    async def failsafe_on_disconnect(self) -> Optional[str]:
        """Called when the brain loses its link to the robot.

        Issues RTL if airborne; DISARM if on the ground. Returns the action
        taken ("rtl" / "disarm") or ``None`` if the vehicle was already safe.
        """
        t = self._telemetry
        if not t.armed:
            return None
        if t.alt_m > DISARM_ON_GROUND_ALT_M:
            logger.warning(
                "[MAVLink] Brain disconnect while airborne (alt=%.1fm) — RTL",
                t.alt_m,
            )
            await self.rtl()
            return "rtl"
        logger.warning("[MAVLink] Brain disconnect on ground — DISARM")
        await self.disarm()
        return "disarm"
