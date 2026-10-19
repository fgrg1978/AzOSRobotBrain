# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Binary protocol for robot <-> brain communication.

Packet format:
  MAGIC (2B) | TYPE (1B) | LEN (2B LE) | PAYLOAD (0-1400B) | CRC8 (1B)

Robot -> Server types (0x01-0x7F):
  0x01  SENSOR_PACKET   wheeled: 62B, drone: 68B, humanoid: variable
  0x02  CAMERA_FRAME    variable
  0x03  STATUS          8 bytes (includes robot_type)
  0x04  OTA_ACK         3 bytes (status + slot + reserved)
  0x05  SENSOR_COMPACT  20 bytes (low-bandwidth: LoRa/RF)

Server -> Robot types (0x80-0xFF):
  0x80  ACTUATOR_CMD    3 + 2*N bytes (generic: type + channels)
  0x81  MODE_CMD        1 byte
  0x82  WAYPOINT_CMD    14 bytes
  0x83  CONFIG_CMD      40 bytes

Robot types:
  0  WHEELED    differential drive (2 channels: speed_l, speed_r)
  1  DRONE      quad rotor (4 channels: throttle, roll, pitch, yaw)
  2  HUMANOID   joint angles (N channels)
  3  ACKERMANN  car/tractor (2 channels: speed, steer_angle)
"""

import asyncio
import collections
import os
import struct
import weakref
from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING

MAGIC = b"\x42\x52"  # "BR"

# Packet types — Robot -> Server
SENSOR_PACKET = 0x01
CAMERA_FRAME = 0x02
STATUS = 0x03
OTA_ACK = 0x04  # kernel ack for OTA chunks (matches kernel PKT_OTA_ACK)
SENSOR_COMPACT = 0x05  # low-bandwidth sensor frame (LoRa/RF)

# RFC-0039 (AzOS/rfcs): capability declaration, both directions.
# RESERVED, not yet sent or handled on either side — the number is claimed so a
# future implementation and a current build cannot disagree about what 0x06
# means.
#
# Payload will be `epoch u8 | n u8 | types [u8; n]`: the packet types this side
# both parses and ACTS ON. The distinction matters in both directions. Here,
# `_dispatch` in server.py handles exactly three types and OTA_ACK/SENSOR_COMPACT
# are declared with no handler; on the robot, WAYPOINT and the three OTA types
# are decoded and dispatched nowhere. From a sender's side those are as
# ineffective as a type that does not exist, so they must not be declared.
#
# Why this exists: the wire has no version, no feature bits and no handshake,
# and neither side's dispatch has a final `else`. A brain that puts a robot in
# CONTAINED with SEMANTIC_LEVEL against a kernel predating RFC-0037 gets no
# error and the robot keeps running at FULL. The robot now RECORDS a type it
# does not act on (AzOS `SAFETY_UNKNOWN_PKT`); this handshake is the other
# half.
CAPS = 0x06  # RFC-0039 capability declaration — reserved, not implemented

# Packet types — Server -> Robot
ACTUATOR_CMD = 0x80
VELOCITY_CMD = 0x80  # alias (backward compat)
MODE_CMD = 0x81
WAYPOINT_CMD = 0x82
CONFIG_CMD = 0x83
OTA_BEGIN = 0x84
OTA_CHUNK = 0x85
OTA_END = 0x86
PAYLOAD_CMD = 0x87  # E04: payload control (spray, gripper, cam trigger)
ESTOP_CMD = 0x88  # remote emergency stop
PREDICT_CMD = 0x89  # RFC-0034: predicted next actuator command + confidence
DEGRADE_CMD = 0x8A  # RFC-0036: brain-triggered degraded mode (capability containment)
SEMANTIC_LEVEL_CMD = 0x8B  # RFC-0037: graded degrade-level index (0=FULL…3=CONTAINED)

# Robot types
ROBOT_WHEELED = 0
ROBOT_DRONE = 1
ROBOT_HUMANOID = 2
ROBOT_ACKERMANN = 3

# Config/CLI robot-type name ↔ ROBOT_* wire constant. Single source of truth:
# the mapping used to exist as two independent inline dicts (policy.get_
# translator and server._on_status) that had already drifted — the server-side
# copy was missing ackermann=3, so an ackermann chassis silently reported
# itself as "wheeled" to the experience/meta/task-planner stack. Type
# selection now feeds the safety profile (see server.BrainServer), so a
# disagreement here is a safety bug, not a cosmetic one.
ROBOT_TYPE_BY_NAME: dict[str, int] = {
    "wheeled": ROBOT_WHEELED,
    "drone": ROBOT_DRONE,
    "humanoid": ROBOT_HUMANOID,
    "ackermann": ROBOT_ACKERMANN,
}
ROBOT_TYPE_NAME_BY_ID: dict[int, str] = {v: k for k, v in ROBOT_TYPE_BY_NAME.items()}

#: Fallback when a config string / wire byte names no known robot type.
ROBOT_TYPE_DEFAULT_NAME: str = "wheeled"

# ActuatorCmd flags
FLAG_EMERGENCY = 0x01
FLAG_ALERT = 0x02
FLAG_LOW_CONFIDENCE = 0x04  # RFC-0035: kernel tightens the motor envelope

# Actuator types (for ActuatorCmd.actuator_type)
ACT_DIFF_DRIVE = 0
ACT_QUAD_ROTOR = 1
ACT_HUMANOID = 2
ACT_ACKERMANN = 3

# LED state codes (sent via CONFIG_CMD, config_key=LED_CONFIG_KEY)
LED_CONFIG_KEY = 0x10  # config key byte for LED commands
LED_OFF = 0x00
LED_GREEN = 0x01  # monitoring, all clear
LED_GREEN_BLINK = 0x02  # mapping perimeter
LED_YELLOW = 0x03  # possible detection (VLM analyzing)
LED_YELLOW_BLINK = 0x04  # investigating (navigating to zone)
LED_RED = 0x05  # confirmed detection, recording
LED_RED_BLINK = 0x06  # active tracking
LED_RED_STROBE = 0x07  # panic / deterrent
LED_BLUE = 0x08  # returning to dock
LED_BLUE_BLINK = 0x09  # low battery
LED_WHITE_FLASH = 0x0A  # photo taken (feedback)

# Buzzer config (sent via CONFIG_CMD, config_key=BUZZER_CONFIG_KEY)
BUZZER_CONFIG_KEY = 0x15
BUZZER_OFF = 0x00
BUZZER_BEEP = 0x01  # 3 short beeps (alert acknowledgement)
BUZZER_SIREN = 0x02  # continuous siren (deterrent)
BUZZER_CHIRP = 0x03  # single chirp (confirmation feedback)

# Deterrent hardware config keys (sent via CONFIG_CMD)
SIREN_CONFIG_KEY = 0x16  # siren module (12V via MOSFET)
SPOTLIGHT_CONFIG_KEY = 0x17  # LED 10W COB spotlight (via MOSFET)
LASER_CONFIG_KEY = 0x18  # green laser 532nm (via MOSFET)
SERVO_PAN_KEY = 0x19  # pan servo angle (0-180 degrees)
SERVO_TILT_KEY = 0x1A  # tilt servo angle (0-180 degrees)
SPEAKER_CONFIG_KEY = 0x1B  # PAM8403 amplifier + speaker

# Device on/off values (siren, spotlight, laser)
DEVICE_OFF = 0x00
DEVICE_ON = 0x01
SPOTLIGHT_STROBE = 0x02  # strobing mode for spotlight

# Speaker audio file IDs
SPEAKER_STOP = 0x00
SPEAKER_WARNING = 0x01  # spoken warning clip ("Attention. Monitored area.")
SPEAKER_DOG_BARK = 0x02  # dog bark loop
SPEAKER_SIREN_FX = 0x03  # siren sound effect (software, not hardware siren)

# Digital sensor flags (bit flags in sensor_flags u16 field)
SENSOR_FLAG_PIR = 0x0001  # PIR motion detected
SENSOR_FLAG_SOUND = 0x0002  # Sound sensor triggered (glass break, impact)
SENSOR_FLAG_IR = 0x0004  # IR proximity triggered


def crc8(data: bytes) -> int:
    """CRC-8/MAXIM (polynomial 0x31)."""
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_packet(pkt_type: int, payload: bytes) -> bytes:
    """Build a wire-format packet."""
    header = MAGIC + struct.pack("<BH", pkt_type, len(payload))
    frame = header + payload
    return frame + bytes([crc8(frame)])


def parse_packet(data: bytes) -> Optional[tuple[int, bytes]]:
    """Parse a wire-format packet. Returns (type, payload) or None."""
    if len(data) < 6:
        return None
    if data[:2] != MAGIC:
        return None
    pkt_type, length = struct.unpack_from("<BH", data, 2)
    total = 5 + length + 1
    if len(data) < total:
        return None
    payload = data[5 : 5 + length]
    if data[5 + length] != crc8(data[: 5 + length]):
        return None
    return (pkt_type, payload)


#: Largest legitimate packet payload we will ever accept. Anything
#: bigger is treated as malformed/hostile so a corrupt or malicious
#: peer can't make us readexactly() up to 64 KiB on every read and
#: hang the connection (slowloris-style DoS).
MAX_PAYLOAD_BYTES = 4 * 1024


#
# ── The brain link: a Noise session per connection ───────────────────────────
#
# When a link key is configured (`AZOS_BRAIN_LINK_KEY`), the wire frame is
#   Noise( brain_protocol_frame )
# and there is no other mode: no MAC-only link, no switch to turn it off. The
# transport's message counter stops in-session replay and a fresh handshake
# per connection stops cross-session replay. The static Noise keys derive from
# the link key — one provisioned secret, same as the kernel's key sector
# (secure_channel.py has the handshake: the robot is the initiator, this brain
# the responder, IK when `AZOS_BRAIN_NOISE_SK` is set and the robot's signed
# topology pins its public half, XX else). `server.py` does not start without
# the key.
#
# The session is per TCP connection: `perform_handshake()` registers a `Link`
# under the connection's reader and writer once the handshake reaches
# ESTABLISHED, and `read_packet`/`send_packet` find it from the stream they are
# given. A second connection (C1's camera connection) gets its own channel and
# record counters instead of replacing the control connection's.
#
if TYPE_CHECKING:
    from secure_channel import SecureChannel

_encrypt_psk: "bytes | None" = None
#: The brain's own static Noise private key (`AZOS_BRAIN_NOISE_SK`): the key
#: whose public half the robot pins for Noise_IK. `None`: only XX is served.
_noise_static: "bytes | None" = None


class Link:
    """One connection's Noise session."""

    def __init__(self, secure_channel: "SecureChannel") -> None:
        self.secure_channel = secure_channel


#: Established links by stream. Weak keys, so a link never outlives its
#: streams; `drop_link` removes it on disconnect.
_links_by_reader: "weakref.WeakKeyDictionary[asyncio.StreamReader, Link]" = (
    weakref.WeakKeyDictionary()
)
_links_by_writer: "weakref.WeakKeyDictionary[asyncio.StreamWriter, Link]" = (
    weakref.WeakKeyDictionary()
)
#: Session ids of recent handshakes; a repeat is refused.  Process lifetime
#: only — it does not survive a brain restart.
_RECENT_SESSION_IDS = 64
_recent_session_ids: "collections.deque[bytes]" = collections.deque(maxlen=_RECENT_SESSION_IDS)

def enable_encrypt_link() -> bool:
    """Arm the encrypted link iff a link key is configured. Idempotent.

    With a key the link is always a Noise session: there is no environment
    switch that turns it off, and no MAC-only mode. Without one this returns
    False, and `server.py` refuses to run.

    Arming only records the PSK; the actual `SecureChannel` (with fresh
    per-connection ephemeral keys) is created in `perform_handshake`. A
    malformed `AZOS_BRAIN_NOISE_SK` raises: pinned-key deployments must not
    quietly fall back to the derived XX identity.
    """
    global _encrypt_psk, _noise_static
    from secure_channel import load_link_key, load_noise_static

    key = load_link_key()
    if key is None:
        _encrypt_psk = None
        _noise_static = None
        return False
    _noise_static = load_noise_static()
    _encrypt_psk = key
    return True


def encrypt_link_armed() -> bool:
    """True iff the encrypted link is armed (a handshake must run per
    connection before the channel is usable)."""
    return _encrypt_psk is not None


def link_for_reader(reader: asyncio.StreamReader) -> "Link | None":
    """The established RFC-0019 link of the connection `reader` belongs to."""
    return _links_by_reader.get(reader)


def link_for_writer(writer: asyncio.StreamWriter) -> "Link | None":
    """The established RFC-0019 link of the connection `writer` belongs to."""
    return _links_by_writer.get(writer)


def forbid_sends(writer: asyncio.StreamWriter) -> None:
    """Make an established connection receive-only: its reader keeps the
    session, and with encryption armed any `send_packet` on `writer` refuses
    (no link for the stream)."""
    _links_by_writer.pop(writer, None)


def drop_link(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Forget this connection's session — call on disconnect so a reconnect
    performs a fresh handshake with new ephemeral keys. Other connections'
    links are untouched."""
    _links_by_reader.pop(reader, None)
    _links_by_writer.pop(writer, None)


async def _read_handshake_frame(reader: asyncio.StreamReader) -> bytes:
    """One handshake frame: the 2-byte length prefix, checked before the body
    is read, then the body."""
    from secure_channel import FRAME_HEAD, handshake_frame_size

    head = await reader.readexactly(FRAME_HEAD)
    total = handshake_frame_size(head)
    return head + await reader.readexactly(total - FRAME_HEAD)


async def perform_handshake(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, timeout: float = 10.0
) -> bool:
    """Brain-side (responder) RFC-0019 handshake. Call once, immediately after
    the TCP connection is accepted and BEFORE any packet is sent. The robot
    speaks first: one pattern byte, then Noise handshake frames.

    Returns True on success (channel established) or when encryption is not
    armed (no-op pass-through). Returns False on any handshake failure — the
    caller MUST drop the connection (no silent plaintext fallback). There is
    no REJECT frame: a refused handshake just ends the connection.

    `timeout` is deliberately generous: under QEMU TCG the kernel's X25519
    work is wall-clock slow (~830k cycles measured), so a tight deadline would
    make the live handshake flaky for the same reason the TCP handshake
    deadline had to move 500ms→2s.
    """
    if _encrypt_psk is None:
        return True  # encryption not armed → nothing to do
    from secure_channel import SecureChannel

    sc = SecureChannel(_encrypt_psk, is_initiator=False, static_key=_noise_static)
    drop_link(reader, writer)

    async def run() -> None:
        sc.begin((await reader.readexactly(1))[0])
        while not sc.is_established:
            reply = sc.handle_handshake(await _read_handshake_frame(reader))
            if reply:
                writer.write(reply)
                await writer.drain()

    try:
        await asyncio.wait_for(run(), timeout)
        sid = sc.session_id
        if sid is None or sid in _recent_session_ids:
            raise PermissionError("session id repeated")
    except (
        asyncio.TimeoutError,
        asyncio.IncompleteReadError,
        ConnectionError,
        ValueError,
        PermissionError,  # LinkViolation is one
        RuntimeError,
    ) as e:
        sc.close()
        print(f"[BRAIN] secure_channel: RFC-0019 handshake failed: {str(e) or type(e).__name__}")
        return False
    _recent_session_ids.append(sid)
    link = Link(sc)
    _links_by_reader[reader] = link
    _links_by_writer[writer] = link
    print("[BRAIN] secure_channel: encrypted link established")
    return True


#
# ── Optional RFC-0021 multi-stream framing (OUTERMOST layer) ──────────────────
#
# When `AZOS_BRAIN_MULTI_STREAM=1`, every wire frame is prefixed with a
# 3-byte multi-stream header `[stream_id][len LE u16]` (multi_stream.py, mirror
# of crates/multi-stream). The control traffic (brain protocol, sealed in a
# Noise session when a key is configured) rides STREAM_CONTROL; camera/lidar/etc. get their own
# stream-ids (Phase D). Demux happens BEFORE decode, so unencrypted bulk
# streams (video) and the encrypted control stream coexist on one TCP conn.
# Independent of the link key — composes outside the session.
#
_multi_stream_armed: bool = False


def enable_multi_stream() -> bool:
    """Arm RFC-0021 multi-stream framing if `AZOS_BRAIN_MULTI_STREAM=1`.
    Pure framing (no key); composes outside the Noise session. Idempotent."""
    global _multi_stream_armed
    _multi_stream_armed = os.environ.get("AZOS_BRAIN_MULTI_STREAM", "").strip() == "1"
    return _multi_stream_armed


def multi_stream_armed() -> bool:
    return _multi_stream_armed


def _refuse_unlinked_stream() -> None:
    """Encryption is armed but this stream never completed a handshake: stop
    rather than read or write it as plaintext (RFC-0019: no silent fallback)."""
    raise ConnectionResetError("RFC-0019: no established link for this connection")


def _decode_inner(frame: bytes, link: "Link | None") -> Optional[tuple[int, bytes]]:
    """Decode a COMPLETE inner frame (already de-multiplexed) into
    (pkt_type, payload): open the sealed message and parse the brain frame it
    carries, or just parse it when `link` (the connection's Noise session) is
    None. Returns None on any decrypt/parse failure."""
    if link is not None:
        from secure_channel import LinkViolation

        # One control-stream frame carries exactly one sealed message.
        try:
            message = link.secure_channel.open_message(frame)
        except LinkViolation as e:
            print(f"[BRAIN] secure_channel: transport violation ({e}) — closing")
            return None
        return parse_packet(message)
    return parse_packet(frame)


async def read_packet(reader: asyncio.StreamReader) -> Optional[tuple[int, bytes]]:
    """Read one packet from an asyncio StreamReader, with the RFC-0019 session
    of the connection `reader` belongs to."""
    link = _links_by_reader.get(reader)
    if link is None and _encrypt_psk is not None:
        _refuse_unlinked_stream()
    if _multi_stream_armed:
        # Outermost framing: read [stream_id][len][payload], demux by stream-id,
        # then decode the inner frame. Non-control streams (camera/etc.) are
        # drained and skipped here — the control reader only yields STREAM_CONTROL
        # packets (video routing is Phase D, a separate consumer).
        import multi_stream as _ms

        while True:
            head = await reader.readexactly(_ms.HEADER_LEN)
            stream_id = head[0]
            plen = int.from_bytes(head[_ms.STREAM_ID_BYTES : _ms.HEADER_LEN], "little")
            if plen > _ms.MAX_PAYLOAD_LEN:
                return None
            payload = await reader.readexactly(plen)
            if stream_id == _ms.STREAM_CONTROL:
                return _decode_inner(payload, link)
            # else: non-control stream — ignore in the control reader for now.
    if link is not None:
        # Encrypted mode: read frames until one message is complete (a rekey
        # frame in between advances the receive key), then parse the brain
        # frame it carries. Frame layout: length(2 BE) + ciphertext || tag.
        from secure_channel import FRAME_HEAD, LinkViolation, SecureChannel

        sc = link.secure_channel
        try:
            while True:
                head = await reader.readexactly(FRAME_HEAD)
                # A bad length ends the session before any body is read, so
                # an oversize length claim never becomes a readexactly()
                # (slowloris guard, mirrors the plaintext-path check below).
                body = await reader.readexactly(sc.frame_size(head) - FRAME_HEAD)
                kind, message = sc.open_frame(head + body)
                if kind == SecureChannel.RECORD_DATA:
                    break
        except LinkViolation as e:
            print(f"[BRAIN] secure_channel: transport violation ({e}) — closing")
            return None
        return parse_packet(message)
    # Plaintext path (unchanged).
    header = await reader.readexactly(5)
    if header[:2] != MAGIC:
        return None
    pkt_type, length = struct.unpack_from("<BH", header, 2)
    if length > MAX_PAYLOAD_BYTES:
        # Refuse oversize. Returning None signals the caller to drop
        # this connection; without this guard a peer claiming length=65535
        # makes us block readexactly(65536) forever even if they only
        # intend to send a few bytes.
        return None
    rest = await reader.readexactly(length + 1)
    payload = rest[:length]
    if crc8(header + payload) != rest[length]:
        return None
    return (pkt_type, payload)


async def send_packet(writer: asyncio.StreamWriter, pkt_type: int, payload: bytes) -> None:
    """Send one packet via an asyncio StreamWriter, with the RFC-0019 session
    of the connection `writer` belongs to."""
    frame = build_packet(pkt_type, payload)
    link = _links_by_writer.get(writer)
    if link is not None:
        if not link.secure_channel.is_established:
            # The session ended (a transport violation); never fall back.
            raise ConnectionResetError("RFC-0019 session ended")
        inner = link.secure_channel.seal_message(frame)
    elif _encrypt_psk is not None:
        _refuse_unlinked_stream()
        return
    else:
        inner = frame
    # Outermost: multi-stream framing on STREAM_CONTROL when armed.
    if _multi_stream_armed:
        import multi_stream as _ms

        inner = _ms.wrap(_ms.STREAM_CONTROL, inner)
    writer.write(inner)
    await writer.drain()


# ── SensorPacket — common header + per-type payload ──────────────────────────

# Common header: timestamp(8) + battery(2) + accel(12) + gyro(12) = 34 bytes
_HDR_FMT = "<Q3i3iH"  # 34 bytes
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # 34

# Wheeled extra: odom_dist(4) + odom_hdg(4) + enc_l(8) + enc_r(8)
#                + range_front(2) + range_right(2) = 28 bytes
# Extended: + sensor_flags(2) = 30 bytes
_WHL_FMT = "<2i2q2H"
_WHL_SIZE = struct.calcsize(_WHL_FMT)  # 28
_WHL_FLAGS_FMT = "<H"
_WHL_FLAGS_SIZE = struct.calcsize(_WHL_FLAGS_FMT)  # 2

# Drone extra: baro(4) + mag(6) + gps_lat(4) + gps_lon(4) + gps_alt(4) + sonar(2) = 24 bytes
_DRN_FMT = "<i3h3iH"
_DRN_SIZE = struct.calcsize(_DRN_FMT)  # 24 — but header already has battery


@dataclass
class SensorPacket:
    """Wheeled robot sensor packet (common header + wheeled payload).

    Extended format includes sensor_flags (u16 bit flags for PIR/sound/IR).
    Parser auto-detects legacy (62B) vs extended (64B) payloads.
    """

    # Common header
    timestamp_ms: int
    battery_mv: int
    accel_mg: tuple[int, int, int]
    gyro_mdps: tuple[int, int, int]
    # Wheeled payload
    odom_dist_mm: int
    odom_hdg_cdeg: int
    encoder_l: int
    encoder_r: int
    range_front_mm: int
    range_right_mm: int
    # Extended: digital sensor flags (PIR, sound, IR)
    sensor_flags: int = 0

    ROBOT_TYPE = ROBOT_WHEELED

    #: Minimum payload length (header + wheeled fields). `sensor_flags` is an
    #: optional extension beyond this — see the auto-detect below.
    MIN_SIZE = _HDR_SIZE + _WHL_SIZE

    @classmethod
    def from_bytes(cls, data: bytes) -> "SensorPacket":
        if len(data) < cls.MIN_SIZE:
            raise ValueError(f"wheeled sensor packet truncated: {len(data)} < {cls.MIN_SIZE} bytes")
        ts, ax, ay, az, gx, gy, gz, batt = struct.unpack_from(_HDR_FMT, data)
        od, oh, el, er, rf, rr = struct.unpack_from(_WHL_FMT, data, _HDR_SIZE)
        # Auto-detect extended format with sensor_flags
        flags = 0
        if len(data) >= _HDR_SIZE + _WHL_SIZE + _WHL_FLAGS_SIZE:
            (flags,) = struct.unpack_from(_WHL_FLAGS_FMT, data, _HDR_SIZE + _WHL_SIZE)
        return cls(
            timestamp_ms=ts,
            battery_mv=batt,
            accel_mg=(ax, ay, az),
            gyro_mdps=(gx, gy, gz),
            odom_dist_mm=od,
            odom_hdg_cdeg=oh,
            encoder_l=el,
            encoder_r=er,
            range_front_mm=rf,
            range_right_mm=rr,
            sensor_flags=flags,
        )

    def to_bytes(self) -> bytes:
        hdr = struct.pack(
            _HDR_FMT, self.timestamp_ms, *self.accel_mg, *self.gyro_mdps, self.battery_mv
        )
        whl = struct.pack(
            _WHL_FMT,
            self.odom_dist_mm,
            self.odom_hdg_cdeg,
            self.encoder_l,
            self.encoder_r,
            self.range_front_mm,
            self.range_right_mm,
        )
        flags = struct.pack(_WHL_FLAGS_FMT, self.sensor_flags)
        return hdr + whl + flags


@dataclass
class SensorPacketDrone:
    """Drone sensor packet (common header + drone payload)."""

    # Common header
    timestamp_ms: int
    battery_mv: int
    accel_mg: tuple[int, int, int]
    gyro_mdps: tuple[int, int, int]
    # Drone payload
    baro_pa: int
    mag_ut: tuple[int, int, int]
    gps_lat_deg7: int
    gps_lon_deg7: int
    gps_alt_cm: int
    sonar_down_mm: int

    ROBOT_TYPE = ROBOT_DRONE

    #: Minimum payload length (header + drone fields).
    MIN_SIZE = _HDR_SIZE + _DRN_SIZE

    @classmethod
    def from_bytes(cls, data: bytes) -> "SensorPacketDrone":
        if len(data) < cls.MIN_SIZE:
            raise ValueError(f"drone sensor packet truncated: {len(data)} < {cls.MIN_SIZE} bytes")
        ts, ax, ay, az, gx, gy, gz, batt = struct.unpack_from(_HDR_FMT, data)
        baro, mx, my, mz, lat, lon, alt, sonar = struct.unpack_from(_DRN_FMT, data, _HDR_SIZE)
        return cls(
            timestamp_ms=ts,
            battery_mv=batt,
            accel_mg=(ax, ay, az),
            gyro_mdps=(gx, gy, gz),
            baro_pa=baro,
            mag_ut=(mx, my, mz),
            gps_lat_deg7=lat,
            gps_lon_deg7=lon,
            gps_alt_cm=alt,
            sonar_down_mm=sonar,
        )

    def to_bytes(self) -> bytes:
        hdr = struct.pack(
            _HDR_FMT, self.timestamp_ms, *self.accel_mg, *self.gyro_mdps, self.battery_mv
        )
        drn = struct.pack(
            _DRN_FMT,
            self.baro_pa,
            *self.mag_ut,
            self.gps_lat_deg7,
            self.gps_lon_deg7,
            self.gps_alt_cm,
            self.sonar_down_mm,
        )
        return hdr + drn


@dataclass
class SensorPacketHumanoid:
    """Humanoid sensor packet (common header + joint angles + foot pressure)."""

    # Common header
    timestamp_ms: int
    battery_mv: int
    accel_mg: tuple[int, int, int]
    gyro_mdps: tuple[int, int, int]
    # Humanoid payload
    joint_angles: list[int]  # centidegrees, variable length
    foot_pressure_l: int  # mN
    foot_pressure_r: int  # mN

    ROBOT_TYPE = ROBOT_HUMANOID

    # Anti-DoS bound: real humanoids have <= 32 actuated joints. Reject
    # packets claiming more so a malformed kernel/man-in-the-middle can't
    # cause struct.unpack_from to either read past the buffer (raising
    # struct.error and spamming the asyncio handler) or — worse on
    # platforms where bytes are pre-padded — consume all available memory.
    MAX_JOINTS = 32

    @classmethod
    def from_bytes(cls, data: bytes) -> "SensorPacketHumanoid":
        if len(data) < _HDR_SIZE + 1:
            raise ValueError("humanoid sensor packet truncated (header)")
        ts, ax, ay, az, gx, gy, gz, batt = struct.unpack_from(_HDR_FMT, data)
        offset = _HDR_SIZE
        num_joints = data[offset]
        offset += 1
        if num_joints > cls.MAX_JOINTS:
            raise ValueError(
                f"humanoid sensor packet num_joints={num_joints} > MAX={cls.MAX_JOINTS}"
            )
        joints_bytes = num_joints * 2
        if len(data) < offset + joints_bytes + 4:
            raise ValueError("humanoid sensor packet truncated (joints+feet)")
        joints = list(struct.unpack_from(f"<{num_joints}h", data, offset))
        offset += joints_bytes
        fl, fr = struct.unpack_from("<HH", data, offset)
        return cls(
            timestamp_ms=ts,
            battery_mv=batt,
            accel_mg=(ax, ay, az),
            gyro_mdps=(gx, gy, gz),
            joint_angles=joints,
            foot_pressure_l=fl,
            foot_pressure_r=fr,
        )

    def to_bytes(self) -> bytes:
        n = len(self.joint_angles)
        hdr = struct.pack(
            _HDR_FMT, self.timestamp_ms, *self.accel_mg, *self.gyro_mdps, self.battery_mv
        )
        joints = struct.pack(f"<B{n}h", n, *self.joint_angles)
        foot = struct.pack("<HH", self.foot_pressure_l, self.foot_pressure_r)
        return hdr + joints + foot


def sensor_packet_from_bytes(
    robot_type: int, data: bytes
) -> Union["SensorPacket", "SensorPacketDrone", "SensorPacketHumanoid"]:
    """Parse a sensor packet according to robot type."""
    if robot_type == ROBOT_WHEELED:
        return SensorPacket.from_bytes(data)
    if robot_type == ROBOT_DRONE:
        return SensorPacketDrone.from_bytes(data)
    if robot_type == ROBOT_HUMANOID:
        return SensorPacketHumanoid.from_bytes(data)
    return SensorPacket.from_bytes(data)  # fallback to wheeled


# ── SensorCompact — low-bandwidth (LoRa/RF) ──────────────────────────────────


@dataclass
class SensorCompact:
    """Compact sensor packet for low-bandwidth links (20 bytes).

    NOTE: Digital sensor flags (PIR motion, IR proximity, sound detection)
    could be carried as bit flags in the 'mode' byte's upper bits or in a
    dedicated flags field if the format is extended. Currently unused bits
    in 'mode' (bits 3-7) are available for this purpose.
    """

    lat_deg7: int  # latitude × 1e7
    lon_deg7: int  # longitude × 1e7
    alt_cm: int  # altitude cm
    battery_mv: int
    mode: int
    gps_fix: int
    speed_cms: int
    heading_cdeg: int

    FORMAT = "<iiHHBBHH"  # 20 bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "SensorCompact":
        lat, lon, alt, batt, mode, fix, spd, hdg = struct.unpack(cls.FORMAT, data)
        return cls(
            lat_deg7=lat,
            lon_deg7=lon,
            alt_cm=alt,
            battery_mv=batt,
            mode=mode,
            gps_fix=fix,
            speed_cms=spd,
            heading_cdeg=hdg,
        )

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.lat_deg7,
            self.lon_deg7,
            self.alt_cm,
            self.battery_mv,
            self.mode,
            self.gps_fix,
            self.speed_cms,
            self.heading_cdeg,
        )


# ── ActuatorCmd — generic (replaces VelocityCmd) ─────────────────────────────


@dataclass
class ActuatorCmd:
    """Generic actuator command for any robot type.

    actuator_type: 0=diff_drive, 1=quad_rotor, 2=humanoid, 3=ackermann
    channels:      list of i16 values (meaning depends on actuator_type)
      diff_drive:  [speed_l, speed_r]            (-100..100)
      quad_rotor:  [throttle, roll, pitch, yaw]  (PWM 1000-2000)
      humanoid:    [joint_0_cdeg, ..., joint_N]  (centidegrees)
      ackermann:   [speed, steer_angle_cdeg]
    flags: bit0=emergency_stop, bit1=alert
    """

    actuator_type: int
    channels: list[int]
    flags: int = 0

    def to_bytes(self) -> bytes:
        n = len(self.channels)
        return struct.pack(f"<BBB{n}h", self.actuator_type, n, self.flags, *self.channels)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ActuatorCmd":
        act_type, n, flags = struct.unpack_from("<BBB", data)
        channels = list(struct.unpack_from(f"<{n}h", data, 3))
        return cls(actuator_type=act_type, channels=channels, flags=flags)

    @classmethod
    def stop(cls, actuator_type: int = ACT_DIFF_DRIVE, n_channels: int = 2) -> "ActuatorCmd":
        """Emergency stop for any robot type."""
        return cls(actuator_type=actuator_type, channels=[0] * n_channels, flags=FLAG_EMERGENCY)

    @classmethod
    def wheeled(cls, speed_l: int, speed_r: int, flags: int = 0) -> "ActuatorCmd":
        """Convenience constructor for differential drive."""
        return cls(actuator_type=ACT_DIFF_DRIVE, channels=[speed_l, speed_r], flags=flags)

    @classmethod
    def drone(cls, throttle: int, roll: int, pitch: int, yaw: int, flags: int = 0) -> "ActuatorCmd":
        """Convenience constructor for quad rotor."""
        return cls(actuator_type=ACT_QUAD_ROTOR, channels=[throttle, roll, pitch, yaw], flags=flags)


# ── PredictCmd — RFC-0034 speculative actuation ──────────────────────────────


@dataclass
class PredictCmd:
    """The brain's predicted NEXT actuator command, plus a confidence byte.

    Sent under PKT_PREDICT so the kernel can act on it ahead of the confirmed
    command (gated by the kernel safety envelope + SPECULATIVE_ACTUATION, default
    off). Wire format mirrors the kernel `decode_predict_cmd`: the ActuatorCmd
    bytes followed by one trailing confidence byte (0..=255; 255 = certain, e.g.
    a deterministic scripted step).
    """

    cmd: ActuatorCmd
    confidence: int = 255  # 0..255

    def to_bytes(self) -> bytes:
        return self.cmd.to_bytes() + bytes([max(0, min(255, self.confidence))])

    @classmethod
    def from_bytes(cls, data: bytes) -> "PredictCmd":
        # Last byte is confidence; the rest is the ActuatorCmd.
        return cls(cmd=ActuatorCmd.from_bytes(data[:-1]), confidence=data[-1])


# ── VelocityCmd — kept for backward compatibility ────────────────────────────


@dataclass
class VelocityCmd:
    """Differential drive command (backward compat). Prefer ActuatorCmd.wheeled()."""

    speed_l: int
    speed_r: int
    flags: int = 0

    # Legacy wire format (i32 channels). The KERNEL only knows the ActuatorCmd
    # format (i16 channels) under PKT_ACTUATOR — sending raw `<iiB>` bytes would
    # silently corrupt the wire. `to_bytes()` therefore now emits the
    # ActuatorCmd encoding; the `_LEGACY_FORMAT` is kept only so `from_bytes`
    # can still decode any old recorded log frames.
    _LEGACY_FORMAT = "<iiB"  # 9 bytes — DO NOT use for new wire-level sends.

    def to_bytes(self) -> bytes:
        # Route through the ActuatorCmd encoding so any caller that still does
        # `vel.to_bytes()` (legacy path) produces a kernel-compatible frame.
        return self.to_actuator_cmd().to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "VelocityCmd":
        sl, sr, flags = struct.unpack(cls._LEGACY_FORMAT, data)
        return cls(speed_l=sl, speed_r=sr, flags=flags)

    def to_actuator_cmd(self) -> ActuatorCmd:
        return ActuatorCmd.wheeled(self.speed_l, self.speed_r, self.flags)


# ── ModeCmd (PKT_MODE = 0x81) ─────────────────────────────────────────────────
#
# Server -> Robot: operator mode switch, 1-byte payload = mode_id. `ESTOP_CMD`
# only ever ARMS (see EStopCmd above) — MODE_CMD is the ONLY rearm path on the
# wire, and it also clears degraded mode (RFC-0036) on the robot. Because of
# that, only ONE mode_id may do either: `MODE_ID_ESTOP_RESET`. Every other
# mode_id changes mode and must leave both containment states exactly as they
# were. Owner decision, 2026-09-06.
#
# Must match `MODE_ID_ESTOP_RESET` in AzOS `crates/behavior/src/
# brain_protocol.rs` — the two sides MUST agree, or a brain built against one
# and a robot built against the other disagree about which byte clears
# containment.
#
# Not 0: a short/truncated frame, or a payload buffer that was zeroed and
# never actually filled in, reads as `mode_id == 0` by default — 0 would let
# the single most common malformed-frame case rearm the robot. Not a small
# value like 1 or 2 either: those are the values most likely to be the first
# "real" mode ids an implementation assigns, so claiming one as the reset
# sentinel would collide with an actual operating mode the moment one is
# defined. 0xFF sits at the far end of the byte range, away from where
# sequential mode ids will be allocated, cannot arise from a zeroed/short
# payload, and matches this protocol's existing convention for a
# reserved/sentinel byte (see `DEGRADE_REASON_MALFORMED` = 0xFF on the robot
# side).
MODE_ID_ESTOP_RESET: int = 0xFF


def mode_clears_estop(mode_id: int) -> bool:
    """Whether a PKT_MODE `mode_id` is the reserved one that clears e-stop /
    degraded-mode containment on the robot.

    Mirrors `mode_clears_containment` in AzOS `brain_protocol.rs`: keep
    both in lockstep. Any code on this side that sends `MODE_CMD` intending
    to rearm the robot should check this first — sending anything else just
    changes mode and leaves the robot's containment state untouched.
    """
    return mode_id == MODE_ID_ESTOP_RESET


@dataclass
class ModeCmd:
    """Operator mode-switch command (1-byte payload).

    Only `mode_id == MODE_ID_ESTOP_RESET` clears an armed e-stop or exits
    degraded mode on the robot; every other mode_id just changes mode.
    """

    mode_id: int = MODE_ID_ESTOP_RESET

    FORMAT = "<B"  # 1 byte

    def to_bytes(self) -> bytes:
        return struct.pack(self.FORMAT, self.mode_id & 0xFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ModeCmd":
        (m,) = struct.unpack(cls.FORMAT, data[:1])
        return cls(mode_id=m)


# ── WaypointCmd (PKT_WAYPOINT = 0x82) ────────────────────────────────────────
#
# Server → Robot: a GPS waypoint to fly/drive to. Layout must byte-match the
# kernel's `decode_waypoint_cmd` in `crates/behavior/src/brain_protocol.rs`
# (14-byte payload, little-endian).

# Waypoint action codes (must match the kernel-side values).
WAYPOINT_ACT_GOTO: int = 0
WAYPOINT_ACT_LOITER: int = 1
WAYPOINT_ACT_LAND: int = 2
WAYPOINT_ACT_RTL: int = 3


@dataclass
class WaypointCmd:
    """GPS waypoint command. Kernel-compatible wire format."""

    lat_deg7: int  # latitude × 1e7 (i32)
    lon_deg7: int  # longitude × 1e7 (i32)
    alt_cm: int  # altitude in cm (u16, 0..65535)
    speed_cms: int  # target speed in cm/s (u16)
    action: int = WAYPOINT_ACT_GOTO  # u8
    flags: int = 0  # u8 (reserved / future)

    FORMAT = "<iiHHBB"  # 14 bytes, byte-identical to kernel

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.lat_deg7,
            self.lon_deg7,
            self.alt_cm,
            self.speed_cms,
            self.action,
            self.flags,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "WaypointCmd":
        lat, lon, alt, spd, act, flg = struct.unpack(cls.FORMAT, data[:14])
        return cls(lat_deg7=lat, lon_deg7=lon, alt_cm=alt, speed_cms=spd, action=act, flags=flg)


# ── EStopCmd (PKT_ESTOP = 0x88) ──────────────────────────────────────────────
#
# Server → Robot: emergency stop override-all-layers command. Single-byte
# payload carrying a reason code that the kernel logs and routes to the L0
# safety layer. Reason values must match `ESTOP_REASON_*` in `brain_protocol.rs`.

ESTOP_REASON_OPERATOR: int = 0  # human operator pressed the button
ESTOP_REASON_SAFETY: int = 1  # safety profile violation
ESTOP_REASON_GEOFENCE: int = 2  # geofence breach


@dataclass
class EStopCmd:
    """Emergency-stop command with reason code (1-byte payload)."""

    reason: int = ESTOP_REASON_OPERATOR

    FORMAT = "<B"  # 1 byte

    def to_bytes(self) -> bytes:
        return struct.pack(self.FORMAT, self.reason)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EStopCmd":
        (r,) = struct.unpack(cls.FORMAT, data[:1])
        return cls(reason=r)


# ── DegradeCmd (PKT_DEGRADE = 0x8A) — RFC-0036 ───────────────────────────────
#
# Server → Robot: arm/clear degraded mode. The brain arms it when it detects a
# situational hazard only it can perceive (most concretely: perception has gone
# blind). In degraded mode the kernel contains the userspace blast radius (every
# write/actuation through a user-task capability is denied at the cap chokepoint)
# while the in-kernel control loop keeps safe-stopping. 1-byte payload = reason;
# reason 0 clears. Values must match `DEGRADE_*` in `brain_protocol.rs`.

DEGRADE_CLEAR: int = 0  # exit degraded mode (brain recovered)
DEGRADE_REASON_PERCEPTION_BLIND: int = 1  # perception failed N cycles in a row
DEGRADE_REASON_SENSOR_INCOHERENT: int = 2  # fused sensor state inconsistent
DEGRADE_REASON_UNMODELLED_HAZARD: int = 3  # situational anomaly

# ── SemanticLevelCmd level indices (PKT_SEMANTIC_LEVEL = 0x8B) — RFC-0037 ─────
#
# Ordered restriction level sent as a single byte. Higher = more restrictive.
# Values must match `DEGRADE_LEVEL_*` consts in `crates/ipc/src/cap.rs` and
# `SAFETY_DEGRADE_*_CAP_PCT` in `crates/behavior/src/safety.rs`.
SEMANTIC_LEVEL_FULL: int = 0  # no extra limit — normal operation
SEMANTIC_LEVEL_CAUTIOUS: int = 1  # speed ceiling 70 % of per-type maximum
SEMANTIC_LEVEL_SLOW: int = 2  # speed ceiling 30 % of per-type maximum
SEMANTIC_LEVEL_CONTAINED: int = 3  # speed ceiling 0 % (stop) + cap-denial


@dataclass
class DegradeCmd:
    """Degraded-mode trigger with reason code (1-byte payload). reason 0 clears."""

    reason: int = DEGRADE_REASON_PERCEPTION_BLIND

    FORMAT = "<B"  # 1 byte

    def to_bytes(self) -> bytes:
        return struct.pack(self.FORMAT, self.reason & 0xFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DegradeCmd":
        (r,) = struct.unpack(cls.FORMAT, data[:1])
        return cls(reason=r)


# ── SemanticLevelCmd (PKT_SEMANTIC_LEVEL = 0x8B) — RFC-0037 ──────────────────
#
# Server → Robot: set the graded degrade level. 1-byte payload = level index
# (SEMANTIC_LEVEL_FULL … SEMANTIC_LEVEL_CONTAINED). The kernel clamps any
# out-of-range index to CONTAINED (fail-closed). The level is sticky; comms-loss
# safe-stop is handled by the motor watchdog (500 ms), not a TTL here — same
# pattern as DegradeCmd. Byte layout must match the kernel PKT_SEMANTIC_LEVEL
# handler in `kernel/src/main.rs` (both TCP and UART paths).


@dataclass
class SemanticLevelCmd:
    """Graded degrade-level command (1-byte payload). level 0 = FULL, 3 = CONTAINED."""

    level: int = SEMANTIC_LEVEL_FULL

    FORMAT = "<B"  # 1 byte — level index

    def to_bytes(self) -> bytes:
        return struct.pack(self.FORMAT, self.level & 0xFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SemanticLevelCmd":
        (lv,) = struct.unpack(cls.FORMAT, data[:1])
        return cls(level=lv)


# ── StatusPacket ──────────────────────────────────────────────────────────────


@dataclass
class StatusPacket:
    """Status packet from robot. Includes robot_type for protocol negotiation."""

    mode: int
    tasks_ok: int
    canary_ok: int
    uptime_s: int
    robot_type: int = ROBOT_WHEELED  # optional — legacy packets don't have it

    FORMAT_V1 = "<BBBI"  # 7 bytes (legacy, no robot_type): mode, tasks_ok, canary_ok, uptime_s
    FORMAT_V2 = "<BBBIB"  # 8 bytes: mode, tasks_ok, canary_ok, uptime_s, robot_type

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT_V2, self.mode, self.tasks_ok, self.canary_ok, self.uptime_s, self.robot_type
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "StatusPacket":
        if len(data) == 7:
            # Legacy format (no robot_type field)
            m, t, c, u = struct.unpack(cls.FORMAT_V1, data)
            return cls(mode=m, tasks_ok=t, canary_ok=c, uptime_s=u)
        m, t, c, u, rt = struct.unpack(cls.FORMAT_V2, data)
        return cls(mode=m, tasks_ok=t, canary_ok=c, uptime_s=u, robot_type=rt)


# ── ConfigCmd (LED + generic config) ─────────────────────────────────────────


@dataclass
class ConfigCmd:
    """Config command to robot. Key-value: config_key(1B) + value(1B) + reserved(2B).

    Used for LED state, power mode, and other runtime configuration.
    """

    config_key: int
    value: int
    reserved: int = 0

    FORMAT = "<BBH"  # 4 bytes: key, value, reserved_u16

    def to_bytes(self) -> bytes:
        return struct.pack(self.FORMAT, self.config_key, self.value, self.reserved)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConfigCmd":
        k, v, r = struct.unpack(cls.FORMAT, data[:4])
        return cls(config_key=k, value=v, reserved=r)

    @classmethod
    def led(cls, state: int) -> "ConfigCmd":
        """Create a LED state command."""
        return cls(config_key=LED_CONFIG_KEY, value=state)

    @classmethod
    def buzzer(cls, pattern: int) -> "ConfigCmd":
        """Create a buzzer command."""
        return cls(config_key=BUZZER_CONFIG_KEY, value=pattern)


# ── PayloadCmd (E04) — spray pump, gripper servo, camera trigger ──────────────

# Payload types (payload_type field)
PAYLOAD_TYPE_SPRAY = 0  # spray pump (GPIO on/off)
PAYLOAD_TYPE_GRIPPER = 1  # gripper servo (0=closed … 100=open)
PAYLOAD_TYPE_CAM_TRIGGER = 2  # external camera shutter trigger (GPIO pulse)

# Generic on/off values
PAYLOAD_OFF = 0
PAYLOAD_ON = 1

# Gripper position shortcuts
GRIPPER_OPEN = 100
GRIPPER_CLOSED = 0


@dataclass
class PayloadCmd:
    """E04: payload control command (spray, gripper, external camera trigger).

    Wire format (5 bytes): payload_type(1) + channel(1) + value(1) + duration_ms(2 LE)

    payload_type: PAYLOAD_TYPE_SPRAY / GRIPPER / CAM_TRIGGER
    channel:      device index (0 = first of that type)
    value:        0/1 for on/off; 0-100 for gripper position
    duration_ms:  0 = indefinite / one-shot; >0 = auto-off after N ms
    """

    payload_type: int
    channel: int = 0
    value: int = 0
    duration_ms: int = 0

    FORMAT = "<BBBH"  # 5 bytes

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT, self.payload_type, self.channel, self.value, self.duration_ms
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PayloadCmd":
        pt, ch, val, dur = struct.unpack(cls.FORMAT, data[:5])
        return cls(payload_type=pt, channel=ch, value=val, duration_ms=dur)

    @classmethod
    def spray(cls, on: bool, channel: int = 0, duration_ms: int = 0) -> "PayloadCmd":
        """Turn spray pump on or off."""
        return cls(
            payload_type=PAYLOAD_TYPE_SPRAY,
            channel=channel,
            value=PAYLOAD_ON if on else PAYLOAD_OFF,
            duration_ms=duration_ms,
        )

    @classmethod
    def gripper(cls, pos: int, channel: int = 0) -> "PayloadCmd":
        """Set gripper position (0=closed, 100=open)."""
        return cls(payload_type=PAYLOAD_TYPE_GRIPPER, channel=channel, value=max(0, min(100, pos)))

    @classmethod
    def cam_trigger(cls, channel: int = 0) -> "PayloadCmd":
        """Fire external camera shutter trigger pulse."""
        return cls(payload_type=PAYLOAD_TYPE_CAM_TRIGGER, channel=channel, value=1)
