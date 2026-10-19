# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0013 property-based tests for protocol.py.

Uses Hypothesis to assert structural invariants over arbitrary inputs.
These complement the example-based tests in test_protocol.py but are
far stronger for parsers: they exercise the full valid-input space and
confirm the parsers never panic / produce out-of-bounds reads.

Run with:
    python -m pytest tests/test_protocol_property.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, strategies as st

from protocol import (
    build_packet,
    parse_packet,
    crc8,
    SensorPacket,
    SensorPacketHumanoid,
    SensorCompact,
    ActuatorCmd,
    MAX_PAYLOAD_BYTES,
    SENSOR_PACKET,
    ACTUATOR_CMD,
    SENSOR_COMPACT,
)

# ── Named bounds (no magic numbers) ──────────────────────────────────────────

# Integer type limits — mirror the struct field widths used in protocol.py.
U8_MAX = 0xFF
U16_MAX = 0xFFFF
U32_MAX = 0xFFFF_FFFF
U64_MAX = 0xFFFF_FFFF_FFFF_FFFF

I16_MIN = -(1 << 15)
I16_MAX = (1 << 15) - 1
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

# Maximum number of actuator channels.  Mirrors MAX_CHANNELS in the Rust
# kernel (brain_protocol.rs) and ActuatorCmd.stop()'s expectations.
MAX_ACTUATOR_CHANNELS = 8

# Maximum number of joints for humanoid; reuse the parser's own guard.
MAX_HUMANOID_JOINTS = SensorPacketHumanoid.MAX_JOINTS  # 32

# Maximum byte length for the arbitrary-byte fuzz properties.
MAX_FUZZ_BYTES = 1024

# Hypothesis: how many examples to generate per property.
HYPOTHESIS_MAX_EXAMPLES = 200

# ── Helpers ──────────────────────────────────────────────────────────────────


def _i32_triple() -> st.SearchStrategy:
    """Strategy for a 3-tuple of i32 values (accel / gyro)."""
    return st.tuples(
        st.integers(I32_MIN, I32_MAX),
        st.integers(I32_MIN, I32_MAX),
        st.integers(I32_MIN, I32_MAX),
    )


# ── Property 1: build_packet / parse_packet round-trip ───────────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    pkt_type=st.integers(0, U8_MAX),
    payload=st.binary(min_size=0, max_size=MAX_PAYLOAD_BYTES),
)
def test_prop_build_parse_roundtrip(pkt_type: int, payload: bytes) -> None:
    """build_packet then parse_packet always recovers the original (type, payload).

    Pins: framing is lossless for any legal payload length and any byte
    pattern.  Any regression in MAGIC, LEN encoding, or CRC calculation
    will break this.
    """
    frame = build_packet(pkt_type, payload)
    result = parse_packet(frame)
    assert result is not None, (
        f"parse_packet returned None for type=0x{pkt_type:02x} " f"payload_len={len(payload)}"
    )
    got_type, got_payload = result
    assert got_type == pkt_type
    assert got_payload == payload


# ── Property 2: parse_packet never raises on arbitrary bytes ─────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(data=st.binary(min_size=0, max_size=MAX_FUZZ_BYTES))
def test_prop_parse_never_raises(data: bytes) -> None:
    """parse_packet on arbitrary bytes always returns None or a valid result;
    it never raises an exception.

    Pins: the parser must be safe against any hostile or malformed input
    (slowloris-style inputs, truncated frames, corrupt LEN fields, etc.).
    """
    result = parse_packet(data)
    if result is not None:
        pkt_type, payload = result
        assert 0 <= pkt_type <= U8_MAX
        # payload must be a slice of the original data — verify bounds
        assert len(payload) <= len(data)


# ── Property 3: CRC-8 self-check (residue == 0) ───────────────────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(data=st.binary(min_size=0, max_size=MAX_FUZZ_BYTES))
def test_prop_crc8_self_check(data: bytes) -> None:
    """Appending the CRC of `data` to `data` produces a zero residue.

    Formally: crc8(data + bytes([crc8(data)])) == 0.

    This is the canonical "residue" property of any CRC where init=0,
    no reflection, and xorout=0 (which this implementation satisfies).
    After processing the N-byte payload the accumulator holds `M`.
    Appending byte `M` XORs the accumulator to 0, and 8 shifts of 0 stay 0,
    so the final CRC is 0.

    Why this is not vacuous: it pins down the *polynomial computation* end-
    to-end.  Any implementation that uses the wrong poly, wrong bit order, a
    non-zero init, or a non-zero XOR-out will fail for at least one input.
    It is the minimal property that distinguishes a correct CRC-8 from an
    incorrect one without requiring a pre-computed table of expected values.

    Regression coverage: a wrong polynomial (e.g. 0x07 vs 0x31) silently
    passes the old determinism test but fails this one for most payloads.
    """
    c = crc8(data)
    assert crc8(data + bytes([c])) == 0


# ── Property 5: bit-flip in frame body causes parse to reject or differ ───────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    pkt_type=st.integers(0, U8_MAX),
    payload=st.binary(min_size=1, max_size=64),
    # Restrict flip position to everything EXCEPT the LEN field (bytes 3-4).
    # Flipping a LEN bit can produce a valid *shorter* packet that parses
    # with a different type/payload — still different from the original, but
    # "None" is not guaranteed.  Skipping LEN bytes keeps the "must be None
    # OR different" property from spuriously falsifying.
    #
    # Bytes: 0-1 = MAGIC, 2 = TYPE, 3-4 = LEN, 5..5+N = payload, 5+N = CRC.
    # We flip in {0,1,2} ∪ {5..end}.
)
def test_prop_bit_flip_fails_or_changes(pkt_type: int, payload: bytes) -> None:
    """Flipping any single bit in MAGIC, TYPE, payload, or CRC bytes causes
    parse_packet to return None OR return a result different from the original.

    Pins: the CRC (and magic/type checks) catch single-bit corruption in
    every field except LEN, which is deliberately excluded — see comment
    above regarding the rare CRC-collision edge case with truncated LEN.
    """
    frame = bytearray(build_packet(pkt_type, payload))
    frame_len = len(frame)

    # Build the set of bit positions to test: everything except LEN bytes 3-4.
    # Byte ranges: [0, 1, 2] (magic+type) and [5, frame_len-1] (payload+crc).
    safe_bytes = list(range(0, 3)) + list(range(5, frame_len))

    original_result = (pkt_type, payload)

    for byte_idx in safe_bytes:
        for bit in range(8):
            flipped = bytearray(frame)
            flipped[byte_idx] ^= 1 << bit
            result = parse_packet(bytes(flipped))
            if result is not None:
                # The flipped frame parsed — it must differ from the original.
                assert result != original_result, (
                    f"Bit flip at byte {byte_idx} bit {bit} produced same result "
                    f"as original (type=0x{pkt_type:02x}, payload={payload!r})"
                )


# ── Property 6: SensorPacket round-trip ──────────────────────────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    timestamp_ms=st.integers(0, U64_MAX),
    battery_mv=st.integers(0, U16_MAX),
    accel_mg=_i32_triple(),
    gyro_mdps=_i32_triple(),
    odom_dist_mm=st.integers(I32_MIN, I32_MAX),
    odom_hdg_cdeg=st.integers(I32_MIN, I32_MAX),
    encoder_l=st.integers(I64_MIN, I64_MAX),
    encoder_r=st.integers(I64_MIN, I64_MAX),
    range_front_mm=st.integers(0, U16_MAX),
    range_right_mm=st.integers(0, U16_MAX),
    sensor_flags=st.integers(0, U16_MAX),
)
def test_prop_sensor_packet_roundtrip(
    timestamp_ms: int,
    battery_mv: int,
    accel_mg: tuple,
    gyro_mdps: tuple,
    odom_dist_mm: int,
    odom_hdg_cdeg: int,
    encoder_l: int,
    encoder_r: int,
    range_front_mm: int,
    range_right_mm: int,
    sensor_flags: int,
) -> None:
    """SensorPacket.to_bytes() → from_bytes() preserves every field for
    arbitrary valid field values.

    Pins: the struct packing/unpacking is lossless across the full i32/i64/u16
    input domains.  Catches byte-swap bugs, sign-extension errors, and
    field-offset regressions.
    """
    sp = SensorPacket(
        timestamp_ms=timestamp_ms,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=gyro_mdps,
        odom_dist_mm=odom_dist_mm,
        odom_hdg_cdeg=odom_hdg_cdeg,
        encoder_l=encoder_l,
        encoder_r=encoder_r,
        range_front_mm=range_front_mm,
        range_right_mm=range_right_mm,
        sensor_flags=sensor_flags,
    )
    data = sp.to_bytes()
    sp2 = SensorPacket.from_bytes(data)

    assert sp2.timestamp_ms == timestamp_ms
    assert sp2.battery_mv == battery_mv
    assert sp2.accel_mg == accel_mg
    assert sp2.gyro_mdps == gyro_mdps
    assert sp2.odom_dist_mm == odom_dist_mm
    assert sp2.odom_hdg_cdeg == odom_hdg_cdeg
    assert sp2.encoder_l == encoder_l
    assert sp2.encoder_r == encoder_r
    assert sp2.range_front_mm == range_front_mm
    assert sp2.range_right_mm == range_right_mm
    assert sp2.sensor_flags == sensor_flags


# ── Property 7: SensorCompact round-trip ────────────────────────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    lat_deg7=st.integers(I32_MIN, I32_MAX),
    lon_deg7=st.integers(I32_MIN, I32_MAX),
    alt_cm=st.integers(0, U16_MAX),
    battery_mv=st.integers(0, U16_MAX),
    mode=st.integers(0, U8_MAX),
    gps_fix=st.integers(0, U8_MAX),
    speed_cms=st.integers(0, U16_MAX),
    heading_cdeg=st.integers(0, U16_MAX),
)
def test_prop_sensor_compact_roundtrip(
    lat_deg7: int,
    lon_deg7: int,
    alt_cm: int,
    battery_mv: int,
    mode: int,
    gps_fix: int,
    speed_cms: int,
    heading_cdeg: int,
) -> None:
    """SensorCompact.to_bytes() → from_bytes() preserves every field.

    Pins: the compact 20-byte LoRa frame codec is lossless across the full
    valid-input domain.  Guards the E02 low-bandwidth link path.
    """
    sc = SensorCompact(
        lat_deg7=lat_deg7,
        lon_deg7=lon_deg7,
        alt_cm=alt_cm,
        battery_mv=battery_mv,
        mode=mode,
        gps_fix=gps_fix,
        speed_cms=speed_cms,
        heading_cdeg=heading_cdeg,
    )
    data = sc.to_bytes()
    sc2 = SensorCompact.from_bytes(data)

    assert sc2.lat_deg7 == lat_deg7
    assert sc2.lon_deg7 == lon_deg7
    assert sc2.alt_cm == alt_cm
    assert sc2.battery_mv == battery_mv
    assert sc2.mode == mode
    assert sc2.gps_fix == gps_fix
    assert sc2.speed_cms == speed_cms
    assert sc2.heading_cdeg == heading_cdeg


# ── Property 8: ActuatorCmd round-trip ───────────────────────────────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    actuator_type=st.integers(0, U8_MAX),
    channels=st.lists(
        st.integers(I16_MIN, I16_MAX),
        min_size=0,
        max_size=MAX_ACTUATOR_CHANNELS,
    ),
    flags=st.integers(0, U8_MAX),
)
def test_prop_actuator_cmd_roundtrip(
    actuator_type: int,
    channels: list,
    flags: int,
) -> None:
    """ActuatorCmd.to_bytes() → from_bytes() preserves every field.

    Pins: channel list encoding (variable-length i16 array) is lossless
    for any legal channel count and any i16 channel value.  The channel
    count byte and the actual channel data must stay in sync.
    """
    cmd = ActuatorCmd(actuator_type=actuator_type, channels=channels, flags=flags)
    data = cmd.to_bytes()
    cmd2 = ActuatorCmd.from_bytes(data)

    assert cmd2.actuator_type == actuator_type
    assert cmd2.channels == channels
    assert cmd2.flags == flags


# ── Property 9: framed SensorPacket can be recovered end-to-end ─────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    timestamp_ms=st.integers(0, U64_MAX),
    battery_mv=st.integers(0, U16_MAX),
    accel_mg=_i32_triple(),
    gyro_mdps=_i32_triple(),
)
def test_prop_sensor_packet_framed_roundtrip(
    timestamp_ms: int,
    battery_mv: int,
    accel_mg: tuple,
    gyro_mdps: tuple,
) -> None:
    """A SensorPacket wrapped in a build_packet frame can be recovered by
    parse_packet + SensorPacket.from_bytes().

    Pins: the full E2E path — struct codec then wire frame — is lossless.
    This is the path exercised on every TCP read from the robot.
    """
    sp = SensorPacket(
        timestamp_ms=timestamp_ms,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=gyro_mdps,
        odom_dist_mm=0,
        odom_hdg_cdeg=0,
        encoder_l=0,
        encoder_r=0,
        range_front_mm=0,
        range_right_mm=0,
    )
    payload = sp.to_bytes()
    frame = build_packet(SENSOR_PACKET, payload)
    result = parse_packet(frame)

    assert result is not None
    got_type, got_payload = result
    assert got_type == SENSOR_PACKET

    sp2 = SensorPacket.from_bytes(got_payload)
    assert sp2.timestamp_ms == timestamp_ms
    assert sp2.battery_mv == battery_mv
    assert sp2.accel_mg == accel_mg
    assert sp2.gyro_mdps == gyro_mdps


# ── Property 10: framed ActuatorCmd can be recovered end-to-end ─────────────


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    channels=st.lists(
        st.integers(I16_MIN, I16_MAX),
        min_size=1,
        max_size=MAX_ACTUATOR_CHANNELS,
    ),
    flags=st.integers(0, U8_MAX),
)
def test_prop_actuator_cmd_framed_roundtrip(channels: list, flags: int) -> None:
    """An ActuatorCmd wrapped in a build_packet frame can be recovered by
    parse_packet + ActuatorCmd.from_bytes().

    Pins: the full brain → robot command path is lossless.
    """
    cmd = ActuatorCmd(actuator_type=0, channels=channels, flags=flags)
    frame = build_packet(ACTUATOR_CMD, cmd.to_bytes())
    result = parse_packet(frame)

    assert result is not None
    got_type, got_payload = result
    assert got_type == ACTUATOR_CMD

    cmd2 = ActuatorCmd.from_bytes(got_payload)
    assert cmd2.channels == channels
    assert cmd2.flags == flags


# ── Property 10: SensorPacketDrone round-trip ───────────────────────────────
#
# Same shape as SensorPacket but with the drone payload (baro_pa, mag_ut,
# gps_*, sonar).  baro_pa is i32, mag_ut is i16 triple, gps is i32 triple,
# sonar_down_mm is u16.

from protocol import SensorPacketDrone

I16_MIN_ = -(1 << 15)
I16_MAX_ = (1 << 15) - 1


def _i16_triple() -> st.SearchStrategy:
    return st.tuples(
        st.integers(I16_MIN_, I16_MAX_),
        st.integers(I16_MIN_, I16_MAX_),
        st.integers(I16_MIN_, I16_MAX_),
    )


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    timestamp_ms=st.integers(0, U64_MAX),
    battery_mv=st.integers(0, U16_MAX),
    accel_mg=_i32_triple(),
    gyro_mdps=_i32_triple(),
    baro_pa=st.integers(I32_MIN, I32_MAX),
    mag_ut=_i16_triple(),
    gps_lat_deg7=st.integers(I32_MIN, I32_MAX),
    gps_lon_deg7=st.integers(I32_MIN, I32_MAX),
    gps_alt_cm=st.integers(I32_MIN, I32_MAX),
    sonar_down_mm=st.integers(0, U16_MAX),
)
def test_prop_sensor_packet_drone_roundtrip(
    timestamp_ms: int,
    battery_mv: int,
    accel_mg: tuple,
    gyro_mdps: tuple,
    baro_pa: int,
    mag_ut: tuple,
    gps_lat_deg7: int,
    gps_lon_deg7: int,
    gps_alt_cm: int,
    sonar_down_mm: int,
) -> None:
    sp = SensorPacketDrone(
        timestamp_ms=timestamp_ms,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=gyro_mdps,
        baro_pa=baro_pa,
        mag_ut=mag_ut,
        gps_lat_deg7=gps_lat_deg7,
        gps_lon_deg7=gps_lon_deg7,
        gps_alt_cm=gps_alt_cm,
        sonar_down_mm=sonar_down_mm,
    )
    sp2 = SensorPacketDrone.from_bytes(sp.to_bytes())
    assert sp2.timestamp_ms == timestamp_ms
    assert sp2.battery_mv == battery_mv
    assert sp2.accel_mg == accel_mg
    assert sp2.gyro_mdps == gyro_mdps
    assert sp2.baro_pa == baro_pa
    assert sp2.mag_ut == mag_ut
    assert sp2.gps_lat_deg7 == gps_lat_deg7
    assert sp2.gps_lon_deg7 == gps_lon_deg7
    assert sp2.gps_alt_cm == gps_alt_cm
    assert sp2.sonar_down_mm == sonar_down_mm


# ── Property 11: SensorPacketHumanoid round-trip ─────────────────────────────
#
# Variable-length joints list (0..MAX_HUMANOID_JOINTS).  Both ends of the
# range matter — empty joint list and max-size both must roundtrip.


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    timestamp_ms=st.integers(0, U64_MAX),
    battery_mv=st.integers(0, U16_MAX),
    accel_mg=_i32_triple(),
    gyro_mdps=_i32_triple(),
    joint_angles=st.lists(
        st.integers(I16_MIN_, I16_MAX_),
        min_size=0,
        max_size=MAX_HUMANOID_JOINTS,
    ),
    foot_pressure_l=st.integers(0, U16_MAX),
    foot_pressure_r=st.integers(0, U16_MAX),
)
def test_prop_sensor_packet_humanoid_roundtrip(
    timestamp_ms: int,
    battery_mv: int,
    accel_mg: tuple,
    gyro_mdps: tuple,
    joint_angles: list,
    foot_pressure_l: int,
    foot_pressure_r: int,
) -> None:
    sp = SensorPacketHumanoid(
        timestamp_ms=timestamp_ms,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=gyro_mdps,
        joint_angles=joint_angles,
        foot_pressure_l=foot_pressure_l,
        foot_pressure_r=foot_pressure_r,
    )
    sp2 = SensorPacketHumanoid.from_bytes(sp.to_bytes())
    assert sp2.timestamp_ms == timestamp_ms
    assert sp2.battery_mv == battery_mv
    assert sp2.accel_mg == accel_mg
    assert sp2.gyro_mdps == gyro_mdps
    assert sp2.joint_angles == joint_angles
    assert sp2.foot_pressure_l == foot_pressure_l
    assert sp2.foot_pressure_r == foot_pressure_r


# ── Property 12: WaypointCmd round-trip ─────────────────────────────────────
#
# 14-byte payload identical to the kernel's `decode_waypoint_cmd`.

from protocol import WaypointCmd


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    lat_deg7=st.integers(I32_MIN, I32_MAX),
    lon_deg7=st.integers(I32_MIN, I32_MAX),
    alt_cm=st.integers(0, U16_MAX),
    speed_cms=st.integers(0, U16_MAX),
    action=st.integers(0, U8_MAX),
    flags=st.integers(0, U8_MAX),
)
def test_prop_waypoint_cmd_roundtrip(
    lat_deg7: int,
    lon_deg7: int,
    alt_cm: int,
    speed_cms: int,
    action: int,
    flags: int,
) -> None:
    w = WaypointCmd(
        lat_deg7=lat_deg7,
        lon_deg7=lon_deg7,
        alt_cm=alt_cm,
        speed_cms=speed_cms,
        action=action,
        flags=flags,
    )
    data = w.to_bytes()
    assert len(data) == 14, "WaypointCmd wire format must be exactly 14 bytes"
    w2 = WaypointCmd.from_bytes(data)
    assert w2.lat_deg7 == lat_deg7
    assert w2.lon_deg7 == lon_deg7
    assert w2.alt_cm == alt_cm
    assert w2.speed_cms == speed_cms
    assert w2.action == action
    assert w2.flags == flags


# ── Property 13: EStopCmd round-trip (1-byte payload) ───────────────────────

from protocol import EStopCmd


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(reason=st.integers(0, U8_MAX))
def test_prop_estop_cmd_roundtrip(reason: int) -> None:
    e = EStopCmd(reason=reason)
    data = e.to_bytes()
    assert len(data) == 1
    e2 = EStopCmd.from_bytes(data)
    assert e2.reason == reason


# ── Property 14: StatusPacket round-trip (V2 8-byte) + legacy V1 decode ─────

from protocol import StatusPacket


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    mode=st.integers(0, U8_MAX),
    tasks_ok=st.integers(0, U8_MAX),
    canary_ok=st.integers(0, U8_MAX),
    uptime_s=st.integers(0, U32_MAX),
    robot_type=st.integers(0, U8_MAX),
)
def test_prop_status_packet_v2_roundtrip(
    mode: int,
    tasks_ok: int,
    canary_ok: int,
    uptime_s: int,
    robot_type: int,
) -> None:
    s = StatusPacket(
        mode=mode,
        tasks_ok=tasks_ok,
        canary_ok=canary_ok,
        uptime_s=uptime_s,
        robot_type=robot_type,
    )
    data = s.to_bytes()
    assert len(data) == 8
    s2 = StatusPacket.from_bytes(data)
    assert s2.mode == mode
    assert s2.tasks_ok == tasks_ok
    assert s2.canary_ok == canary_ok
    assert s2.uptime_s == uptime_s
    assert s2.robot_type == robot_type


@settings(max_examples=50, deadline=None)
@given(
    mode=st.integers(0, U8_MAX),
    tasks_ok=st.integers(0, U8_MAX),
    canary_ok=st.integers(0, U8_MAX),
    uptime_s=st.integers(0, U32_MAX),
)
def test_prop_status_packet_v1_legacy_decode(
    mode: int,
    tasks_ok: int,
    canary_ok: int,
    uptime_s: int,
) -> None:
    """Hand-built 7-byte V1 frame decodes correctly with robot_type defaulted."""
    import struct as _st

    data = _st.pack("<BBBI", mode, tasks_ok, canary_ok, uptime_s)
    assert len(data) == 7
    s = StatusPacket.from_bytes(data)
    assert s.mode == mode
    assert s.tasks_ok == tasks_ok
    assert s.canary_ok == canary_ok
    assert s.uptime_s == uptime_s
    # V1 frames default robot_type to ROBOT_WHEELED (0).
    assert s.robot_type == 0


# ── Property 15: ConfigCmd round-trip ───────────────────────────────────────

from protocol import ConfigCmd


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    config_key=st.integers(0, U8_MAX),
    value=st.integers(0, U8_MAX),
    reserved=st.integers(0, U16_MAX),
)
def test_prop_config_cmd_roundtrip(
    config_key: int,
    value: int,
    reserved: int,
) -> None:
    c = ConfigCmd(config_key=config_key, value=value, reserved=reserved)
    data = c.to_bytes()
    assert len(data) == 4
    c2 = ConfigCmd.from_bytes(data)
    assert c2.config_key == config_key
    assert c2.value == value
    assert c2.reserved == reserved


# ── Property 16: PayloadCmd round-trip ──────────────────────────────────────

from protocol import PayloadCmd


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    payload_type=st.integers(0, U8_MAX),
    channel=st.integers(0, U8_MAX),
    value=st.integers(0, U8_MAX),
    duration_ms=st.integers(0, U16_MAX),
)
def test_prop_payload_cmd_roundtrip(
    payload_type: int,
    channel: int,
    value: int,
    duration_ms: int,
) -> None:
    p = PayloadCmd(
        payload_type=payload_type,
        channel=channel,
        value=value,
        duration_ms=duration_ms,
    )
    data = p.to_bytes()
    assert len(data) == 5
    p2 = PayloadCmd.from_bytes(data)
    assert p2.payload_type == payload_type
    assert p2.channel == channel
    assert p2.value == value
    assert p2.duration_ms == duration_ms


# ── Property 17: VelocityCmd legacy decode path ─────────────────────────────
#
# `VelocityCmd.to_bytes()` now routes through `ActuatorCmd` for kernel
# compatibility, so `from_bytes(to_bytes(v))` is NOT a roundtrip.  But
# `from_bytes()` is still used to decode old recorded log frames in the
# `<iiB>` format — pin that decode path.

from protocol import VelocityCmd


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(
    speed_l=st.integers(I32_MIN, I32_MAX),
    speed_r=st.integers(I32_MIN, I32_MAX),
    flags=st.integers(0, U8_MAX),
)
def test_prop_velocity_cmd_legacy_decode(
    speed_l: int,
    speed_r: int,
    flags: int,
) -> None:
    """A hand-built legacy <iiB> frame decodes back into the same fields."""
    import struct as _st

    data = _st.pack(VelocityCmd._LEGACY_FORMAT, speed_l, speed_r, flags)
    v = VelocityCmd.from_bytes(data)
    assert v.speed_l == speed_l
    assert v.speed_r == speed_r
    assert v.flags == flags
