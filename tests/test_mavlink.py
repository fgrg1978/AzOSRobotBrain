# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for mavlink_client.py — E08 MAVLink bridge.

Uses a fake in-memory transport (no real socket / SITL required) so the
suite runs in CI. Covers:

  * CRC and frame encode/decode round-trip
  * COMMAND_LONG encode → decode → field-by-field parse
  * Telemetry parsers (heartbeat, attitude, position, battery)
  * Skill → MAVLink command translation
  * High-level client commands send the right wire bytes
  * Failsafe logic (battery RTL / LAND, brain-disconnect RTL / DISARM)
"""

import asyncio
import math
import struct

import pytest

from mavlink_client import (
    # Wire constants
    MAVLINK_V1_STX,
    MAVLINK_SYSTEM_ID,
    MAVLINK_TARGET_SYSTEM,
    MAVLINK_TARGET_COMPONENT,
    MAVLINK_MSG_ID_HEARTBEAT,
    MAVLINK_MSG_ID_COMMAND_LONG,
    MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    MAVLINK_MSG_ID_ATTITUDE,
    MAVLINK_MSG_ID_SYS_STATUS,
    MAVLINK_MSG_ID_BATTERY_STATUS,
    MAVLINK_PAYLOAD_LEN,
    MAVLINK_HEADER_LEN,
    # Command IDs
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    MAV_CMD_NAV_WAYPOINT,
    MAV_CMD_COMPONENT_ARM_DISARM,
    ARM_DISARM_ARM,
    ARM_DISARM_DISARM,
    # Timings
    HEARTBEAT_INTERVAL_S,
    TELEMETRY_TIMEOUT_S,
    BATTERY_RTL_PCT,
    BATTERY_LAND_PCT,
    DISARM_ON_GROUND_ALT_M,
    DEFAULT_UDP_URL,
    # Types
    MavTelemetry,
    MavlinkFrame,
    MAVLinkClient,
    # Codec
    mavlink_crc_accumulate,
    encode_mavlink_v1,
    decode_mavlink_v1,
    encode_heartbeat,
    encode_command_long,
    parse_heartbeat,
    parse_global_position_int,
    parse_attitude,
    parse_sys_status,
    parse_battery_status,
    parse_gps_raw_int,
    parse_connection_string,
    translate_skill,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeTransport:
    """In-memory transport with controllable rx/tx queues."""

    def __init__(self):
        self.sent: list[bytes] = []
        self.rx_queue: list[bytes] = []
        self.closed = False

    def send(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    def recv(self, maxlen: int = 4096) -> bytes:
        if not self.rx_queue:
            return b""
        return self.rx_queue.pop(0)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------
class TestConstants:
    def test_command_ids(self):
        assert MAV_CMD_NAV_TAKEOFF == 22
        assert MAV_CMD_NAV_LAND == 21
        assert MAV_CMD_NAV_RETURN_TO_LAUNCH == 20
        assert MAV_CMD_NAV_WAYPOINT == 16
        assert MAV_CMD_COMPONENT_ARM_DISARM == 400

    def test_message_ids(self):
        assert MAVLINK_MSG_ID_HEARTBEAT == 0
        assert MAVLINK_MSG_ID_COMMAND_LONG == 76
        assert MAVLINK_MSG_ID_GLOBAL_POSITION_INT == 33

    def test_system_ids(self):
        assert MAVLINK_SYSTEM_ID == 255
        assert MAVLINK_TARGET_SYSTEM == 1
        assert MAVLINK_TARGET_COMPONENT == 1

    def test_timeouts_positive(self):
        assert HEARTBEAT_INTERVAL_S > 0
        assert TELEMETRY_TIMEOUT_S > 0

    def test_failsafe_thresholds(self):
        assert 0 < BATTERY_LAND_PCT < BATTERY_RTL_PCT < 100
        assert DISARM_ON_GROUND_ALT_M > 0

    def test_default_udp_url(self):
        assert DEFAULT_UDP_URL.startswith("udp://")
        assert "14540" in DEFAULT_UDP_URL  # PX4 SITL GCS port


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------
class TestCrc:
    def test_init_value(self):
        # Empty buffer returns the initial seed.
        assert mavlink_crc_accumulate(b"") == 0xFFFF

    def test_deterministic(self):
        a = mavlink_crc_accumulate(b"AzOSRobotBrain")
        b = mavlink_crc_accumulate(b"AzOSRobotBrain")
        assert a == b

    def test_diff_on_diff_input(self):
        a = mavlink_crc_accumulate(b"abc")
        b = mavlink_crc_accumulate(b"abd")
        assert a != b

    def test_chained_matches_monolithic(self):
        mono = mavlink_crc_accumulate(b"hello world")
        partial = mavlink_crc_accumulate(b"hello ")
        chained = mavlink_crc_accumulate(b"world", partial)
        assert mono == chained


# ---------------------------------------------------------------------------
# Frame encode / decode
# ---------------------------------------------------------------------------
class TestFrameCodec:
    def test_heartbeat_roundtrip(self):
        frame = encode_heartbeat(seq=7)
        assert frame[0] == MAVLINK_V1_STX
        assert frame[1] == MAVLINK_PAYLOAD_LEN[MAVLINK_MSG_ID_HEARTBEAT]
        assert frame[2] == 7  # seq
        decoded = decode_mavlink_v1(frame)
        assert decoded is not None
        assert decoded.msgid == MAVLINK_MSG_ID_HEARTBEAT
        assert decoded.seq == 7
        hb = parse_heartbeat(decoded.payload)
        # GCS should not be "armed"
        assert hb["armed"] is False

    def test_command_long_roundtrip(self):
        frame = encode_command_long(
            MAV_CMD_NAV_TAKEOFF,
            param7=15.5,
            seq=42,
        )
        decoded = decode_mavlink_v1(frame)
        assert decoded is not None
        assert decoded.msgid == MAVLINK_MSG_ID_COMMAND_LONG
        assert decoded.seq == 42
        # param1..7(7*f32) + command(u16) + target_sys + target_comp + confirm
        p1, p2, p3, p4, p5, p6, p7, cmd, tsys, tcomp, conf = struct.unpack(
            "<fffffffHBBB",
            decoded.payload[:33],
        )
        assert cmd == MAV_CMD_NAV_TAKEOFF
        assert tsys == MAVLINK_TARGET_SYSTEM
        assert tcomp == MAVLINK_TARGET_COMPONENT
        assert conf == 0
        assert math.isclose(p7, 15.5, rel_tol=1e-5)

    def test_bad_crc_rejected(self):
        frame = bytearray(encode_heartbeat(seq=0))
        frame[-1] ^= 0xFF  # flip CRC byte
        assert decode_mavlink_v1(bytes(frame)) is None

    def test_unknown_stx_rejected(self):
        frame = bytearray(encode_heartbeat(seq=0))
        frame[0] = 0x00
        assert decode_mavlink_v1(bytes(frame)) is None

    def test_short_frame_returns_none(self):
        assert decode_mavlink_v1(b"") is None
        assert decode_mavlink_v1(b"\xfe\x09") is None  # header only


# ---------------------------------------------------------------------------
# Telemetry parsers
# ---------------------------------------------------------------------------
class TestTelemetryParsers:
    def _build_frame(self, msgid, payload):
        # Build a valid v1 frame for the decoder so we exercise real bytes.
        return encode_mavlink_v1(msgid, payload, seq=0)

    def test_global_position_int(self):
        payload = struct.pack(
            "<IiiiihhhH",
            123456,  # time_boot_ms
            int(40.1234567 * 1e7),  # lat
            int(-3.9876543 * 1e7),  # lon
            55_000,  # alt (mm) absolute
            12_500,  # relative_alt (mm) = 12.5 m
            100,
            -50,
            25,  # vx, vy, vz cm/s
            9000,  # hdg centideg = 90 deg
        )
        d = parse_global_position_int(payload)
        assert math.isclose(d["lat_deg"], 40.1234567, rel_tol=1e-6)
        assert math.isclose(d["lon_deg"], -3.9876543, rel_tol=1e-6)
        assert math.isclose(d["relative_alt_m"], 12.5, rel_tol=1e-6)
        assert math.isclose(d["heading_deg"], 90.0, rel_tol=1e-6)

    def test_attitude(self):
        payload = struct.pack(
            "<Iffffff",
            1000,
            0.1,
            -0.2,
            1.5,  # roll / pitch / yaw (rad)
            0.0,
            0.0,
            0.0,
        )
        d = parse_attitude(payload)
        assert math.isclose(d["roll_deg"], math.degrees(0.1), rel_tol=1e-4)
        assert math.isclose(d["pitch_deg"], math.degrees(-0.2), rel_tol=1e-4)
        assert math.isclose(d["yaw_deg"], math.degrees(1.5), rel_tol=1e-4)

    def test_sys_status(self):
        # Full 31B SYS_STATUS layout per MAVLink v1 spec.
        payload = struct.pack(
            "<IIIHHhHHHHHHb",
            0,
            0,
            0,  # sensors present/enabled/health
            500,  # load 0.5%
            11_800,  # voltage_battery mV
            -1,  # current_battery (cA)
            0,  # drop_rate_comm
            0,  # errors_comm
            0,
            0,
            0,
            0,  # errors_count 1..4
            87,  # battery_remaining %
        )
        d = parse_sys_status(payload)
        assert d["voltage_mv"] == 11_800
        assert d["battery_remaining_pct"] == 87

    def test_battery_status(self):
        voltages = [12_100] + [0xFFFF] * 9
        payload = (
            struct.pack("<BBBh", 0, 0, 3, 250)
            + struct.pack("<10H", *voltages)
            + struct.pack("<hiib", 0, 0, 0, 72)
        )
        d = parse_battery_status(payload)
        assert d["voltage_mv"] == 12_100
        assert d["battery_remaining_pct"] == 72

    def test_gps_raw_int(self):
        payload = struct.pack(
            "<QiiiHHHHBB",
            0,
            int(37.5 * 1e7),
            int(-122.3 * 1e7),
            50_000,
            100,
            100,
            0,
            0,
            3,
            12,
        )
        d = parse_gps_raw_int(payload)
        assert math.isclose(d["lat_deg"], 37.5, rel_tol=1e-6)
        assert math.isclose(d["lon_deg"], -122.3, rel_tol=1e-6)
        assert d["fix_type"] == 3
        assert d["satellites"] == 12


# ---------------------------------------------------------------------------
# Skill translation
# ---------------------------------------------------------------------------
class TestSkillTranslation:
    def test_takeoff(self):
        cmd = translate_skill("TAKEOFF", {"altitude_m": 20})
        assert cmd["command"] == MAV_CMD_NAV_TAKEOFF
        assert math.isclose(cmd["params"][6], 20.0)

    def test_takeoff_default_altitude(self):
        cmd = translate_skill("TAKEOFF")
        assert cmd["command"] == MAV_CMD_NAV_TAKEOFF
        assert cmd["params"][6] > 0  # sane default

    def test_land(self):
        cmd = translate_skill("LAND")
        assert cmd["command"] == MAV_CMD_NAV_LAND

    def test_return_home(self):
        for s in ("RETURN_HOME", "RTL", "RTH"):
            cmd = translate_skill(s)
            assert cmd["command"] == MAV_CMD_NAV_RETURN_TO_LAUNCH

    def test_navigate_to(self):
        cmd = translate_skill(
            "NAVIGATE_TO",
            {"lat": 40.0, "lon": -3.7, "alt_m": 25.0},
        )
        assert cmd["command"] == MAV_CMD_NAV_WAYPOINT
        # lat / lon go into param5 / param6
        assert math.isclose(cmd["params"][4], 40.0)
        assert math.isclose(cmd["params"][5], -3.7)
        assert math.isclose(cmd["params"][6], 25.0)

    def test_arm_disarm(self):
        arm = translate_skill("ARM")
        disarm = translate_skill("DISARM")
        assert arm["command"] == MAV_CMD_COMPONENT_ARM_DISARM
        assert disarm["command"] == MAV_CMD_COMPONENT_ARM_DISARM
        assert arm["params"][0] == float(ARM_DISARM_ARM)
        assert disarm["params"][0] == float(ARM_DISARM_DISARM)

    def test_hover_is_noop(self):
        # HOVER is handled by PX4 LOITER, no MAVLink command needed.
        assert translate_skill("HOVER") is None

    def test_unknown_skill(self):
        assert translate_skill("DANCE_LIKE_NOBODYS_WATCHING") is None


# ---------------------------------------------------------------------------
# Connection string parsing
# ---------------------------------------------------------------------------
class TestConnectionString:
    def test_udp_url(self):
        s, h, p = parse_connection_string("udp://192.168.1.5:14550")
        assert s == "udp"
        assert h == "192.168.1.5"
        assert p == 14550

    def test_tcp_url(self):
        s, h, p = parse_connection_string("tcp://sim.local:5760")
        assert s == "tcp"
        assert h == "sim.local"
        assert p == 5760

    def test_default_scheme_udp(self):
        s, _, p = parse_connection_string("127.0.0.1:14540")
        assert s == "udp"
        assert p == 14540


# ---------------------------------------------------------------------------
# MavTelemetry
# ---------------------------------------------------------------------------
class TestMavTelemetry:
    def test_defaults(self):
        t = MavTelemetry()
        assert t.lat == 0.0
        assert not t.armed
        assert not t.connected

    def test_connected_requires_recent_heartbeat(self):
        import time as _t

        t = MavTelemetry()
        t.last_heartbeat = _t.time()
        assert t.connected
        t.last_heartbeat = _t.time() - TELEMETRY_TIMEOUT_S - 1
        assert not t.connected


# ---------------------------------------------------------------------------
# MAVLinkClient — uses FakeTransport, no sockets
# ---------------------------------------------------------------------------
class TestMAVLinkClient:
    def _build_client(self):
        tx = FakeTransport()
        c = MAVLinkClient(transport=tx)
        # We bypass connect() so no background tasks run in unit tests.
        c._transport = tx
        return c, tx

    def test_init_default_url(self):
        c = MAVLinkClient()
        assert c.connection_string == DEFAULT_UDP_URL
        assert not c.connected
        assert c.telemetry.lat == 0.0

    def test_telemetry_isolation(self):
        a = MAVLinkClient()
        b = MAVLinkClient()
        a.telemetry.lat = 12.0
        assert b.telemetry.lat == 0.0

    def test_arm_sends_command_long(self):
        c, tx = self._build_client()
        asyncio.run(c.arm())
        assert len(tx.sent) == 1
        frame = decode_mavlink_v1(tx.sent[0])
        assert frame is not None
        assert frame.msgid == MAVLINK_MSG_ID_COMMAND_LONG
        p1, *_rest, cmd, _ts, _tc, _conf = struct.unpack(
            "<fffffffHBBB",
            frame.payload[:33],
        )
        assert cmd == MAV_CMD_COMPONENT_ARM_DISARM
        assert p1 == float(ARM_DISARM_ARM)

    def test_disarm_sends_command_long(self):
        c, tx = self._build_client()
        asyncio.run(c.disarm())
        frame = decode_mavlink_v1(tx.sent[0])
        p1, *_r, cmd, _ts, _tc, _conf = struct.unpack(
            "<fffffffHBBB",
            frame.payload[:33],
        )
        assert cmd == MAV_CMD_COMPONENT_ARM_DISARM
        assert p1 == float(ARM_DISARM_DISARM)

    def test_takeoff_sends_altitude(self):
        c, tx = self._build_client()
        asyncio.run(c.takeoff(altitude_m=17.5))
        frame = decode_mavlink_v1(tx.sent[0])
        _p1, _p2, _p3, _p4, _p5, _p6, p7, cmd, *_ = struct.unpack(
            "<fffffffHBBB",
            frame.payload[:33],
        )
        assert cmd == MAV_CMD_NAV_TAKEOFF
        assert math.isclose(p7, 17.5, rel_tol=1e-5)

    def test_land_command(self):
        c, tx = self._build_client()
        asyncio.run(c.land())
        frame = decode_mavlink_v1(tx.sent[0])
        _p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert _p[7] == MAV_CMD_NAV_LAND

    def test_rtl_command(self):
        c, tx = self._build_client()
        asyncio.run(c.rtl())
        frame = decode_mavlink_v1(tx.sent[0])
        _p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert _p[7] == MAV_CMD_NAV_RETURN_TO_LAUNCH

    def test_navigate_to_waypoint(self):
        c, tx = self._build_client()
        asyncio.run(c.navigate_to_waypoint(40.1, -3.7, 25.0))
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_NAV_WAYPOINT
        assert math.isclose(p[4], 40.1, rel_tol=1e-5)  # lat
        assert math.isclose(p[5], -3.7, rel_tol=1e-5)  # lon
        assert math.isclose(p[6], 25.0, rel_tol=1e-5)  # alt

    def test_execute_skill_takeoff(self):
        c, tx = self._build_client()
        asyncio.run(c.execute_skill("TAKEOFF", {"altitude_m": 8}))
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_NAV_TAKEOFF
        assert math.isclose(p[6], 8.0, rel_tol=1e-5)

    def test_execute_skill_unknown_is_noop(self):
        c, tx = self._build_client()
        ok = asyncio.run(c.execute_skill("NONEXISTENT"))
        assert ok is False
        assert tx.sent == []

    def test_execute_skill_hover_is_noop(self):
        c, tx = self._build_client()
        ok = asyncio.run(c.execute_skill("HOVER"))
        assert ok is False  # no MAVLink command sent
        assert tx.sent == []

    def test_sequence_counter_increments(self):
        c, tx = self._build_client()
        asyncio.run(c.arm())
        asyncio.run(c.takeoff(10.0))
        asyncio.run(c.land())
        seqs = [decode_mavlink_v1(f).seq for f in tx.sent]
        assert seqs == [0, 1, 2]


# ---------------------------------------------------------------------------
# Rx path — feeding real wire bytes through the client updates telemetry.
# ---------------------------------------------------------------------------
class TestRxPath:
    def test_heartbeat_marks_connection_live(self):
        c = MAVLinkClient(transport=FakeTransport())
        frame = encode_heartbeat(seq=0)
        c.feed_bytes(frame)
        assert c.telemetry.last_heartbeat > 0

    def test_global_position_updates_state(self):
        c = MAVLinkClient(transport=FakeTransport())
        payload = struct.pack(
            "<IiiiihhhH",
            1,
            int(45.0 * 1e7),
            int(10.0 * 1e7),
            100_000,
            15_000,
            0,
            0,
            0,
            18_000,
        )
        frame = encode_mavlink_v1(MAVLINK_MSG_ID_GLOBAL_POSITION_INT, payload, seq=0)
        c.feed_bytes(frame)
        assert math.isclose(c.telemetry.lat, 45.0, rel_tol=1e-6)
        assert math.isclose(c.telemetry.lon, 10.0, rel_tol=1e-6)
        assert math.isclose(c.telemetry.alt_m, 15.0, rel_tol=1e-3)
        assert math.isclose(c.telemetry.heading_deg, 180.0, rel_tol=1e-3)

    def test_heartbeat_armed_flag(self):
        c = MAVLinkClient(transport=FakeTransport())
        # Build a heartbeat with base_mode including the armed bit.
        payload = struct.pack(
            "<IBBBBB",
            0,
            6,
            8,
            0x80,
            4,
            3,
        )
        frame = encode_mavlink_v1(MAVLINK_MSG_ID_HEARTBEAT, payload, seq=0)
        c.feed_bytes(frame)
        assert c.telemetry.armed is True

    def test_partial_bytes_are_buffered(self):
        c = MAVLinkClient(transport=FakeTransport())
        frame = encode_heartbeat(seq=1)
        # Split in half — the first feed must not decode anything.
        mid = len(frame) // 2
        assert c.feed_bytes(frame[:mid]) == []
        frames = c.feed_bytes(frame[mid:])
        assert len(frames) == 1
        assert frames[0].msgid == MAVLINK_MSG_ID_HEARTBEAT

    def test_garbage_then_frame_resyncs(self):
        c = MAVLinkClient(transport=FakeTransport())
        junk = b"\x00\x11\x22"
        frame = encode_heartbeat(seq=2)
        frames = c.feed_bytes(junk + frame)
        assert len(frames) == 1
        assert frames[0].msgid == MAVLINK_MSG_ID_HEARTBEAT


# ---------------------------------------------------------------------------
# Failsafe
# ---------------------------------------------------------------------------
class TestFailsafe:
    def _with_fresh_heartbeat(self, c):
        import time as _t

        c.telemetry.last_heartbeat = _t.time()

    def test_no_action_when_disconnected(self):
        c = MAVLinkClient(transport=FakeTransport())
        c.telemetry.battery_pct = 5  # critical
        # No heartbeat — can't trust telemetry
        action = asyncio.run(c.check_failsafe())
        assert action is None

    def test_critical_battery_triggers_land(self):
        tx = FakeTransport()
        c = MAVLinkClient(transport=tx)
        self._with_fresh_heartbeat(c)
        c.telemetry.battery_pct = BATTERY_LAND_PCT - 1
        action = asyncio.run(c.check_failsafe())
        assert action == "land"
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_NAV_LAND

    def test_low_battery_triggers_rtl(self):
        tx = FakeTransport()
        c = MAVLinkClient(transport=tx)
        self._with_fresh_heartbeat(c)
        c.telemetry.battery_pct = BATTERY_RTL_PCT - 1
        action = asyncio.run(c.check_failsafe())
        assert action == "rtl"
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_NAV_RETURN_TO_LAUNCH

    def test_healthy_battery_no_action(self):
        c = MAVLinkClient(transport=FakeTransport())
        self._with_fresh_heartbeat(c)
        c.telemetry.battery_pct = 80
        assert asyncio.run(c.check_failsafe()) is None

    def test_disconnect_in_air_triggers_rtl(self):
        tx = FakeTransport()
        c = MAVLinkClient(transport=tx)
        c.telemetry.armed = True
        c.telemetry.alt_m = 25.0
        action = asyncio.run(c.failsafe_on_disconnect())
        assert action == "rtl"
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_NAV_RETURN_TO_LAUNCH

    def test_disconnect_on_ground_triggers_disarm(self):
        tx = FakeTransport()
        c = MAVLinkClient(transport=tx)
        c.telemetry.armed = True
        c.telemetry.alt_m = 0.5  # below DISARM_ON_GROUND_ALT_M
        action = asyncio.run(c.failsafe_on_disconnect())
        assert action == "disarm"
        frame = decode_mavlink_v1(tx.sent[0])
        p = struct.unpack("<fffffffHBBB", frame.payload[:33])
        assert p[7] == MAV_CMD_COMPONENT_ARM_DISARM
        assert p[0] == float(ARM_DISARM_DISARM)

    def test_disconnect_disarmed_no_action(self):
        c = MAVLinkClient(transport=FakeTransport())
        c.telemetry.armed = False
        action = asyncio.run(c.failsafe_on_disconnect())
        assert action is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
