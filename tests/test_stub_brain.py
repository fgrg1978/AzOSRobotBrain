# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Self-test for tools/stub_brain.py — exercises the kernel↔brain wire
loop end-to-end in-process: spawn the stub on a free port, connect as
a fake robot, send N sensor packets, expect an ActuatorCmd back.

This is the Python-only half of the Phase 1 E2E story. The kernel-in-
QEMU half is a separate Makefile target (e2e-wheeled-qemu); both
exercise the same stub_brain.py so a green pytest run here implies
the wire-format glue is sound before involving QEMU.
"""

import asyncio
import os
import socket
import sys
import time
from contextlib import contextmanager
from typing import Optional

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import protocol
from protocol import (
    build_packet,
    parse_packet,
    SensorPacket,
    SENSOR_PACKET,
    ACTUATOR_CMD,
)
from secure_channel import KEY_BYTES, LinkViolation
from stub_brain import (
    EXIT_AUTH_FAILURE,
    EXIT_NO_LOOP,
    EXIT_OK,
    StubBrain,
    _corrupt_key_hex,
    arm_link_layer,
    compute_exit_code,
)
from tests.mock_kernel import MockRobot, reset_protocol_state


def _free_port() -> int:
    """Ask the kernel for an ephemeral port we can bind to. Avoids
    the test-flakiness of hardcoding 8080 on a dev box that might
    already have something listening there."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_sensor_packet(seq: int) -> bytes:
    """Build a minimal SensorPacket-shaped byte stream the stub
    will accept. Field values don't matter — we just want the
    wire format to validate."""
    pkt = SensorPacket(
        timestamp_ms=seq * 100,
        battery_mv=7400,
        accel_mg=(0, 0, 1000),
        gyro_mdps=(0, 0, 0),
        odom_dist_mm=seq * 10,
        odom_hdg_cdeg=0,
        encoder_l=seq,
        encoder_r=seq,
        range_front_mm=1500,
        range_right_mm=1500,
    )
    return build_packet(SENSOR_PACKET, pkt.to_bytes())


def test_loop_closes_after_n_sensors():
    """Sync wrapper that drives the loop via asyncio.run, so the
    test suite doesn't need pytest-asyncio installed."""
    port = _free_port()
    brain = StubBrain(
        duration_s=5,
        sensors_to_go=3,
        forward_pct=50,
        quiet=True,
    )

    async def run() -> "Optional[bytes]":
        server = await asyncio.start_server(brain.handle, host="127.0.0.1", port=port)

        async def fake_robot() -> "Optional[bytes]":
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            for seq in range(3):
                writer.write(_make_sensor_packet(seq))
                await writer.drain()
                await asyncio.sleep(0.05)
            buf = b""
            deadline = asyncio.get_event_loop().time() + 2.0
            actuator_payload = None  # Optional[bytes]
            while asyncio.get_event_loop().time() < deadline:
                try:
                    chunk = await asyncio.wait_for(reader.read(256), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    break
                buf += chunk
                parsed = parse_packet(buf)
                if parsed is not None:
                    pkt_type, payload = parsed
                    if pkt_type == ACTUATOR_CMD:
                        actuator_payload = payload
                        break
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return actuator_payload

        async with server:
            return await asyncio.wait_for(fake_robot(), timeout=10)

    actuator_payload = asyncio.run(run())
    assert actuator_payload is not None, "brain never sent an ActuatorCmd"
    assert brain.pkt_counts.get(SENSOR_PACKET, 0) == 3
    assert brain.cmds_sent == 1


def test_summary_format_smoke():
    brain = StubBrain(duration_s=10, sensors_to_go=3, forward_pct=50, quiet=True)
    # Call _summary with empty state — must not blow up before
    # the first packet arrives.
    s = brain._summary()
    assert "rx[" in s and "tx[" in s


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted-link coverage (RFC-0019)
#
# With a key armed the stub is the Noise responder and every read goes through
# the session. These tests pin the property that matters to the gate: with a
# key armed the harness MUST reject what does not verify, and say so in its
# exit code.
#
# No listeners here — StubBrain.handle() is driven over a socketpair by a mock
# robot (the Noise initiator, `tests/mock_kernel.py`).
# ─────────────────────────────────────────────────────────────────────────────

#: Deterministic test keys (never used on a real link).
_KEY = bytes(range(KEY_BYTES))
_KEY_HEX = _KEY.hex()
_WRONG_KEY = bytes(bytearray(_KEY[:-1]) + bytes([_KEY[-1] ^ 0x01]))

#: Generous when a command is expected, short when one must NOT arrive.
_EXPECT_ACTUATOR_TIMEOUT_S = 3.0
_NO_ACTUATOR_TIMEOUT_S = 0.5
_JOIN_TIMEOUT_S = 5.0
_SESSION_DURATION_S = 10


@contextmanager
def _armed(key_hex):
    """Arm protocol.py's link state exactly as the CLI does, and tear it down
    so the rest of the suite stays plaintext."""
    prev = os.environ.get("AZOS_COMPANION_LINK_KEY")
    os.environ["AZOS_COMPANION_LINK_KEY"] = key_hex
    try:
        assert protocol.enable_encrypt_link() is True
        yield
    finally:
        reset_protocol_state()
        if prev is None:
            os.environ.pop("AZOS_COMPANION_LINK_KEY", None)
        else:
            os.environ["AZOS_COMPANION_LINK_KEY"] = prev


async def _read_actuator(robot, timeout_s):
    """Robot-side reader: the first ActuatorCmd payload the brain sends, or
    None (nothing within `timeout_s`, or the brain hung up)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            parsed = await asyncio.wait_for(robot.read_packet(), timeout=deadline - time.monotonic())
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, LinkViolation):
            return None
        if parsed is not None and parsed[0] == ACTUATOR_CMD:
            return parsed[1]
    return None


def _drive(brain, script, *, psk=_KEY, handshake=True):
    """Play the robot: hand one end of a socketpair to StubBrain.handle, run
    the mock robot's handshake (unless `handshake` is False), then `script`
    on it, then EOF. Returns what the script returned; a handshake the brain
    or the robot refuses returns None."""

    async def run():
        sock_stub, sock_bot = socket.socketpair()
        r_stub, w_stub = await asyncio.open_connection(sock=sock_stub)
        r_bot, w_bot = await asyncio.open_connection(sock=sock_bot)
        task = asyncio.create_task(brain.handle(r_stub, w_stub))
        robot = MockRobot(r_bot, w_bot, psk=psk)
        result = None
        try:
            if handshake:
                await robot.handshake()
            result = await script(robot)
        except (asyncio.IncompleteReadError, ConnectionError, LinkViolation):
            pass
        finally:
            w_bot.close()
            try:
                await w_bot.wait_closed()
            except Exception:
                pass
        await asyncio.wait_for(task, timeout=_JOIN_TIMEOUT_S)
        return result

    return asyncio.run(run())


def _sensors(n, read_timeout_s=_EXPECT_ACTUATOR_TIMEOUT_S):
    """Script: send `n` sealed sensor packets, then read the ActuatorCmd."""

    async def script(robot):
        for seq in range(n):
            await robot.send_message(_make_sensor_packet(seq))
        return await _read_actuator(robot, read_timeout_s)

    return script


def _brain(sensors_to_go=3, expect_auth_failure=False):
    return StubBrain(
        duration_s=_SESSION_DURATION_S,
        sensors_to_go=sensors_to_go,
        forward_pct=50,
        quiet=True,
        expect_auth_failure=expect_auth_failure,
    )


def test_encrypted_loop_closes_with_correct_key():
    """Positive: robot and brain share the key, packets open, the brain answers."""
    with _armed(_KEY_HEX):
        brain = _brain()
        assert brain.encrypted is True
        payload = _drive(brain, _sensors(3))
        code = compute_exit_code(brain)

    assert payload is not None, "brain never sent a readable ActuatorCmd"
    assert brain.verified_frames == 3
    assert brain.auth_failures == 0
    assert brain.bad_inner_frames == 0
    assert brain.pkt_counts.get(SENSOR_PACKET, 0) == 3
    assert code == EXIT_OK


def test_a_wrong_key_is_refused_at_the_handshake():
    """Negative: another key → no session, nothing reaches the brain, gate fails."""
    with _armed(_KEY_HEX):
        brain = _brain()
        payload = _drive(brain, _sensors(3, _NO_ACTUATOR_TIMEOUT_S), psk=_WRONG_KEY)
        code = compute_exit_code(brain)
        # Even --accept-tcp-only must not paper over a crypto failure.
        tcp_only_code = compute_exit_code(brain, accept_tcp_only=True)

    assert payload is None
    assert brain.verified_frames == 0
    assert brain.auth_failures == 1, "the refused handshake must be counted"
    assert brain.pkt_counts == {}
    assert code == EXIT_AUTH_FAILURE
    assert tcp_only_code == EXIT_AUTH_FAILURE


def test_a_tampered_frame_ends_the_session_and_fails_the_gate():
    """One flipped bit in a sealed frame: the session ends, nothing after it
    is read, and the verdict is a link failure, not noise."""

    async def script(robot):
        await robot.send_message(_make_sensor_packet(0))
        wire = bytearray(robot.sc.seal_message(_make_sensor_packet(1)))
        wire[-1] ^= 0x01
        robot.writer.write(bytes(wire))
        robot.writer.write(robot.sc.seal_message(_make_sensor_packet(2)))
        await robot.writer.drain()
        return await _read_actuator(robot, _NO_ACTUATOR_TIMEOUT_S)

    with _armed(_KEY_HEX):
        brain = _brain()
        payload = _drive(brain, script)
        code = compute_exit_code(brain)

    assert payload is None
    assert brain.verified_frames == 1
    assert brain.auth_failures == 1
    assert code == EXIT_AUTH_FAILURE


def test_expect_auth_failure_mode_passes_only_when_nothing_verifies():
    """`--expect-auth-failure`: exit 0 iff the robot never got a session."""
    with _armed(_KEY_HEX):
        rejected = _brain(expect_auth_failure=True)
        _drive(rejected, _sensors(3, _NO_ACTUATOR_TIMEOUT_S), psk=_WRONG_KEY)
        rejected_code = compute_exit_code(rejected)

        accepted = _brain(expect_auth_failure=True)
        _drive(accepted, _sensors(3))
        accepted_code = compute_exit_code(accepted)

        silent = _brain(expect_auth_failure=True)

        async def nothing(robot):
            return None

        _drive(silent, nothing, handshake=False)
        silent_code = compute_exit_code(silent)

    assert rejected_code == EXIT_OK and rejected.auth_failures == 1
    # Anything verifying under a wrong key means the gate is not enforcing.
    assert accepted_code == EXIT_AUTH_FAILURE and accepted.verified_frames == 3
    # And a silent link proves nothing — it must not pass either.
    assert silent_code == EXIT_NO_LOOP


def test_a_stream_cut_after_the_session_is_not_an_auth_failure():
    """The kernel closing between frames is ordinary TCP teardown. The gate
    must fail on cryptography, not on disconnects."""
    with _armed(_KEY_HEX):
        brain = _brain(sensors_to_go=2)
        payload = _drive(brain, _sensors(2))
        code = compute_exit_code(brain)

    assert payload is not None
    assert brain.verified_frames == 2
    assert brain.auth_failures == 0
    assert code == EXIT_OK


def test_an_opened_message_that_does_not_parse_is_fatal():
    """A frame that opens but a parser that rejects it means the two repos
    disagree on the wire format — exactly what this harness must catch."""

    async def script(robot):
        await robot.send_message(b"\x00" * 32)
        return await _read_actuator(robot, _NO_ACTUATOR_TIMEOUT_S)

    with _armed(_KEY_HEX):
        brain = _brain()
        _drive(brain, script)
        code = compute_exit_code(brain)

    assert brain.bad_inner_frames == 1
    assert brain.auth_failures == 0
    assert brain.pkt_counts == {}
    assert code == EXIT_AUTH_FAILURE


def _drive_plain(brain, stream, read_timeout_s=_EXPECT_ACTUATOR_TIMEOUT_S):
    """Plaintext development link: hand `stream` raw to StubBrain.handle over
    a socketpair and collect the ActuatorCmd payload."""

    async def read_actuator(reader):
        buf = b""
        deadline = time.monotonic() + read_timeout_s
        while time.monotonic() < deadline:
            try:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                return None
            buf += chunk
            parsed = parse_packet(buf)
            if parsed is not None and parsed[0] == ACTUATOR_CMD:
                return parsed[1]
        return None

    async def run():
        sock_stub, sock_bot = socket.socketpair()
        r_stub, w_stub = await asyncio.open_connection(sock=sock_stub)
        r_bot, w_bot = await asyncio.open_connection(sock=sock_bot)
        task = asyncio.create_task(brain.handle(r_stub, w_stub))
        w_bot.write(stream)
        await w_bot.drain()
        payload = await read_actuator(r_bot)
        w_bot.close()
        try:
            await w_bot.wait_closed()
        except Exception:
            pass
        await asyncio.wait_for(task, timeout=_JOIN_TIMEOUT_S)
        return payload

    return asyncio.run(run())


def test_plaintext_mode_still_resyncs_and_stays_non_fatal():
    """No key → nothing to verify. Leading junk is counted and logged
    but must not turn currently-green keyless scenarios red."""
    brain = _brain()
    assert brain.encrypted is False
    stream = b"\x99" * 7 + b"".join(_make_sensor_packet(i) for i in range(3))
    payload = _drive_plain(brain, stream)
    code = compute_exit_code(brain)

    assert payload is not None
    assert brain.pkt_counts.get(SENSOR_PACKET, 0) == 3
    assert brain.skew_bytes == 7
    assert brain.noise_bytes == 0
    assert code == EXIT_OK


def test_arm_link_layer_never_downgrades_silently():
    prev = os.environ.get("AZOS_COMPANION_LINK_KEY")
    try:
        # No key → plaintext, no error.
        os.environ.pop("AZOS_COMPANION_LINK_KEY", None)
        assert arm_link_layer(quiet=True, expect_auth_failure=False) is None
        assert protocol.encrypt_link_armed() is False

        # Malformed key → hard failure, NOT a quiet fallback to plaintext.
        os.environ["AZOS_COMPANION_LINK_KEY"] = "not-hex"
        assert arm_link_layer(quiet=True, expect_auth_failure=False) == EXIT_AUTH_FAILURE
        assert protocol.encrypt_link_armed() is False

        # Negative mode without a key is a misconfiguration, not a pass.
        os.environ.pop("AZOS_COMPANION_LINK_KEY", None)
        assert arm_link_layer(quiet=True, expect_auth_failure=True) == EXIT_AUTH_FAILURE

        # Negative mode with a key: arms with a DIFFERENT (corrupted) key.
        os.environ["AZOS_COMPANION_LINK_KEY"] = _KEY_HEX
        assert arm_link_layer(quiet=True, expect_auth_failure=True) is None
        assert protocol.encrypt_link_armed() is True
        assert os.environ["AZOS_COMPANION_LINK_KEY"] != _KEY_HEX
        assert bytes.fromhex(os.environ["AZOS_COMPANION_LINK_KEY"]) != _KEY
    finally:
        reset_protocol_state()
        if prev is None:
            os.environ.pop("AZOS_COMPANION_LINK_KEY", None)
        else:
            os.environ["AZOS_COMPANION_LINK_KEY"] = prev


def test_corrupt_key_hex_changes_exactly_one_bit():
    corrupted = bytes.fromhex(_corrupt_key_hex(_KEY_HEX))
    assert len(corrupted) == KEY_BYTES
    assert corrupted != _KEY
    diff = sum((a ^ b).bit_count() for a, b in zip(_KEY, corrupted))
    assert diff == 1
    with pytest.raises(ValueError):
        _corrupt_key_hex("00" * (KEY_BYTES - 1))
