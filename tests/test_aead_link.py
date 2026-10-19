# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""The brain link's Noise channel (RFC-0019), brain side.

Covers `secure_channel.SecureChannel` as the brain uses it (responder) and as
the robot does (initiator, the role the Rust side plays): both patterns, the
static-key checks, the framed transport, rekey and the terminal violations.
The cross-implementation byte pins (fixed keys, produced by the Rust code in
`crates/core/crypto/src/secure_channel.rs`) are in `test_secure_channel.py`.
"""

from __future__ import annotations

import pytest

from secure_channel import (
    BRAIN_STATIC_INFO,
    FRAME_HEAD,
    FRAME_OVERHEAD,
    HANDSHAKE_MSG_MAX,
    MAX_MESSAGE_BYTES,
    PATTERN_IK,
    PATTERN_XX,
    REKEY_MAX_RECORDS,
    REKEY_MIN_RECORDS,
    CipherState,
    HandshakeState,
    LinkViolation,
    NoiseError,
    SecureChannel,
    brain_static_key,
    handshake_frame_size,
    kernel_static_key,
    public_key,
    rekey_interval_s,
)

pytest.importorskip("cryptography")

PSK = bytes(range(32))
BRAIN_SK = bytes([0x77]) * 32
E_ROBOT = bytes([0x33]) * 32
E_BRAIN = bytes([0x44]) * 32


def _pair(
    pinned: bool = True,
    *,
    robot_psk: bytes = PSK,
    brain_psk: bytes = PSK,
    clock=lambda: 0.0,
) -> tuple[SecureChannel, SecureChannel]:
    """A completed handshake: (robot = initiator, brain = responder)."""
    robot = SecureChannel(
        robot_psk,
        True,
        remote_pinned=public_key(BRAIN_SK) if pinned else None,
        clock=clock,
        _testing_eph_priv=E_ROBOT,
    )
    brain = SecureChannel(
        brain_psk,
        False,
        static_key=BRAIN_SK if pinned else None,
        clock=clock,
        _testing_eph_priv=E_BRAIN,
    )
    first = robot.start_handshake()
    brain.begin(first[0])
    msg = first[1:]
    turn = brain
    while not (robot.is_established and brain.is_established):
        out = turn.handle_handshake(msg)
        if out is None:
            break
        msg = out
        turn = robot if turn is brain else brain
    assert robot.is_established and brain.is_established
    return robot, brain


# ── Handshakes ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pinned", [True, False])
def test_both_patterns_establish_and_traffic_flows_both_ways(pinned: bool) -> None:
    robot, brain = _pair(pinned)
    assert robot.session_id == brain.session_id and robot.session_id is not None
    for i in range(40):
        msg = bytes([i]) * (1 + 7 * i)
        assert brain.open_message(robot.seal_message(msg)) == msg
        assert robot.open_message(brain.seal_message(msg)) == msg


def test_ik_authenticates_the_pinned_static_and_xx_the_derived_one() -> None:
    robot, brain = _pair(True)
    assert robot.remote_static == public_key(BRAIN_SK)
    assert brain.remote_static == public_key(kernel_static_key(PSK))
    robot, brain = _pair(False)
    assert robot.remote_static == public_key(brain_static_key(PSK))
    assert brain.remote_static == public_key(kernel_static_key(PSK))


def test_a_brain_with_another_link_key_is_refused_by_the_robot_in_xx() -> None:
    other = bytes((b ^ 1) for b in PSK)
    robot = SecureChannel(PSK, True, _testing_eph_priv=E_ROBOT)
    brain = SecureChannel(other, False, _testing_eph_priv=E_BRAIN)
    first = robot.start_handshake()
    brain.begin(first[0])
    msg2 = brain.handle_handshake(first[1:])
    assert msg2 is not None
    with pytest.raises(LinkViolation, match="link key derives"):
        robot.handle_handshake(msg2)  # message 3 (the robot's static) never goes out
    assert robot.is_rejected and not robot.is_established


def test_a_robot_with_another_link_key_is_refused_by_the_brain_in_ik() -> None:
    other = bytes((b ^ 1) for b in PSK)
    robot = SecureChannel(other, True, remote_pinned=public_key(BRAIN_SK), _testing_eph_priv=E_ROBOT)
    brain = SecureChannel(PSK, False, static_key=BRAIN_SK, _testing_eph_priv=E_BRAIN)
    first = robot.start_handshake()
    brain.begin(first[0])
    with pytest.raises(LinkViolation, match="link key derives"):
        brain.handle_handshake(first[1:])
    assert brain.is_rejected


def test_an_ik_robot_that_pinned_another_key_is_refused_by_the_brain() -> None:
    robot = SecureChannel(PSK, True, remote_pinned=public_key(bytes([0x78]) * 32), _testing_eph_priv=E_ROBOT)
    brain = SecureChannel(PSK, False, static_key=BRAIN_SK, _testing_eph_priv=E_BRAIN)
    first = robot.start_handshake()
    brain.begin(first[0])
    with pytest.raises(LinkViolation, match="handshake"):
        brain.handle_handshake(first[1:])


def test_ik_without_a_static_key_and_an_unknown_pattern_are_refused_at_begin() -> None:
    brain = SecureChannel(PSK, False)
    with pytest.raises(LinkViolation, match="AZOS_BRAIN_NOISE_SK"):
        brain.begin(PATTERN_IK)
    brain = SecureChannel(PSK, False)
    with pytest.raises(LinkViolation, match="unknown handshake pattern"):
        brain.begin(0x51)


def test_the_pattern_byte_is_bound_into_the_transcript() -> None:
    # The same messages with the other pattern's prologue fail the first AEAD.
    robot = SecureChannel(PSK, True, remote_pinned=public_key(BRAIN_SK), _testing_eph_priv=E_ROBOT)
    first = robot.start_handshake()
    hs = HandshakeState(PATTERN_IK, False, BRAIN_SK, None, b"azos-brain-link-v1" + bytes([PATTERN_XX]))
    with pytest.raises(NoiseError):
        hs.read_message(first[3:])


def test_handshake_frame_checks() -> None:
    assert handshake_frame_size(b"\x00\x60") == 2 + 0x60
    with pytest.raises(ValueError):
        handshake_frame_size(b"\x00\x00")
    with pytest.raises(ValueError):
        handshake_frame_size(bytes([0xFF, 0xFF]))
    assert HANDSHAKE_MSG_MAX >= 96
    robot = SecureChannel(PSK, True, _testing_eph_priv=E_ROBOT)
    brain = SecureChannel(PSK, False, _testing_eph_priv=E_BRAIN)
    first = robot.start_handshake()
    brain.begin(first[0])
    with pytest.raises(LinkViolation):
        brain.handle_handshake(first[1:-1])  # the prefix lies about the length
    with pytest.raises(RuntimeError):
        brain.handle_handshake(first[1:])  # a failed channel does not resume


def test_a_low_order_ephemeral_is_refused() -> None:
    robot = SecureChannel(PSK, True, _testing_eph_priv=E_ROBOT)
    robot.start_handshake()
    zero_e = bytes(32)
    msg = (96).to_bytes(2, "big") + zero_e + bytes(64)
    with pytest.raises(LinkViolation):
        robot.handle_handshake(msg)


def test_keys_derive_per_side_and_purpose() -> None:
    assert kernel_static_key(PSK) != brain_static_key(PSK) != PSK
    assert kernel_static_key(PSK) != kernel_static_key(bytes((b ^ 1) for b in PSK))
    assert BRAIN_STATIC_INFO != b""


# ── Transport violations are terminal ────────────────────────────────────────


def _dead(sc: SecureChannel) -> None:
    assert not sc.is_established
    with pytest.raises(RuntimeError):
        sc.seal_message(b"x")
    with pytest.raises(LinkViolation):
        sc.open_frame(bytes(40))


def test_a_forged_frame_is_terminal() -> None:
    robot, brain = _pair()
    wire = bytearray(robot.seal_message(b"stop everything"))
    wire[5] ^= 1
    with pytest.raises(LinkViolation):
        brain.open_frame(bytes(wire))
    assert brain.drops_decrypt == 1
    _dead(brain)


def test_the_length_prefix_is_authenticated() -> None:
    robot, brain = _pair()
    wire = bytearray(robot.seal_message(b"stop everything"))
    wire[1] += 2
    wire += b"\x00\x00"
    with pytest.raises(LinkViolation):
        brain.open_frame(bytes(wire))
    _dead(brain)


def test_replayed_reordered_dropped_and_reflected_frames_are_terminal() -> None:
    robot, brain = _pair()
    f1 = robot.seal_message(b"first")
    brain.open_frame(f1)
    with pytest.raises(LinkViolation):
        brain.open_frame(f1)  # replay
    robot, brain = _pair()
    robot.seal_message(b"first")
    f2 = robot.seal_message(b"second")
    with pytest.raises(LinkViolation):
        brain.open_frame(f2)  # f1 was dropped
    robot, brain = _pair()
    with pytest.raises(LinkViolation):
        robot.open_frame(robot.seal_message(b"mine"))  # reflected at its sender


def test_a_frame_from_another_session_does_not_open() -> None:
    _, brain_a = _pair(True)
    robot_b, _ = _pair(False)
    # Both sessions are deterministic here; their keys still differ.
    with pytest.raises(LinkViolation):
        brain_a.open_frame(robot_b.seal_message(b"recorded elsewhere"))


def test_bad_lengths_are_refused_before_a_body_is_read() -> None:
    _, brain = _pair()
    with pytest.raises(LinkViolation):
        brain.frame_size(bytes([0xFF, 0xFF]))
    _, brain = _pair()
    with pytest.raises(LinkViolation):
        brain.frame_size(b"\x00\x03")


def test_seal_refusals() -> None:
    robot, _ = _pair()
    with pytest.raises(ValueError):
        robot.seal_message(b"")
    with pytest.raises(ValueError):
        robot.seal_message(bytes(MAX_MESSAGE_BYTES + 1))
    assert len(robot.seal_message(bytes(MAX_MESSAGE_BYTES))) == FRAME_OVERHEAD + MAX_MESSAGE_BYTES
    robot.close()
    with pytest.raises(RuntimeError):
        robot.seal_message(b"x")


def test_open_message_takes_one_message_and_refuses_the_rest() -> None:
    robot, brain = _pair()
    a = robot.seal_message(b"one")
    b = robot.seal_message(b"two")
    with pytest.raises(LinkViolation, match="after the message"):
        brain.open_message(a + b)
    robot, brain = _pair()
    wire = robot.seal_message(b"one")
    with pytest.raises(LinkViolation, match="torn"):
        brain.open_message(wire[:-1])
    robot, brain = _pair()
    robot.request_rekey()
    assert brain.open_message(robot.seal_message(b"after")) == b"after"  # rekey + data


# ── Rekey ────────────────────────────────────────────────────────────────────


def test_a_requested_rekey_precedes_the_message_and_both_ends_follow() -> None:
    robot, brain = _pair()
    assert len(robot.seal_message(b"gen0")) == FRAME_OVERHEAD + 4
    robot2, brain2 = _pair()
    brain2.open_message(robot2.seal_message(b"gen0"))
    robot2.request_rekey()
    wire = robot2.seal_message(b"gen1")
    assert len(wire) == 2 * FRAME_OVERHEAD + 4
    assert robot2.tx_generation == 1
    assert brain2.open_message(wire) == b"gen1"
    assert brain2.rx_generation == 1
    assert len(robot2.seal_message(b"still gen1")) == FRAME_OVERHEAD + 10


def test_a_frame_from_before_the_rekey_does_not_open_after_it() -> None:
    robot, brain = _pair()
    old = robot.seal_message(b"old")
    brain.open_frame(old)
    robot.request_rekey()
    brain.open_message(robot.seal_message(b"new"))
    with pytest.raises(LinkViolation):
        brain.open_frame(old)


def test_message_and_byte_limits_trigger_a_rekey() -> None:
    robot, brain = _pair()
    robot.set_rekey_limits(4, 1 << 30)
    brain.set_rekey_limits(4, 1 << 30)
    gens = []
    for i in range(10):
        brain.open_message(robot.seal_message(bytes([i]) * 9))
        gens.append(robot.tx_generation)
    # 3 data messages and a rekey per generation.
    assert gens == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    robot, _ = _pair()
    robot.set_rekey_limits(1 << 20, 16 * 1024)
    robot.seal_message(bytes(9000))
    assert robot.tx_generation == 0
    robot.seal_message(bytes(9000))
    assert robot.tx_generation == 1


def test_a_peer_that_skips_the_rekey_is_refused() -> None:
    robot, brain = _pair()
    robot.set_rekey_limits(1 << 20, 1 << 30)  # the sender is lax...
    brain.set_rekey_limits(4, 1 << 30)  # ...the receiver is not
    for i in range(3):
        brain.open_frame(robot.seal_message(bytes([i]) * 5))
    with pytest.raises(LinkViolation, match="rekey limit"):
        brain.open_frame(robot.seal_message(b"fourth"))
    _dead(brain)


def test_wall_clock_triggers_a_rekey() -> None:
    now = [0.0]
    robot, brain = _pair(clock=lambda: now[0])
    robot.set_rekey_limits(REKEY_MAX_RECORDS, 1 << 30, interval_s=60.0)
    brain.open_message(robot.seal_message(b"early"))
    assert robot.tx_generation == 0
    now[0] = 61.0
    assert brain.open_message(robot.seal_message(b"late")) == b"late"
    assert robot.tx_generation == 1


def test_rekey_limits_are_clamped_and_the_interval_only_shortens() -> None:
    robot, _ = _pair()
    robot.set_rekey_limits(0, 0)
    assert len(robot.seal_message(b"x")) >= FRAME_OVERHEAD + 1
    assert REKEY_MIN_RECORDS == 2
    assert rekey_interval_s(None) == 3600.0
    assert rekey_interval_s(86400) == 3600.0
    assert rekey_interval_s(60) == 60.0
    assert rekey_interval_s(1) == 5.0


# ── Primitives ───────────────────────────────────────────────────────────────


def test_cipherstate_counter_advances_only_on_success_and_rekey_keeps_it() -> None:
    a, b = CipherState(bytes(32)), CipherState(bytes(32))
    ct = a.encrypt(b"ad", b"payload")
    bad = bytes([ct[0] ^ 1]) + ct[1:]
    with pytest.raises(NoiseError):
        b.decrypt(b"ad", bad)
    assert b.n == 0
    assert b.decrypt(b"ad", ct) == b"payload" and b.n == 1
    a.rekey()
    b.rekey()
    assert a.n == 1 and a.k == b.k and a.k != bytes(32)
    assert b.decrypt(b"", a.encrypt(b"", b"x")) == b"x"
