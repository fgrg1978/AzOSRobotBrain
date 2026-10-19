# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for secure_channel.py: the link key, and the byte pins of the Noise
brain link against the Rust implementation.

The Rust side (`tests/host/aead-link-tests`, `tests/host/crypto-tests`) is the
other half of every pin below: its host tests assert the same hex, and the
vectors were produced there first. A failing pin is a finding on one side or
the other, never a constant to adjust.
"""

from secure_channel import load_link_key


def test_load_link_key_missing_returns_none(monkeypatch):
    monkeypatch.delenv("AZOS_BRAIN_LINK_KEY", raising=False)
    assert load_link_key() is None


def test_load_link_key_invalid_hex_returns_none(monkeypatch, capsys):
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", "zz")
    assert load_link_key() is None
    out = capsys.readouterr().out
    assert "not valid hex" in out


def test_load_link_key_wrong_length_returns_none(monkeypatch, capsys):
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", "00" * 31)
    assert load_link_key() is None
    out = capsys.readouterr().out
    assert "must be" in out


def test_load_link_key_valid_32_bytes(monkeypatch):
    hex32 = "00" * 32
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", hex32)
    key = load_link_key()
    assert key is not None
    assert len(key) == 32


# ── Noise brain link: byte pins against the Rust implementation ─────────────
#
# Scenario of `tests/host/aead-link-tests/tests/link.rs`: link key
# bytes(range(32)), robot ephemeral 0x33*32, brain ephemeral 0x44*32, IK toward
# the key of 0x77*32. Produced by the Rust code (`print_vectors`), reproduced
# here by an implementation that shares nothing with it.

import pytest

from secure_channel import (
    CipherState,
    HandshakeState,
    PATTERN_IK,
    PATTERN_XX,
    SecureChannel,
    brain_static_key,
    kernel_static_key,
    public_key,
)

pytest.importorskip("cryptography")

_PSK = bytes(range(32))
_BRAIN_IK_SK = bytes([0x77]) * 32
_E_ROBOT = bytes([0x33]) * 32
_E_BRAIN = bytes([0x44]) * 32

KERNEL_SK = "9f1fcde4e89ea1e2a099cfef639daab5e6cbce97d016e6cc1a0a1898c423f774"
KERNEL_PK = "ed8af765db3e506ce9993ceb2cb824c2d1640b5531d8bb3dd19700a5243b271d"
BRAIN_SK = "ac9b2eb32bf8bb4e742800c7d14826bedcddabdc85a489981a12e501dfa94d7d"
BRAIN_PK = "b36a27f7b9be42fc6c11307347556eb1a34e81095684b6df11af3781382b1434"
IK_BRAIN_PK = "1cf579aba45a10ba1d1ef06d91fca2aa9ed0a1150515653155405d0b18cb9a67"
IK_MSG1 = "4900607b0d47d93427f8311160781c7c733fd89f88970aef490d8aa0ee19a4cb8a1b140f31ce1e7b91df672735b75efa7d1eb743576872dc47b54ab044e26204667554f5208f24537e08182160bf762ed828cffd85fb0ec9fbe0da4b4d0fb42acd6f26"
IK_MSG2 = "0030ff2ee45601ec1b67310c7790404585ae697331eee1c1f8cf2419731c1fff3e6b1346dc7690404d34f646d6a09752a7fd"
IK_SID = "4dc0b2a9167f8b13bec0936fd40a3179"
IK_ROBOT_F1 = "001b57e276ed408c8d0521e7132db429421eb654291aa56c19b3ce72d6"
IK_ROBOT_F2 = "001e2070952fc43a92b28242947a8af5023c287e63277a329ca8a2faf6d47500"
IK_BRAIN_F1 = "001b25b2d87f1d737f45a590b176ed6a807f8307713130335a82865f7b"
IK_ROBOT_F3_REKEY = "0010955a5c3aac132c8b360ed7c56203cb61001b2a45bd172f6ff7b1188fb23992a8bca8125d00ae3041ca1bfa6af6"
XX_MSG1 = "5800207b0d47d93427f8311160781c7c733fd89f88970aef490d8aa0ee19a4cb8a1b14"
XX_MSG2 = "0060ff2ee45601ec1b67310c7790404585ae697331eee1c1f8cf2419731c1fff3e6b406dfee9ebefd84127fadf3bdc15ad1f16be32cfa431ebeff801e358e531010a4f77fce2a2489b29f805adb69afd37198e8d917e40a8c3e1e3513f6682243dfd"
XX_MSG3 = "0040bf490b653f124ebe7b2c2f514e4c107e92839895c8e19bd66328b57810d84cb76e8c50871e943be1cb3246bde1040012af44047aa7c60f52998553d4ffa162e1"
XX_SID = "70742d52f1efae547de5ed473d7dd0c2"
XX_ROBOT_F1 = "001b0f817fca54e339881ea8c89296fd3791fbab2eb2d101c1469eb738"
XX_BRAIN_F1 = "001bfdb82326098580d5bfabe8205bf997936778601eb86cb104ca7b3f"


def test_derived_static_keys_match_rust() -> None:
    assert kernel_static_key(_PSK).hex() == KERNEL_SK
    assert public_key(kernel_static_key(_PSK)).hex() == KERNEL_PK
    assert brain_static_key(_PSK).hex() == BRAIN_SK
    assert public_key(brain_static_key(_PSK)).hex() == BRAIN_PK
    assert public_key(_BRAIN_IK_SK).hex() == IK_BRAIN_PK


def _run(pinned: bool):
    robot = SecureChannel(
        _PSK, True, remote_pinned=public_key(_BRAIN_IK_SK) if pinned else None, _testing_eph_priv=_E_ROBOT
    )
    brain = SecureChannel(
        _PSK, False, static_key=_BRAIN_IK_SK if pinned else None, _testing_eph_priv=_E_BRAIN
    )
    wire = [robot.start_handshake()]
    brain.begin(wire[0][0])
    turn = brain
    msg = wire[0][1:]
    while not (robot.is_established and brain.is_established):
        out = turn.handle_handshake(msg)
        if out is None:
            break
        wire.append(out)
        msg = out
        turn = robot if turn is brain else brain
    return robot, brain, wire


def test_ik_handshake_and_transport_match_rust() -> None:
    robot, brain, wire = _run(True)
    assert [w.hex() for w in wire] == [IK_MSG1, IK_MSG2]
    assert robot.session_id is not None and robot.session_id.hex() == IK_SID
    assert brain.session_id == robot.session_id
    assert robot.seal_message(b"hello brain").hex() == IK_ROBOT_F1
    assert robot.seal_message(b"second message").hex() == IK_ROBOT_F2
    assert brain.seal_message(b"hello robot").hex() == IK_BRAIN_F1
    robot.request_rekey()
    assert robot.seal_message(b"after rekey").hex() == IK_ROBOT_F3_REKEY


def test_xx_handshake_and_transport_match_rust() -> None:
    robot, brain, wire = _run(False)
    assert [w.hex() for w in wire] == [XX_MSG1, XX_MSG2, XX_MSG3]
    assert robot.session_id is not None and robot.session_id.hex() == XX_SID
    assert robot.seal_message(b"hello brain").hex() == XX_ROBOT_F1
    assert brain.seal_message(b"hello robot").hex() == XX_BRAIN_F1


def test_the_rust_frames_open_here() -> None:
    robot, brain, _ = _run(True)
    # Frames the Rust code sealed (robot -> brain, in order) open in this
    # implementation, which proves the keys and nonces, not just the bytes.
    assert brain.open_frame(bytes.fromhex(IK_ROBOT_F1)) == (SecureChannel.RECORD_DATA, b"hello brain")
    assert brain.open_frame(bytes.fromhex(IK_ROBOT_F2)) == (SecureChannel.RECORD_DATA, b"second message")
    assert robot.open_frame(bytes.fromhex(IK_BRAIN_F1)) == (SecureChannel.RECORD_DATA, b"hello robot")
    assert brain.open_message(bytes.fromhex(IK_ROBOT_F3_REKEY)) == b"after rekey"
    assert brain.rx_generation == 1


# The generic Noise vectors of `tests/host/crypto-tests/tests/noise_handshakes.rs`
# (static 0x11/0x22, ephemeral 0x33/0x44, prologue "azos-test-prologue",
# payloads, transcript hash, transport frames): this HandshakeState reproduces
# them, payloads included.
_SI, _SR, _EI, _ER = (bytes([b]) * 32 for b in (0x11, 0x22, 0x33, 0x44))
_PROLOGUE = b"azos-test-prologue"
_IK_H = "816e4ebe557cb1671f451ccb4d8903acd1f53afa1f02198c8f8d9a8fab5f4c69"
_IK_MSGS = ['7b0d47d93427f8311160781c7c733fd89f88970aef490d8aa0ee19a4cb8a1b14180209d22e6c95bf3b198efd0e3af2c121a6a3464ded18bd60a996a63e55b86906c3e40bcac34dd41e4f5e92018861ce6490928f5f88674847f4b900e882117299e33adb8d010cc0d90544ee84de31927eccb0', 'ff2ee45601ec1b67310c7790404585ae697331eee1c1f8cf2419731c1fff3e6b2ec8a31b41fc3d83f2d0383f4c517933aeb41d25']
_IK_TRANSPORT = ['dfc329e2577c8bc8acfda71c78cfd6e61d28def8bd', '67ec578c899eaa49d02a12c1367df14c4df908ee92', 'ad6d09e16803b9d6e1de8bcff9c0182ff5dbdc7fbf']
_IK_C1_REKEYED = "7b4693c298ac347c5c1fd90a943506f1fa0317af62c8c11f56fbd89c7d398b00"
_XX_H = "d2802ece6dcb69c89ccb64fa31349796315c572c9234ba158517c0998dd4807c"
_XX_MSGS = ['7b0d47d93427f8311160781c7c733fd89f88970aef490d8aa0ee19a4cb8a1b1468656c6c6f', 'ff2ee45601ec1b67310c7790404585ae697331eee1c1f8cf2419731c1fff3e6bfcadb15080d9fd0434a18565751d3b6022bec571f33b6212486a1bffa54d1a1e285cbb44e7c744f18200da4a5b870145a9acabd8be8a7ce7077b38f16ebd9369f940466f7814', 'a7ea7dd47dddbcfcd736b91b174c6107b2ad26c161965a119a7b644e0c6b3d06f089b04f2601563c6f1bed63ab4375be05d22753a3fa39f4c98232e63c592b9a6c01f2a3c2']
_XX_TRANSPORT = ['bb02a0d6a5131f6d8557c6a3d8d2254c63533cd371', '554f58619934bb2325970656fee22e8086ee24ad1a', '66d2913d84e19521a52ae87c0815f980bbba43e2c5']


def _generic(pattern: int, payloads: list[bytes]):
    rs = public_key(_SR) if pattern == PATTERN_IK else None
    a = HandshakeState(pattern, True, _SI, rs, _PROLOGUE)
    b = HandshakeState(pattern, False, _SR, None, _PROLOGUE)
    wire = []
    for i, payload in enumerate(payloads):
        w, r, eph = (a, b, _EI) if i % 2 == 0 else (b, a, _ER)
        msg = w.write_message(eph, payload)
        assert r.read_message(msg) == payload
        wire.append(msg)
    return wire, a, b


def _seal(cs: CipherState, pt: bytes) -> str:
    return cs.encrypt(b"ad", pt).hex()


@pytest.mark.parametrize(
    "pattern,payloads,h,msgs,transport",
    [
        (PATTERN_IK, [b"ping from initiator", b"pong"], _IK_H, _IK_MSGS, _IK_TRANSPORT),
        (PATTERN_XX, [b"hello", b"world!", b"third"], _XX_H, _XX_MSGS, _XX_TRANSPORT),
    ],
)
def test_generic_noise_vectors_match_rust(pattern, payloads, h, msgs, transport) -> None:
    wire, a, b = _generic(pattern, payloads)
    assert [w.hex() for w in wire] == msgs
    assert a.handshake_hash.hex() == h == b.handshake_hash.hex()
    a_send, _, _, a_rs = a.finish()
    b_send, b_recv, _, b_rs = b.finish()
    assert a_rs == public_key(_SR) and b_rs == public_key(_SI)
    assert _seal(a_send, b"i2r-0") == transport[0]
    assert _seal(a_send, b"i2r-1") == transport[1]
    assert _seal(b_send, b"r2i-0") == transport[2]
    assert b_recv.decrypt(b"ad", bytes.fromhex(transport[0])) == b"i2r-0"


def test_rekey_matches_rust_reference() -> None:
    _, a, _ = _generic(PATTERN_IK, [b"ping from initiator", b"pong"])
    send, _, _, _ = a.finish()
    send.rekey()
    assert send.k is not None and send.k.hex() == _IK_C1_REKEYED
