# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0021 multi-stream framing — wire-format pin + protocol.py wiring tests.

`multi_stream.py` is a byte-for-byte mirror of `crates/multi-stream`; the
wire-format vectors here must match the Rust crate's encoding. The wiring
tests prove the framing composes as the OUTERMOST layer in protocol.py
(plaintext and over the Noise transport).
"""

import asyncio

import pytest

import multi_stream as ms
import protocol

# ── Wire-format pins (must match crates/multi-stream byte layout) ────────────


def test_wire_format_matches_crate():
    # wrap(STREAM_CONTROL=0x00, [0x01,0x02,0x03]) -> 00 03 00 01 02 03
    assert ms.wrap(ms.STREAM_CONTROL, bytes([1, 2, 3])) == bytes(
        [0x00, 0x03, 0x00, 0x01, 0x02, 0x03]
    )
    # camera stream 0 = 0x10; len LE u16
    assert ms.wrap(ms.STREAM_CAMERA_BASE, b"\xaa" * 300)[:3] == bytes(
        [0x10, 0x2C, 0x01]
    )  # 300 = 0x012C LE
    assert ms.HEADER_LEN == 3
    assert ms.STREAM_CONTROL == 0x00
    assert (ms.STREAM_CAMERA_BASE, ms.STREAM_CAMERA_LAST) == (0x10, 0x1F)
    assert ms.STREAM_LIDAR == 0x20 and ms.STREAM_AUDIO == 0x21


def test_wrap_unwrap_roundtrip():
    for sid in (ms.STREAM_CONTROL, ms.STREAM_CAMERA_BASE, ms.STREAM_LIDAR):
        for payload in (b"", b"x", b"BR\x01\x00\x00\xab", b"\xff" * 1000):
            sid_out, p_out = ms.unwrap(ms.wrap(sid, payload))
            assert sid_out == sid and p_out == payload


def test_unwrap_rejects_short_and_truncated():
    assert ms.unwrap(b"") is None
    assert ms.unwrap(b"\x00\x03") is None  # < HEADER_LEN
    # LEN claims 5 but only 2 payload bytes present (length-extension/truncation)
    assert ms.unwrap(bytes([0x00, 0x05, 0x00, 0xAA, 0xBB])) is None


def test_payload_too_large_raises():
    with pytest.raises(ValueError):
        ms.wrap(ms.STREAM_CONTROL, b"\x00" * (ms.MAX_PAYLOAD_LEN + 1))


def test_camera_helpers():
    assert ms.camera_stream_id(0) == 0x10
    assert ms.camera_stream_id(15) == 0x1F
    assert ms.camera_stream_id(16) is None
    assert ms.is_camera_stream(0x10) and ms.is_camera_stream(0x1F)
    assert not ms.is_camera_stream(0x00) and not ms.is_camera_stream(0x20)


# ── protocol.py wiring (multi-stream as outermost layer) ─────────────────────


class _FakeWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, b: bytes) -> None:
        self.buf += b

    async def drain(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_protocol_state():
    def _clear():
        protocol._encrypt_psk = None
        protocol._links_by_reader.clear()
        protocol._links_by_writer.clear()
        protocol._multi_stream_armed = False

    _clear()
    yield
    _clear()


def _feed(data: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    r.feed_data(data)
    r.feed_eof()
    return r


def test_enable_from_env(monkeypatch):
    monkeypatch.delenv("AZOS_BRAIN_MULTI_STREAM", raising=False)
    assert protocol.enable_multi_stream() is False
    monkeypatch.setenv("AZOS_BRAIN_MULTI_STREAM", "1")
    assert protocol.enable_multi_stream() is True
    assert protocol.multi_stream_armed() is True


def test_roundtrip_plaintext_multistream(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_MULTI_STREAM", "1")
    protocol.enable_multi_stream()

    async def run():
        w = _FakeWriter()
        await protocol.send_packet(w, protocol.STATUS, b"\x01\x02\x03")
        # Wire must be a multi-stream frame on STREAM_CONTROL.
        assert w.buf[0] == ms.STREAM_CONTROL
        pkt = await protocol.read_packet(_feed(bytes(w.buf)))
        return pkt

    pkt_type, payload = asyncio.run(run())
    assert pkt_type == protocol.STATUS and payload == b"\x01\x02\x03"


def test_roundtrip_multistream_over_noise(monkeypatch):
    # Compose: multi_stream( Noise( brain_frame ) ). The handshake itself is
    # not multiplexed; every transport frame after it rides STREAM_CONTROL.
    pytest.importorskip("cryptography")
    from tests.mock_kernel import MockRobot, arm_brain

    arm_brain(monkeypatch)
    monkeypatch.setenv("AZOS_BRAIN_MULTI_STREAM", "1")
    assert protocol.enable_multi_stream() is True

    async def run():
        seen = {}
        done = asyncio.Event()

        async def handler(reader, writer):
            assert await protocol.perform_handshake(reader, writer)
            await protocol.send_packet(writer, protocol.SENSOR_PACKET, b"payload-bytes")
            seen["got"] = await protocol.read_packet(reader)
            done.set()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            robot = MockRobot(reader, writer)
            await robot.handshake()
            head = await reader.readexactly(ms.HEADER_LEN)
            assert head[0] == ms.STREAM_CONTROL
            plen = int.from_bytes(head[ms.STREAM_ID_BYTES : ms.HEADER_LEN], "little")
            frame = await reader.readexactly(plen)
            assert frame[:2] != protocol.MAGIC  # sealed, not a raw brain frame
            _kind, message = robot.sc.open_frame(frame)
            assert protocol.parse_packet(message) == (
                protocol.SENSOR_PACKET, b"payload-bytes")
            sealed = robot.sc.seal_message(
                protocol.build_packet(protocol.STATUS, b"ok"))
            writer.write(ms.wrap(ms.STREAM_CONTROL, sealed))
            await writer.drain()
            await asyncio.wait_for(done.wait(), 5.0)
            return seen["got"]
        finally:
            writer.close()
            server.close()

    assert asyncio.run(run()) == (protocol.STATUS, b"ok")


def test_reader_skips_noncontrol_stream(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_MULTI_STREAM", "1")
    protocol.enable_multi_stream()

    async def run():
        # A camera-stream frame followed by a control packet: the control
        # reader must skip the camera frame and return the control one.
        cam = ms.wrap(ms.STREAM_CAMERA_BASE, b"\xde\xad\xbe\xef")
        w = _FakeWriter()
        await protocol.send_packet(w, protocol.STATUS, b"ok")
        return await protocol.read_packet(_feed(cam + bytes(w.buf)))

    pkt_type, payload = asyncio.run(run())
    assert pkt_type == protocol.STATUS and payload == b"ok"
