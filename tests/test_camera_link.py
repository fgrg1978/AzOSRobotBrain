# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""C1 camera connection listener (`camera_link.py`).

A mock kernel DIALS the listener over real loopback, as the kernel's camera
sender will, and plays the RFC-0019 responder. Each test names the canary
that must turn it red.

Run: `python3.12 -m pytest tests/test_camera_link.py`
"""

import asyncio

import pytest

import protocol
from camera_link import CameraListener

sc = pytest.importorskip("secure_channel")
if not getattr(sc, "_HAS_CRYPTOGRAPHY", False):
    pytest.skip("cryptography package not installed", allow_module_level=True)

from tests.mock_kernel import HEXKEY, MockRobot  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_protocol_state():
    def _clear():
        protocol._encrypt_psk = None
        protocol._noise_static = None
        protocol._links_by_reader.clear()
        protocol._links_by_writer.clear()
        protocol._recent_session_ids.clear()
        protocol._multi_stream_armed = False

    _clear()
    yield
    _clear()


def _arm(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", HEXKEY)
    assert protocol.enable_encrypt_link() is True


def _unarmed(monkeypatch):
    monkeypatch.delenv("AZOS_BRAIN_LINK_KEY", raising=False)
    assert protocol.enable_encrypt_link() is False


class _Brain:
    """The listener on an ephemeral loopback port, recording delivered frames."""

    def __init__(self, control="127.0.0.1"):
        self.control = control
        self.frames = []
        self.listener = CameraListener(lambda: self.control, self._on_frame, read_timeout_s=5.0)

    async def _on_frame(self, payload):
        self.frames.append(payload)

    async def start(self):
        self.server = await asyncio.start_server(self.listener.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        self.listener.close()
        self.server.close()
        try:
            await asyncio.wait_for(self.server.wait_closed(), 2.0)
        except asyncio.TimeoutError:
            pass


class _Kernel:
    """The robot end of a camera connection."""

    async def dial(self, port, handshake):
        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", port)
        if handshake:
            self.robot = MockRobot(self.reader, self.writer)
            await asyncio.wait_for(self.robot.handshake(), 2.0)
        return self

    async def send_sealed(self, pkt_type, payload):
        await self.robot.send_packet(pkt_type, payload)

    async def read_or_none(self, timeout):
        """Bytes the brain sent (b"" = it closed), or None if it stayed silent.
        A reset counts as closed: bytes the kernel wrote into a socket the
        brain was closing come back as an RST."""
        try:
            return await asyncio.wait_for(self.reader.read(64), timeout)
        except asyncio.TimeoutError:
            return None
        except ConnectionResetError:
            return b""

    def close(self):
        self.writer.close()


def test_refused_without_a_live_control_connection(monkeypatch):
    """Canary: drop the control-connection check — the brain sends its HELLO."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain(control=None)
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=False)
        try:
            return await kernel.read_or_none(2.0), brain.frames
        finally:
            kernel.close()
            await brain.stop()

    got, frames = asyncio.run(run())
    assert got == b"", "the brain answered a camera connection with no control connection"
    assert frames == []


def test_refused_from_another_source_address(monkeypatch):
    """Canary: drop the source-address check — the brain sends its HELLO."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain(control="10.0.2.99")
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=False)
        try:
            return await kernel.read_or_none(2.0)
        finally:
            kernel.close()
            await brain.stop()

    assert asyncio.run(run()) == b"", "a camera connection from another address was answered"


def test_frames_are_delivered_and_the_brain_writes_nothing_after_the_handshake(monkeypatch):
    """Canary: send an acknowledgement after each frame — the send refuses
    (receive-only), the connection closes, and the second frame is lost."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=True)
        try:
            await kernel.send_sealed(protocol.CAMERA_FRAME, b"frame-1")
            await asyncio.sleep(0.1)
            await kernel.send_sealed(protocol.CAMERA_FRAME, b"frame-2")
            await asyncio.sleep(0.1)
            return brain.frames, await kernel.read_or_none(0.3)
        finally:
            kernel.close()
            await brain.stop()

    frames, after = asyncio.run(run())
    assert frames == [b"frame-1", b"frame-2"]
    assert after is None, f"the brain wrote on the camera connection: {after!r}"


def test_the_camera_connection_cannot_send_once_established(monkeypatch):
    """Receive-only is enforced, not just observed: `send_packet` on the
    camera writer refuses while its reader keeps the session. Canary: skip
    `protocol.forbid_sends` — the send goes out sealed."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=True)
        try:
            await kernel.send_sealed(protocol.CAMERA_FRAME, b"frame-1")
            await asyncio.sleep(0.1)
            camera_writer = brain.listener._writer
            assert camera_writer is not None
            refused = False
            try:
                await protocol.send_packet(camera_writer, protocol.STATUS, b"x")
            except ConnectionResetError:
                refused = True
            await kernel.send_sealed(protocol.CAMERA_FRAME, b"frame-2")
            await asyncio.sleep(0.1)
            return refused, brain.frames, await kernel.read_or_none(0.3)
        finally:
            kernel.close()
            await brain.stop()

    refused, frames, after = asyncio.run(run())
    assert refused, "send_packet went out on the camera connection"
    assert after is None, f"bytes reached the kernel on the camera connection: {after!r}"
    assert frames == [b"frame-1", b"frame-2"], "the reader lost its session"


def test_a_non_camera_packet_closes_the_connection_undelivered(monkeypatch):
    """Canary: deliver every packet type — the sensor payload reaches on_frame."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=True)
        try:
            await kernel.send_sealed(protocol.SENSOR_PACKET, b"sensor")
            await kernel.send_sealed(protocol.CAMERA_FRAME, b"after")
            return brain.frames, await kernel.read_or_none(2.0), brain.listener.connected
        finally:
            kernel.close()
            await brain.stop()

    frames, got, connected = asyncio.run(run())
    assert frames == []
    assert got == b"", "the connection stayed open after a non-camera packet"
    assert connected is False


def test_a_second_camera_connection_is_refused_while_one_is_live(monkeypatch):
    """Canary: drop the one-connection check — the second dial gets a HELLO."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        first = await _Kernel().dial(brain.port, handshake=True)
        second = await _Kernel().dial(brain.port, handshake=False)
        try:
            refused = await second.read_or_none(2.0)
            await first.send_sealed(protocol.CAMERA_FRAME, b"still-first")
            await asyncio.sleep(0.1)
            return refused, brain.frames
        finally:
            second.close()
            first.close()
            await brain.stop()

    refused, frames = asyncio.run(run())
    assert refused == b"", "a second camera connection was answered"
    assert frames == [b"still-first"]


def test_close_ends_the_live_camera_connection(monkeypatch):
    """The server calls `close()` when control drops. Canary: `close()` does
    nothing — the kernel's read times out instead of seeing the close."""
    _arm(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=True)
        try:
            await asyncio.sleep(0.05)
            brain.listener.close()
            got = await kernel.read_or_none(2.0)
            await asyncio.sleep(0.05)
            return got, brain.listener.connected
        finally:
            kernel.close()
            await brain.stop()

    got, connected = asyncio.run(run())
    assert got == b"", "close() left the camera connection open"
    assert connected is False


def test_unarmed_development_link_delivers_plaintext_frames(monkeypatch):
    """Development mode (no link key): pairing is the address match alone,
    which is not authentication. Pins that the path works."""
    _unarmed(monkeypatch)

    async def run():
        brain = _Brain()
        await brain.start()
        kernel = await _Kernel().dial(brain.port, handshake=False)
        try:
            kernel.writer.write(protocol.build_packet(protocol.CAMERA_FRAME, b"plain"))
            await kernel.writer.drain()
            await asyncio.sleep(0.1)
            return brain.frames
        finally:
            kernel.close()
            await brain.stop()

    assert asyncio.run(run()) == [b"plain"]
