# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0019 encrypted-link WIRING tests (protocol.py + server integration).

The crypto/state-machine itself is covered by `test_aead_link.py`. These
tests cover the *wiring* in `protocol.py`:

  - `enable_encrypt_link()` arming + the no-silent-fallback refusal,
  - `perform_handshake()` driving the brain (Noise responder) over real
    loopback streams against a mock robot (initiator, `tests/mock_kernel.py`),
  - the send/read path:  Noise( brain_frame ), nothing else,
  - handshake failure (bad PSK, wrong pattern) → connection refused,
  - one session per connection.

Run: `python -m pytest tests/test_encrypt_link_wiring.py`
"""

import asyncio
import struct

import pytest

import protocol

# Skip the whole module if the optional crypto dep is missing.
sc = pytest.importorskip("secure_channel")
if not getattr(sc, "_HAS_CRYPTOGRAPHY", False):
    pytest.skip("cryptography package not installed", allow_module_level=True)

from secure_channel import LinkViolation, SecureChannel, public_key  # noqa: E402
from tests.mock_kernel import HEXKEY, PSK, MockRobot, arm_brain, reset_protocol_state  # noqa: E402

BRAIN_SK = bytes([0x77]) * 32


@pytest.fixture(autouse=True)
def _reset_protocol_state():
    reset_protocol_state()
    yield
    reset_protocol_state()


# ── enable_encrypt_link / arming ─────────────────────────────────────────────


def test_enable_off_without_a_key(monkeypatch):
    # No key: nothing to arm, and `server.py` refuses to run.
    monkeypatch.delenv("AZOS_BRAIN_LINK_KEY", raising=False)
    assert protocol.enable_encrypt_link() is False
    assert protocol.encrypt_link_armed() is False


def test_enable_arms_with_a_key_and_there_is_no_switch_to_turn_it_off(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", HEXKEY)
    # The old HMAC-only / off switch is gone: setting it changes nothing.
    monkeypatch.setenv("AZOS_BRAIN_ENCRYPT_LINK", "0")
    assert protocol.enable_encrypt_link() is True
    assert protocol.encrypt_link_armed() is True


def test_enable_refuses_a_malformed_noise_key(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", HEXKEY)
    monkeypatch.setenv("AZOS_BRAIN_NOISE_SK", "zz")
    with pytest.raises(ValueError, match="AZOS_BRAIN_NOISE_SK"):
        protocol.enable_encrypt_link()


# ── The brain (responder) against a mock robot over loopback ────────────────


class _Rig:
    """A loopback brain server whose handler runs `perform_handshake` and then
    `after(reader, writer, ok)`; the test drives the robot end."""

    def __init__(self, after=None, handshake_timeout=10.0):
        self.after = after
        self.timeout = handshake_timeout
        self.ok = None
        self.result = None
        self.done = asyncio.Event()

    async def handler(self, reader, writer):
        try:
            self.ok = await protocol.perform_handshake(reader, writer, self.timeout)
            if self.after is not None:
                self.result = await self.after(reader, writer, self.ok)
        finally:
            self.done.set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", port)
        return self

    async def __aexit__(self, *exc):
        self.writer.close()
        self.server.close()
        await asyncio.wait_for(self.server.wait_closed(), 2.0)

    def robot(self, **kw):
        return MockRobot(self.reader, self.writer, **kw)

    async def wait(self):
        await asyncio.wait_for(self.done.wait(), 5.0)


def test_handshake_and_encrypted_roundtrip_xx(monkeypatch):
    arm_brain(monkeypatch)

    async def after(reader, writer, ok):
        assert ok
        link = protocol.link_for_writer(writer)
        assert link is not None and protocol.link_for_reader(reader) is link
        got = await protocol.read_packet(reader)  # robot → brain
        await protocol.send_packet(writer, 0x03, b"\x01\x02")  # brain → robot
        return got

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            await robot.send_packet(0x01, b"hello-payload")
            back = await robot.read_packet()
            await rig.wait()
            return rig.result, back

    got, back = asyncio.run(run())
    assert got == (0x01, b"hello-payload")  # the brain decrypted ours
    assert back == (0x03, b"\x01\x02")  # the robot decrypted the brain's


def test_handshake_and_roundtrip_ik_with_the_pinned_key(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_NOISE_SK", BRAIN_SK.hex())
    arm_brain(monkeypatch)
    assert protocol._noise_static == BRAIN_SK

    async def after(reader, writer, ok):
        assert ok
        return await protocol.read_packet(reader)

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot(pinned=public_key(BRAIN_SK))
            sc_ = await robot.handshake()
            assert sc_.remote_static == public_key(BRAIN_SK)
            await robot.send_packet(0x01, b"ik")
            await rig.wait()
            return rig.result

    assert asyncio.run(run()) == (0x01, b"ik")


def test_an_ik_robot_with_a_brain_that_has_no_static_key_is_refused(monkeypatch):
    # No AZOS_BRAIN_NOISE_SK: the brain serves only XX and refuses IK.
    arm_brain(monkeypatch)

    async def run():
        async with _Rig() as rig:
            robot = rig.robot(pinned=public_key(BRAIN_SK))
            with pytest.raises((asyncio.IncompleteReadError, ConnectionError, LinkViolation)):
                await robot.handshake()
            await rig.wait()
            return rig.ok

    assert asyncio.run(run()) is False


def test_an_ik_robot_that_pinned_another_key_is_refused(monkeypatch):
    monkeypatch.setenv("AZOS_BRAIN_NOISE_SK", BRAIN_SK.hex())
    arm_brain(monkeypatch)

    async def run():
        async with _Rig() as rig:
            robot = rig.robot(pinned=public_key(bytes([0x78]) * 32))
            with pytest.raises((asyncio.IncompleteReadError, ConnectionError, LinkViolation)):
                await robot.handshake()
            await rig.wait()
            return rig.ok, protocol.link_for_writer(rig.writer)

    ok, link = asyncio.run(run())
    assert ok is False and link is None


def test_handshake_fails_on_bad_psk(monkeypatch):
    arm_brain(monkeypatch)
    wrong_psk = bytes((b ^ 0xFF) for b in PSK)

    async def run():
        async with _Rig() as rig:
            robot = rig.robot(psk=wrong_psk)
            # XX: the robot refuses the brain's static (derived from the real
            # key) and never shows its own.
            with pytest.raises((asyncio.IncompleteReadError, ConnectionError, LinkViolation)):
                await robot.handshake()
            rig.writer.close()
            await rig.wait()
            return rig.ok, protocol.link_for_writer(rig.writer)

    ok, link = asyncio.run(run())
    assert ok is False  # no message 3: the brain gives up
    assert link is None  # channel not established


def test_a_garbage_pattern_byte_fails_closed(monkeypatch, capsys):
    arm_brain(monkeypatch)

    async def run():
        async with _Rig() as rig:
            rig.writer.write(b"Q" + bytes(40))
            await rig.writer.drain()
            await rig.wait()
            return rig.ok

    assert asyncio.run(run()) is False
    assert "handshake failed" in capsys.readouterr().out


def test_an_oversize_handshake_frame_is_refused_before_it_is_read(monkeypatch):
    arm_brain(monkeypatch)

    async def run():
        async with _Rig() as rig:
            rig.writer.write(b"X" + struct.pack(">H", 60000))  # no body follows
            await rig.writer.drain()
            await rig.wait()  # returns at once: not waiting for 60000 bytes
            return rig.ok

    assert asyncio.run(run()) is False


def test_handshake_times_out_on_a_silent_robot(monkeypatch):
    arm_brain(monkeypatch)

    async def run():
        async with _Rig(handshake_timeout=0.2) as rig:
            await rig.wait()
            return rig.ok

    assert asyncio.run(run()) is False


def test_perform_handshake_noop_when_not_armed(monkeypatch):
    # Encryption not armed → perform_handshake is a pass-through True, and no
    # bytes are exchanged.
    assert protocol.encrypt_link_armed() is False

    async def run():
        # Dummy reader/writer that would fail if touched.
        reader = asyncio.StreamReader()
        # A real writer isn't needed since the function returns before I/O.
        return await protocol.perform_handshake(reader, None)  # type: ignore[arg-type]

    assert asyncio.run(run()) is True


def test_repeated_session_id_is_refused(monkeypatch):
    arm_brain(monkeypatch)
    eph = b"\xbb" * 32

    async def run_once():
        async with _Rig() as rig:
            robot = rig.robot(eph=eph)
            try:
                await robot.handshake()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            await rig.wait()
            return rig.ok

    # The brain's ephemeral is fresh each time, so two identical robot
    # ephemerals still give distinct session ids: pin the brain's too.
    import functools

    real = sc.SecureChannel
    sc.SecureChannel = functools.partial(real, _testing_eph_priv=b"\xaa" * 32)  # type: ignore[misc]
    try:
        ok1 = asyncio.run(run_once())
        ok2 = asyncio.run(run_once())
    finally:
        sc.SecureChannel = real  # type: ignore[misc]
    assert ok1 is True
    assert ok2 is False, "a repeated session id was accepted"


# ── Frames: rekey, large messages, violations ────────────────────────────────


def test_camera_frame_arrives_across_a_rekey(monkeypatch):
    arm_brain(monkeypatch)
    jpeg = bytes((i * 13) & 0xFF for i in range(6000))
    cam_payload = struct.pack("<HHB", 160, 120, 1) + jpeg
    cam = protocol.build_packet(protocol.CAMERA_FRAME, cam_payload)
    status = protocol.build_packet(0x03, b"\x01\x02")

    async def after(reader, writer, ok):
        assert ok
        p1 = await protocol.read_packet(reader)
        p2 = await protocol.read_packet(reader)
        return p1, p2, protocol.link_for_reader(reader).secure_channel.rx_generation

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            robot.sc.request_rekey()
            await robot.send_message(cam)  # a rekey frame, then the message
            await robot.send_message(status)
            await rig.wait()
            return rig.result

    p1, p2, gen = asyncio.run(run())
    assert p1 == (protocol.CAMERA_FRAME, cam_payload)
    assert p2 == (0x03, b"\x01\x02")
    assert gen == 1


def test_brain_seals_a_large_packet_as_one_message(monkeypatch):
    arm_brain(monkeypatch)
    payload = bytes(range(256)) * 12  # 3072 B

    async def after(reader, writer, ok):
        assert ok
        await protocol.send_packet(writer, 0x85, payload)
        await asyncio.sleep(0.05)

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            pkt = await asyncio.wait_for(robot.read_packet(), 2.0)
            await rig.wait()
            return pkt

    assert asyncio.run(run()) == (0x85, payload)


def test_tampered_frame_ends_the_session(monkeypatch, capsys):
    arm_brain(monkeypatch)
    status = protocol.build_packet(0x03, b"\x01\x02")

    async def after(reader, writer, ok):
        assert ok
        pkt = await protocol.read_packet(reader)
        with pytest.raises(ConnectionResetError):
            await protocol.send_packet(writer, 0x80, b"\x00")
        return pkt

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            wire = bytearray(robot.sc.seal_message(status))
            wire[-1] ^= 0x01
            robot.writer.write(bytes(wire))
            await robot.writer.drain()
            await rig.wait()
            return rig.result

    assert asyncio.run(run()) is None
    assert "transport violation" in capsys.readouterr().out


def test_a_replayed_frame_ends_the_session(monkeypatch):
    arm_brain(monkeypatch)
    status = protocol.build_packet(0x03, b"\x01\x02")

    async def after(reader, writer, ok):
        assert ok
        first = await protocol.read_packet(reader)
        second = await protocol.read_packet(reader)
        return first, second

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            wire = robot.sc.seal_message(status)
            robot.writer.write(wire + wire)
            await robot.writer.drain()
            await rig.wait()
            return rig.result

    first, second = asyncio.run(run())
    assert first == (0x03, b"\x01\x02")
    assert second is None


def test_an_oversize_transport_frame_is_refused_before_it_is_read(monkeypatch):
    arm_brain(monkeypatch)

    async def after(reader, writer, ok):
        assert ok
        return await asyncio.wait_for(protocol.read_packet(reader), 2.0)

    async def run():
        async with _Rig(after) as rig:
            robot = rig.robot()
            await robot.handshake()
            robot.writer.write(struct.pack(">H", 60000))  # the body never comes
            await robot.writer.drain()
            await rig.wait()
            return rig.result

    assert asyncio.run(run()) is None


# ── One session per connection (C1: a camera connection beside control) ─────


async def _two(after_a, after_b):
    """Two brain connections, each with its own rig."""
    a = _Rig(after_a)
    b = _Rig(after_b)
    await a.__aenter__()
    await b.__aenter__()
    return a, b


async def _close_two(a, b):
    await a.__aexit__(None, None, None)
    await b.__aexit__(None, None, None)


def test_a_second_handshake_does_not_replace_the_first_connections_session(monkeypatch):
    """The session used to be one module global, so the second connection's
    handshake took over the first connection's channel, and the first
    connection then sealed and opened with the wrong keys.

    Canary: resolve the link from the most recent handshake instead of the
    stream — connection A's packet reaches its robot undecryptable."""
    arm_brain(monkeypatch)

    async def after_a(reader, writer, ok):
        assert ok
        await asyncio.sleep(0.3)  # B handshakes meanwhile
        got = await protocol.read_packet(reader)
        await protocol.send_packet(writer, 0x03, b"\xaa")
        return got, protocol.link_for_writer(writer)

    async def after_b(reader, writer, ok):
        assert ok
        got = await protocol.read_packet(reader)
        await protocol.send_packet(writer, 0x03, b"\xbb")
        return got, protocol.link_for_writer(writer)

    async def run():
        a, b = await _two(after_a, after_b)
        try:
            ra, rb = a.robot(), b.robot()
            await ra.handshake()
            await rb.handshake()
            await ra.send_packet(0x01, b"to-a")
            await rb.send_packet(0x01, b"to-b")
            back_a = await asyncio.wait_for(ra.read_packet(), 2.0)
            back_b = await asyncio.wait_for(rb.read_packet(), 2.0)
            await a.wait()
            await b.wait()
            return a.result, b.result, back_a, back_b
        finally:
            await _close_two(a, b)

    (got_a, link_a), (got_b, link_b), back_a, back_b = asyncio.run(run())
    assert got_a == (0x01, b"to-a"), "A's packet was opened with another session"
    assert back_a == (0x03, b"\xaa")
    assert got_b == (0x01, b"to-b")
    assert back_b == (0x03, b"\xbb")
    assert link_a is not None and link_b is not None
    assert link_a.secure_channel is not link_b.secure_channel


def test_dropping_one_link_leaves_the_other_established(monkeypatch):
    """Canary: `drop_link` clearing every link — A's send raises."""
    arm_brain(monkeypatch)
    drop_b = asyncio.Event()

    async def after_a(reader, writer, ok):
        assert ok
        await drop_b.wait()
        got = await protocol.read_packet(reader)
        await protocol.send_packet(writer, 0x03, b"\x01")
        return got

    async def after_b(reader, writer, ok):
        assert ok
        protocol.drop_link(reader, writer)
        assert protocol.link_for_writer(writer) is None
        assert protocol.link_for_reader(reader) is None
        drop_b.set()

    async def run():
        a, b = await _two(after_a, after_b)
        try:
            ra, rb = a.robot(), b.robot()
            await ra.handshake()
            await rb.handshake()
            await b.wait()
            await ra.send_packet(0x01, b"still-a")
            back = await asyncio.wait_for(ra.read_packet(), 2.0)
            await a.wait()
            return a.result, back
        finally:
            await _close_two(a, b)

    got, back = asyncio.run(run())
    assert got == (0x01, b"still-a")
    assert back == (0x03, b"\x01")


def test_a_violation_ends_only_its_own_connection(monkeypatch):
    arm_brain(monkeypatch)

    async def after_a(reader, writer, ok):
        assert ok
        return await protocol.read_packet(reader)

    async def after_b(reader, writer, ok):
        assert ok
        return await protocol.read_packet(reader)

    async def run():
        a, b = await _two(after_a, after_b)
        try:
            ra, rb = a.robot(), b.robot()
            await ra.handshake()
            await rb.handshake()
            wire = bytearray(ra.sc.seal_message(protocol.build_packet(0x03, b"x")))
            wire[3] ^= 0x01
            ra.writer.write(bytes(wire))
            await rb.send_packet(0x01, b"fine")
            await a.wait()
            await b.wait()
            return a.result, b.result
        finally:
            await _close_two(a, b)

    ra_, rb_ = asyncio.run(run())
    assert ra_ is None
    assert rb_ == (0x01, b"fine")


def test_an_armed_stream_with_no_session_refuses_instead_of_downgrading(monkeypatch):
    """With encryption armed, a stream that never completed a handshake is
    refused in both directions; nothing is written in plaintext.

    Canary: drop the refusal in `send_packet` — a plaintext frame is written."""
    arm_brain(monkeypatch)

    class _Writer:
        def __init__(self) -> None:
            self.buf = bytearray()

        def write(self, b: bytes) -> None:
            self.buf += b

        async def drain(self) -> None:
            pass

    async def run():
        w = _Writer()
        with pytest.raises(ConnectionResetError):
            await protocol.send_packet(w, 0x01, b"x")
        r = asyncio.StreamReader()
        r.feed_data(bytes(64))
        r.feed_eof()
        with pytest.raises(ConnectionResetError):
            await protocol.read_packet(r)
        return bytes(w.buf)

    assert asyncio.run(run()) == b"", "a frame was written without a session"


def test_the_brain_refuses_to_start_without_the_link_key(monkeypatch):
    """`run()` raises before it binds anything when no key is configured, and
    no environment opt-in changes that (the keyless link mode is gone).

    Canary: restore the old `AZOS_BRAIN_ALLOW_INSECURE_LINK` branch — the
    opt-in then lets `run()` past the check and it tries to bind."""
    from server import BrainServer

    monkeypatch.delenv("AZOS_BRAIN_LINK_KEY", raising=False)
    monkeypatch.setenv("AZOS_BRAIN_ALLOW_INSECURE_LINK", "1")
    srv = BrainServer.__new__(BrainServer)
    srv.config = {"robot": {"listen_port": 0}}
    with pytest.raises(RuntimeError, match="AZOS_BRAIN_LINK_KEY"):
        asyncio.run(srv.run())
