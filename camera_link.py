# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Camera connection listener (C1): the robot's camera frames on their own TCP
connection, beside the control connection.

On one connection, a lost segment of camera bulk stalls every sensor, status
and e-stop frame queued behind it (TCP head-of-line blocking). On its own
connection it stalls only the camera.

The rules, in the order `CameraListener.handle` applies them:

1. A camera connection is accepted only while a control connection is live,
   and only from that connection's source address. The address match pairs
   the two connections; it is NOT authentication. In development the brain
   binds loopback and QEMU user-net arrives from 127.0.0.1, so any local
   process matches.
2. One camera connection at a time.
3. With a link key, the brain runs its responder handshake (the robot dials
   and speaks first) keyed by the same link key. That is what authenticates
   the connection, and the connection gets its own Noise session
   (`protocol.Link`) with its own message counters.
4. The connection is receive-only. With a link key, the writer's link is
   unregistered after the handshake (`protocol.forbid_sends`), so any
   `protocol.send_packet` on it refuses: the kernel's camera task never
   reads its socket, so nothing written there would be heard. Without a key
   (the unauthenticated development link) the listener simply never writes.
5. Only CAMERA_FRAME packets are delivered. Any other type closes the
   connection undelivered: it must not feed sensor or status state.
6. `close()` ends the camera connection; the server calls it when control
   drops.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

import protocol
from protocol import CAMERA_FRAME


def peer_host(writer: asyncio.StreamWriter) -> Optional[str]:
    """The source address of the connection `writer` belongs to."""
    peer = writer.get_extra_info("peername")
    if isinstance(peer, tuple) and peer:
        return str(peer[0])
    return None


class CameraListener:
    """Accepts the camera connection and hands its CAMERA_FRAME payloads to
    `on_frame`.

    `control_peer` returns the source address of the live control connection,
    or None when there is none. `read_timeout_s` bounds each read, as the
    control connection's does (B-A15)."""

    def __init__(
        self,
        control_peer: Callable[[], Optional[str]],
        on_frame: Callable[[bytes], Awaitable[None]],
        read_timeout_s: float,
    ) -> None:
        self._control_peer = control_peer
        self._on_frame = on_frame
        self._read_timeout_s = read_timeout_s
        self._writer: Optional[asyncio.StreamWriter] = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    def _refusal(self, peer: Optional[str]) -> str:
        control = self._control_peer()
        if control is None:
            return "no control connection"
        if peer != control:
            return f"source {peer} is not the control connection's {control}"
        if self._writer is not None:
            return "a camera connection is already live"
        return ""

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """`asyncio.start_server` callback for the camera port."""
        peer = peer_host(writer)
        reason = self._refusal(peer)
        if reason:
            print(f"[BRAIN] camera: refusing connection from {peer}: {reason}")
            await _close(writer)
            return
        # No await between the check and this assignment.
        self._writer = writer
        print(f"[BRAIN] camera: connected from {peer}")
        try:
            if protocol.encrypt_link_armed():
                if not await protocol.perform_handshake(reader, writer):
                    print("[BRAIN] camera: handshake failed, dropping connection")
                    return
                protocol.forbid_sends(writer)
            await self._receive(reader)
        except (asyncio.IncompleteReadError, ConnectionError):
            print("[BRAIN] camera: disconnected")
        except Exception as e:
            print(f"[BRAIN] camera: error: {e}")
        finally:
            protocol.drop_link(reader, writer)
            if self._writer is writer:
                self._writer = None
            await _close(writer)

    async def _receive(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                pkt = await asyncio.wait_for(protocol.read_packet(reader), self._read_timeout_s)
            except asyncio.TimeoutError:
                # Close, never resume: the deadline may have fired mid-frame.
                print(f"[BRAIN] camera: no packet within {self._read_timeout_s:.0f}s, dropping")
                return
            if pkt is None:
                print("[BRAIN] camera: invalid packet, dropping connection")
                return
            pkt_type, payload = pkt
            if pkt_type != CAMERA_FRAME:
                print(f"[BRAIN] camera: packet type 0x{pkt_type:02x} is not a camera frame, dropping")
                return
            await self._on_frame(payload)

    def close(self) -> None:
        """End the live camera connection, if any."""
        if self._writer is not None:
            self._writer.close()


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass
