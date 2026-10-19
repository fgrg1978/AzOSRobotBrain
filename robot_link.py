# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""The robot's end of the brain link, for simulated robots and mock peers.

The robot dials the brain, so it is the TCP client and the Noise initiator.
`RobotLink` runs that handshake over an open connection and then carries
brain-protocol frames, one per Noise transport message: the same session the
kernel speaks, so the SITL clients and the test robot exercise the brain's
real link code instead of a plaintext shortcut. There is no other mode.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import protocol
from secure_channel import (
    FRAME_HEAD,
    SecureChannel,
    handshake_frame_size,
    load_link_key,
)


class RobotLink:
    """The robot end of one connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        psk: bytes,
        pinned: Optional[bytes] = None,
        eph: Optional[bytes] = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.psk = psk
        self.pinned = pinned
        self.eph = eph
        self.sc: Optional[SecureChannel] = None

    @classmethod
    async def connect(cls, host: str, port: int) -> "RobotLink":
        """Dial the brain and complete the handshake with the link key from
        `AZOS_BRAIN_LINK_KEY` (the key the brain was started with)."""
        psk = load_link_key()
        if psk is None:
            raise RuntimeError(
                "AZOS_BRAIN_LINK_KEY is not set: the brain link is always a "
                "Noise session, set the same 64-hex-digit key the brain runs with"
            )
        reader, writer = await asyncio.open_connection(host, port)
        link = cls(reader, writer, psk=psk)
        await link.handshake()
        return link

    async def handshake(self) -> SecureChannel:
        """Run the initiator handshake. Raises `LinkViolation` when the brain's
        answer is refused, `IncompleteReadError` when the brain hangs up."""
        sc = SecureChannel(self.psk, is_initiator=True, remote_pinned=self.pinned, _testing_eph_priv=self.eph)
        self.sc = sc
        self.writer.write(sc.start_handshake())
        await self.writer.drain()
        while not sc.is_established:
            head = await self.reader.readexactly(FRAME_HEAD)
            frame = head + await self.reader.readexactly(handshake_frame_size(head) - FRAME_HEAD)
            out = sc.handle_handshake(frame)
            if out:
                self.writer.write(out)
                await self.writer.drain()
        return sc

    def queue_packet(self, pkt_type: int, payload: bytes) -> None:
        """Seal one brain packet and queue it on the writer."""
        assert self.sc is not None
        self.writer.write(self.sc.seal_message(protocol.build_packet(pkt_type, payload)))

    async def send_message(self, inner: bytes) -> None:
        """Seal one brain frame as one message."""
        assert self.sc is not None
        self.writer.write(self.sc.seal_message(inner))
        await self.writer.drain()

    async def send_packet(self, pkt_type: int, payload: bytes) -> None:
        self.queue_packet(pkt_type, payload)
        await self.writer.drain()

    async def read_message(self) -> bytes:
        """Read frames until one data frame; returns its plaintext (a brain frame)."""
        assert self.sc is not None
        while True:
            head = await self.reader.readexactly(FRAME_HEAD)
            body = await self.reader.readexactly(self.sc.frame_size(head) - FRAME_HEAD)
            kind, plain = self.sc.open_frame(head + body)
            if kind == SecureChannel.RECORD_DATA:
                return plain

    async def read_packet(self) -> Optional[tuple[int, bytes]]:
        return protocol.parse_packet(await self.read_message())

    def close(self) -> None:
        self.writer.close()

    async def wait_closed(self) -> None:
        await self.writer.wait_closed()
