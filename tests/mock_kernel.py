# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""A mock robot (the kernel's side of the brain link) over real loopback streams.

The robot dials the brain, so it is the TCP client and the Noise initiator;
the tests start the brain's code (`protocol.perform_handshake` and friends) as
the server handler and drive this class from the client end. Every message it
seals or opens is one brain-protocol frame: the Noise session is the only
layer.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import protocol
from robot_link import RobotLink

PSK = bytes(range(32))  # 32-byte test key
HEXKEY = PSK.hex()  # 64 hex chars for AZOS_BRAIN_LINK_KEY


class MockRobot(RobotLink):
    """The robot end of one connection, with the test key by default."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        psk: bytes = PSK,
        pinned: Optional[bytes] = None,
        eph: Optional[bytes] = None,
    ) -> None:
        super().__init__(reader, writer, psk=psk, pinned=pinned, eph=eph)


def arm_brain(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Arm the brain's encrypted link from the env."""
    monkeypatch.setenv("AZOS_BRAIN_LINK_KEY", HEXKEY)
    assert protocol.enable_encrypt_link() is True


def reset_protocol_state() -> None:
    """protocol.py keeps process-wide arming state and a registry of
    per-connection links. Reset it around every test so state doesn't leak."""
    protocol._encrypt_psk = None
    protocol._noise_static = None
    protocol._links_by_reader.clear()
    protocol._links_by_writer.clear()
    protocol._recent_session_ids.clear()
