# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""C1 camera connection wiring in `server.py`: where a CAMERA_FRAME goes.

Built on a partial BrainServer (the tests/test_degrade.py pattern), with the
decoder and the perception cycle replaced by recorders. Not covered here:
`handle_robot` setting and clearing `_control_peer` and closing the camera
connection, and `run()` binding the camera port (no harness drives either).
"""

import asyncio

from protocol import CAMERA_FRAME
from server import BrainServer, RobotState


class _FakeWriter:
    def write(self, data):
        pass

    async def drain(self):
        pass


def _make_server():
    srv = BrainServer.__new__(BrainServer)
    srv.state = RobotState()
    srv._writer = None
    srv.cycles = []
    srv._decode_camera = lambda payload: payload

    async def _record_cycle(image_data, writer):
        srv.cycles.append((image_data, writer))

    srv._perception_cycle = _record_cycle
    return srv


def test_a_camera_connection_frame_without_control_is_dropped():
    """Canary: drop the `_writer is None` guard in `_on_camera_frame` — the
    cycle runs with no control writer."""
    srv = _make_server()
    asyncio.run(srv._on_camera_frame(b"jpeg"))
    assert srv.cycles == []
    assert srv.state.last_image == b""


def test_a_camera_connection_frame_commands_the_control_connection():
    """Canary: hand the cycle no writer — the recorded writer is not the
    control connection's."""
    srv = _make_server()
    control = _FakeWriter()
    srv._writer = control
    asyncio.run(srv._on_camera_frame(b"jpeg"))
    assert srv.cycles == [(b"jpeg", control)]
    assert srv.state.last_image == b"jpeg"


def test_a_frame_on_the_control_connection_still_runs_the_cycle():
    """Frames keep working on the control connection until the kernel moves
    them. Canary: drop the CAMERA_FRAME arm of `_dispatch`."""
    srv = _make_server()
    control = _FakeWriter()
    asyncio.run(srv._dispatch(CAMERA_FRAME, b"jpeg", control))
    assert srv.cycles == [(b"jpeg", control)]
