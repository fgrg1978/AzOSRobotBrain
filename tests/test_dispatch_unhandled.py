# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0039 step 1 — brain-side half: "PKT_CAPS reserved and undeclared types
recorded". Only that slice; the handshake itself (steps 2-4) is out of scope.

Before this, `server.py`'s `_dispatch` was an if/elif chain over
SENSOR_PACKET/CAMERA_FRAME/STATUS with no final `else`. OTA_ACK and
SENSOR_COMPACT are declared packet-type constants in protocol.py and listed
as inbound in BRAIN.TOML, but neither ever had a branch in `_dispatch` — they
fell through and vanished with no log line and no counter, indistinguishable
from a packet the brain actually acted on.

These tests exercise both halves of that fix:
  * an undeclared/unhandled type produces the record (log line + ignored
    counter), and
  * a declared, handled type produces NO such record — a test with only the
    positive half would also pass on code that records every packet
    regardless of whether it was handled.
"""

import asyncio
import logging

import pytest

import metrics
import protocol
from protocol import OTA_ACK, ROBOT_WHEELED, SENSOR_COMPACT, STATUS, StatusPacket
from server import BrainServer, RobotState


def _make_dispatch_server() -> BrainServer:
    """Build a BrainServer skipping the heavy __init__ (which needs a full
    config.yaml + LM Studio + a dozen subsystems), wired with just enough
    state for `_dispatch`'s STATUS branch — the same pattern already used by
    tests/test_degrade.py for `_perception_cycle`.
    """
    srv = BrainServer.__new__(BrainServer)
    srv.robot_type = ROBOT_WHEELED
    srv._configured_robot_type = ROBOT_WHEELED
    srv.state = RobotState()
    return srv


class _FakeWriter:
    """`_dispatch` takes a writer for the branches that may reply; the
    STATUS and unhandled-type paths exercised here never write to it."""

    def write(self, data):
        raise AssertionError("this test's packet types must never write to the wire")

    async def drain(self):
        pass


def _ignored_count(pkt_type: int) -> float:
    return metrics.M.pkt_dispatch_ignored_total.labels(type=f"0x{pkt_type:02x}").value


def _handled_count(pkt_type: int) -> float:
    return metrics.M.pkt_dispatch_handled_total.labels(type=f"0x{pkt_type:02x}").value


# ── Positive half: an unhandled type IS recorded ─────────────────────────────


@pytest.mark.parametrize(
    "pkt_type,payload",
    [
        (OTA_ACK, b"\x00\x01\x00"),  # declared in protocol.py, no _dispatch branch
        (SENSOR_COMPACT, b"\x00" * 20),  # declared in protocol.py, no _dispatch branch
        (0x7F, b"\xaa\xbb"),  # not a declared type at all — same fate
    ],
)
def test_unhandled_type_produces_the_record(pkt_type, payload, caplog):
    srv = _make_dispatch_server()
    writer = _FakeWriter()

    before_ignored = _ignored_count(pkt_type)
    before_handled = _handled_count(pkt_type)

    with caplog.at_level(logging.WARNING, logger="brain.server"):
        asyncio.run(srv._dispatch(pkt_type, payload, writer))

    # The counter that separates "handled" from "ignored" moved on the
    # ignored side only.
    assert _ignored_count(pkt_type) == before_ignored + 1
    assert _handled_count(pkt_type) == before_handled

    # A log line visible at the default level, naming the type in hex and
    # the payload length — the two facts an operator needs to tell this
    # apart from a handled packet.
    records = [r for r in caplog.records if "Unhandled inbound packet type" in r.message]
    assert len(records) == 1, f"expected exactly one record, got: {caplog.records}"
    assert records[0].levelno == logging.WARNING
    assert f"0x{pkt_type:02x}" in records[0].message
    assert str(len(payload)) in records[0].message


# ── Negative half: a declared, HANDLED type produces NO such record ─────────


def test_handled_type_produces_no_unhandled_record(caplog):
    srv = _make_dispatch_server()
    writer = _FakeWriter()

    pkt = StatusPacket(mode=0, tasks_ok=1, canary_ok=1, uptime_s=1, robot_type=ROBOT_WHEELED)
    payload = pkt.to_bytes()

    before_ignored = _ignored_count(STATUS)
    before_handled = _handled_count(STATUS)

    with caplog.at_level(logging.WARNING, logger="brain.server"):
        asyncio.run(srv._dispatch(STATUS, payload, writer))

    # Handled counter moved, ignored counter did not — a canary that removes
    # the `else` branch entirely (recording everything, handled included)
    # would fail this half even though it still passes the positive tests
    # above.
    assert _handled_count(STATUS) == before_handled + 1
    assert _ignored_count(STATUS) == before_ignored

    unhandled_records = [r for r in caplog.records if "Unhandled inbound packet type" in r.message]
    assert unhandled_records == []

    # The packet was actually processed, not just silently accepted.
    assert srv.state.status["mode"] == 0
    assert srv.state.status["uptime_s"] == 1


def test_sensor_packet_and_camera_frame_are_still_declared_types():
    """Sanity check on the fixture data above: OTA_ACK/SENSOR_COMPACT really
    are declared protocol constants (RFC-0039's whole point is that
    "declared" and "acted on" are different things), and 0x7F really is
    outside the current type space — both must hit the new `else`, not an
    accidental match on an existing branch.
    """
    assert isinstance(protocol.OTA_ACK, int)
    assert isinstance(protocol.SENSOR_COMPACT, int)
    assert 0x7F not in (
        protocol.SENSOR_PACKET,
        protocol.CAMERA_FRAME,
        protocol.STATUS,
        protocol.OTA_ACK,
        protocol.SENSOR_COMPACT,
        protocol.CAPS,
    )
