# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for data_plane/main.py — robot_id derivation on the first packet.

Regression coverage for the bug where ``_extract_robot_id`` (now
``_robot_id_from_peer``) hand-rolled the wire framing and assumed a STATUS/
SENSOR payload was JSON with a ``robot_id`` key.  It never is: STATUS is a
7-8 byte packed struct (``protocol.StatusPacket``) and SENSOR is ~20-68
packed bytes, so ``json.loads`` raised on every real packet and the
exception was swallowed, silently landing on the peer-address fallback
anyway.  ``protocol.py`` carries no ``robot_id`` field in any packet type,
and ``server.py``'s real ``BrainServer`` independently derives its id the
same way from the peer address, so the fix makes that the sole, deliberate
source instead of an accidental catch-all — joined with ``_`` rather than
``BrainServer``'s ``:``, because this id also travels as an HTTP path
segment to the control plane and its route regex has no colon in the
allowed character class (see ``control_plane/main.py``).
"""

from __future__ import annotations

import asyncio
import json

import protocol
from control_plane.main import _ROUTE_PATH_RE
from data_plane.main import DataPlaneServer, _robot_id_from_peer


# ---------------------------------------------------------------------------
# Positive: a real frame built by protocol.py's own encoder
# ---------------------------------------------------------------------------


def test_robot_id_from_peer_uses_peer_address_for_real_status_frame() -> None:
    """A genuine STATUS frame yields the peer-address id, not packet content.

    This is the direct reproduction of the reported bug: build the frame with
    protocol.py's own StatusPacket + build_packet (no hand-rolled framing),
    confirm the payload is not JSON, then confirm the extractor's output
    depends only on the peer tuple.
    """
    status = protocol.StatusPacket(mode=1, tasks_ok=1, canary_ok=1, uptime_s=42)
    payload = status.to_bytes()

    # The payload is packed binary, not JSON — this is the premise of the bug.
    assert len(payload) in (7, 8)
    try:
        json.loads(payload)
        raised = False
    except (json.JSONDecodeError, UnicodeDecodeError):
        raised = True
    assert raised, "StatusPacket payload decoded as JSON — protocol has changed"

    frame = protocol.build_packet(protocol.STATUS, payload)

    # protocol.py itself can parse the frame it just built.
    parsed = protocol.parse_packet(frame)
    assert parsed is not None
    pkt_type, parsed_payload = parsed
    assert pkt_type == protocol.STATUS
    assert parsed_payload == payload

    peer = ("10.0.0.5", 54321)
    robot_id = _robot_id_from_peer(frame, peer)
    assert robot_id == "10.0.0.5_54321"


def test_robot_id_from_peer_uses_peer_address_for_real_sensor_frame() -> None:
    """Same story for a real SENSOR_PACKET frame (wheeled telemetry)."""
    sensor = protocol.SensorPacket(
        timestamp_ms=1000,
        battery_mv=8000,
        accel_mg=(0, 0, 1000),
        gyro_mdps=(0, 0, 0),
        odom_dist_mm=0,
        odom_hdg_cdeg=0,
        encoder_l=0,
        encoder_r=0,
        range_front_mm=0,
        range_right_mm=0,
    )
    payload = sensor.to_bytes()
    frame = protocol.build_packet(protocol.SENSOR_PACKET, payload)

    parsed = protocol.parse_packet(frame)
    assert parsed is not None
    assert parsed[0] == protocol.SENSOR_PACKET

    peer = ("192.168.1.7", 9100)
    robot_id = _robot_id_from_peer(frame, peer)
    assert robot_id == "192.168.1.7_9100"


def test_robot_id_never_reads_a_json_robot_id_key() -> None:
    """Even a frame whose payload happens to be valid JSON with robot_id must
    not influence the returned id — the protocol has no robot_id field and
    the code must not go looking for one via a JSON detour.
    """
    fake_payload = json.dumps({"robot_id": "attacker_supplied_id"}).encode()
    frame = protocol.build_packet(protocol.STATUS, fake_payload)

    peer = ("203.0.113.9", 4444)
    robot_id = _robot_id_from_peer(frame, peer)
    assert robot_id == "203.0.113.9_4444"
    assert "attacker_supplied_id" not in robot_id


# ---------------------------------------------------------------------------
# Negative: malformed / truncated input must not produce a plausible-looking
# id derived from garbage payload content — it must fall through to the same
# deliberate peer-address derivation, indistinguishably from the well-formed
# case.
# ---------------------------------------------------------------------------


def test_robot_id_from_peer_truncated_frame_still_uses_peer_address() -> None:
    status = protocol.StatusPacket(mode=0, tasks_ok=0, canary_ok=0, uptime_s=0)
    frame = protocol.build_packet(protocol.STATUS, status.to_bytes())
    truncated = frame[:4]  # shorter than the 5-byte header

    assert protocol.parse_packet(truncated) is None

    peer = ("10.0.0.9", 1)
    robot_id = _robot_id_from_peer(truncated, peer)
    assert robot_id == "10.0.0.9_1"


def test_robot_id_from_peer_garbage_bytes_still_uses_peer_address() -> None:
    garbage = b"\x00\x01\x02not-a-protocol-frame-at-all"

    assert protocol.parse_packet(garbage) is None

    peer = ("172.16.0.2", 2222)
    robot_id = _robot_id_from_peer(garbage, peer)
    assert robot_id == "172.16.0.2_2222"


def test_robot_id_from_peer_empty_frame_still_uses_peer_address() -> None:
    peer = ("127.0.0.1", 9999)
    robot_id = _robot_id_from_peer(b"", peer)
    assert robot_id == "127.0.0.1_9999"


def test_robot_id_from_peer_is_consistent_regardless_of_frame_validity() -> None:
    """Well-formed and malformed input for the same peer must yield the exact
    same id — proof that packet content never leaks into the result, so a
    malformed frame can't produce an id that merely *looks* legitimate while
    actually being derived from attacker-controlled bytes.
    """
    peer = ("10.1.1.1", 5555)
    status = protocol.StatusPacket(mode=2, tasks_ok=1, canary_ok=1, uptime_s=7)
    good_frame = protocol.build_packet(protocol.STATUS, status.to_bytes())
    bad_frame = b"garbage"

    assert _robot_id_from_peer(good_frame, peer) == _robot_id_from_peer(bad_frame, peer)


def test_robot_id_from_peer_is_a_valid_control_plane_route_segment() -> None:
    """The id must survive the trip through worker.py's HTTP ownership query.

    ``OwnershipChecker._check_http`` (data_plane/worker.py) formats robot_id
    unescaped into ``/v1/robots/{robot_id}/route`` and control_plane/main.py
    matches that path with ``_ROUTE_PATH_RE`` = ``[A-Za-z0-9_.\\-]+`` — no
    colon.  A ``host:port`` id (matching ``BrainServer``'s own separator)
    would silently 404 every multi-process ownership query.  IPv4 peers are
    the only case this module ever sees in practice (the listener binds
    loopback; see DP_DEFAULT_BIND_HOST), so this is the whole domain today.
    """
    peer = ("10.0.0.5", 54321)
    robot_id = _robot_id_from_peer(b"", peer)
    path = f"/v1/robots/{robot_id}/route"
    match = _ROUTE_PATH_RE.match(path)
    assert match is not None, f"robot_id {robot_id!r} does not survive the CP route regex"
    assert match.group("robot_id") == robot_id


# ---------------------------------------------------------------------------
# `_handle` on ownership rejection — regression coverage for the reported
# finding that the old `_send_reject` wrote raw, unframed `json.dumps(...)`
# bytes onto the same socket a real kernel speaks the framed `protocol.py`
# format on (MAGIC | TYPE | LEN | PAYLOAD | CRC8; see the module docstring
# and `protocol.py`'s own header).
#
# Investigation settled two things:
#   1. `_send_reject` DID reach a robot's link: `_handle` is the sole
#      connection handler for the data-plane listener, so the `writer` it
#      wrote to before closing is the exact socket a kernel's TCP client
#      connects on. This was not a different, internal-only channel.
#   2. It could not desynchronise that link. `../AzOS`'s
#      `kernel/src/main.rs` TCP receive loop (~line 5716-5730) resyncs a
#      failed `parse_packet` by scanning forward for the next `MAGIC` pair
#      ("CL") and, finding none, simply `break`s — discarding the rest of
#      that `recv()`'s buffer. `raw_buf`/`recv_buf` are declared fresh
#      inside the per-tick loop body (~line 5626-5629), so no partial-frame
#      state survives across `recv()` calls or a reconnect for a later
#      genuine frame to inherit. And `_handle`'s `finally` closes the
#      socket immediately after the reject write, so there was never a
#      "next frame" on that connection to misparse in the first place.
#
# So the honest fix (per the task) is not a new wire packet — it's to stop
# writing anything at all: there was no packet type for "not owner", and
# nothing in this repo ever read the JSON back (grep for `not_owner` /
# `_send_reject` outside this module and its own docstring turns up
# nothing). The tests below pin that down.
# ---------------------------------------------------------------------------


class _StubChecker:
    """Stand-in for `OwnershipChecker` that returns a fixed verdict."""

    def __init__(self, owns: bool) -> None:
        self._owns = owns

    async def check_ownership(self, robot_id: str) -> bool:  # noqa: ARG002
        return self._owns


class _FakeWriter:
    """Minimal stand-in for `asyncio.StreamWriter` — records what `_handle`
    does with it without opening a real socket."""

    def __init__(self, peer: tuple[str, int]) -> None:
        self._peer = peer
        self.written: list[bytes] = []
        self.closed = False

    def get_extra_info(self, name: str, default=None):
        if name == "peername":
            return self._peer
        return default

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))

    def close(self) -> None:
        self.closed = True


def _fed_reader(data: bytes) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with `data` followed by EOF, so
    `_handle`'s `reader.read(...)` returns it without a real connection.

    Must be called from inside a running event loop — `StreamReader.__init__`
    looks one up.
    """
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def test_ownership_rejection_writes_nothing_to_the_wire() -> None:
    """When `check_ownership` refuses a robot, `_handle` must write zero
    bytes to the peer before closing — the regression this task exists for.
    """

    async def _pipeline_must_not_run(robot_id, reader, writer) -> None:  # noqa: ARG001
        raise AssertionError("pipeline_fn must not run for a refused robot_id")

    async def _run() -> _FakeWriter:
        server = DataPlaneServer(pipeline_fn=_pipeline_must_not_run)
        server._checker = _StubChecker(owns=False)  # bypass real ring/HTTP lookup

        status = protocol.StatusPacket(mode=0, tasks_ok=1, canary_ok=1, uptime_s=1)
        frame = protocol.build_packet(protocol.STATUS, status.to_bytes())
        reader = _fed_reader(frame)
        writer = _FakeWriter(("10.0.0.9", 9100))

        await server._handle(reader, writer)
        return writer

    writer = asyncio.run(_run())

    assert writer.written == [], (
        f"reject path wrote {writer.written!r} to the wire; a real kernel's "
        "connection expects only framed protocol.py bytes on this socket"
    )
    assert writer.closed, "a refused connection must still be closed"


def test_ownership_acceptance_still_hands_prefixed_bytes_to_pipeline() -> None:
    """Removing the reject write must not disturb the accept path: an owned
    robot's connection (and its already-peeked first frame) must still reach
    the pipeline unchanged."""
    seen: dict[str, object] = {}

    async def _pipeline_fn(robot_id: str, reader, writer) -> None:
        seen["robot_id"] = robot_id
        seen["first_bytes"] = await reader.read(4096)
        seen["writer_is_same"] = writer

    async def _run() -> _FakeWriter:
        server = DataPlaneServer(pipeline_fn=_pipeline_fn)
        server._checker = _StubChecker(owns=True)

        status = protocol.StatusPacket(mode=0, tasks_ok=1, canary_ok=1, uptime_s=1)
        frame = protocol.build_packet(protocol.STATUS, status.to_bytes())
        reader = _fed_reader(frame)
        writer = _FakeWriter(("10.0.0.9", 9100))

        await server._handle(reader, writer)
        seen["frame"] = frame
        return writer

    writer = asyncio.run(_run())

    assert seen["robot_id"] == "10.0.0.9_9100"
    assert seen["first_bytes"] == seen["frame"], "the peeked first frame must reach the pipeline intact"
    assert seen["writer_is_same"] is writer
    assert writer.written == []  # the pipeline (not `_handle`) owns writes on accept


def test_old_reject_body_could_not_have_been_mistaken_for_a_frame() -> None:
    """Documents *why* the removed `_send_reject` payload was harmless even
    while it did reach a robot's link: it never matched the frame format a
    real kernel resyncs on.

    `kernel/src/main.rs`'s receive loop treats a failed `parse_packet` as
    "scan forward for the next MAGIC pair"; it can only be misled by bytes
    that happen to contain "CL" (0x43 0x4c). The old reject body was
    `json.dumps({"error": "not_owner", "robot_id": f"{ip}_{port}"})` —
    lowercase JSON keys/values plus IPv4 digits and underscores, never the
    two-byte sequence "CL". This test would fail (flagging a genuine
    resync-desync bug) if a future robot_id format ever introduced one.
    """
    for robot_id in ("10.0.0.5_54321", "127.0.0.1_1", "203.0.113.9_65535"):
        body = json.dumps({"error": "not_owner", "robot_id": robot_id}).encode()
        assert b"CL" not in body, (
            f"reject body for robot_id={robot_id!r} contains the frame MAGIC "
            "— it could desynchronise a kernel's resync scan"
        )
        # And it never parsed as a frame at all, regardless of MAGIC:
        assert protocol.parse_packet(body) is None
