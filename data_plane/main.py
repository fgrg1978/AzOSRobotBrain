# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Data-plane TCP server.

Accepts kernel connections on a configurable port.  For each new connection:
  1. Reads the first packet (peeking at it with ``protocol.parse_packet`` only
     for logging) and derives ``robot_id`` from the peer address.  The wire
     protocol (see ``protocol.py``) has no robot_id field in any packet type —
     not STATUS, not any SENSOR variant — so there is nothing to extract; see
     ``_robot_id_from_peer`` below.
  2. Calls ``OwnershipChecker.check_ownership(robot_id)``.
  3. If NOT owner → logs the refusal and closes the connection.  Nothing is
     written to the wire: this socket is the same one a real kernel speaks
     the framed ``protocol.py`` format on (MAGIC | TYPE | LEN | PAYLOAD |
     CRC8), and there is no packet type for "not owner" — adding one is an
     owner decision that needs coordination with the OS repo (AzOS
     ``crates/behavior/src/brain_protocol.rs``). An earlier version wrote raw
     ``json.dumps(...)`` bytes here with no framing at all; see
     ``test_data_plane_main.py`` for why that was harmless (the kernel's
     resync scan never finds "BR" in it and the connection is already
     closing) but still worth removing — it had zero consumers on either
     side and violated the "one wire format" invariant documented in
     ``protocol.py``.
  4. If owner → hands the connection to the pipeline (perception / planner /
     policy).  In single-process mode this delegates to the existing
     ``BrainServer`` connection handler.  In stub mode (multi-process without
     a wired pipeline), the robot_id is logged and the connection is left
     open until the kernel disconnects.

Security status (READ THIS BEFORE WIRING A REAL PIPELINE):
  There is no peer authentication here.  ``robot_id`` is derived from the
  peer's TCP source address (see ``_robot_id_from_peer`` — the protocol
  carries no identity field a peer could declare instead), and
  ``check_ownership`` only decides which shard should handle that id — it is
  routing, not identity.  Nothing on this path proves the connection belongs
  to the robot whose address it used: reaching a given robot_id requires
  connecting from that source address, not just claiming a value, but
  nothing stronger than that address is ever checked.  The exposure is
  currently bounded by two things and neither is a substitute for
  authentication: the listener binds loopback by default
  (``DP_DEFAULT_BIND_HOST``), and ``pipeline_fn`` is never wired to anything
  but ``_pipeline_stub``, so no actuation hangs off it.  Wiring a real
  ``pipeline_fn`` — especially ``BrainServer.handle_robot`` — makes this an
  actuation surface, and an authentication step (shared link key, mTLS, or a
  per-worker token) must land first.

Pipeline integration:
  The existing ``BrainServer`` logic in ``server.py`` is NOT imported at
  module load time.  It is imported lazily inside ``_pipeline_stub`` so that
  unit tests for the data plane never trigger LM Studio / YAML / vision
  imports.  When a real pipeline is wired (full multi-process deployment),
  supply a ``pipeline_fn`` coroutine to the ``DataPlaneServer`` constructor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

import protocol
from data_plane.worker import OwnershipChecker
from data_plane import CP_INPROCESS_SENTINEL
from control_plane.discovery import ShardCoordinator

logger = logging.getLogger("brain.data_plane")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Default TCP port for data-plane kernel connections.
DP_DEFAULT_PORT: int = 9100

#: Maximum bytes for the first packet read (peeked at for logging before
#: robot_id — which comes from the peer address, not the packet — is used).
DP_FIRST_PACKET_MAX_BYTES: int = 512

#: Timeout in seconds waiting for the first packet from a new kernel connection.
DP_FIRST_PACKET_TIMEOUT_S: float = 5.0

#: Default bind address.
#:
#: B-A16 (partial fix): this server performs NO authentication of the peer.
#: ``_handle`` takes ``robot_id`` straight from the peer's TCP address (the
#: wire protocol carries no robot_id field to take it from instead), and
#: ``OwnershipChecker.check_ownership`` is shard routing, not authentication —
#: it answers "is this robot mine?", never "is this really that robot?".
#: Binding that to 0.0.0.0, as this did unconditionally, let any LAN peer's
#: source address decide which shard's data it lands on.  Until a real
#: peer-authentication step exists (see the
#: module docstring), keep the socket on loopback so the exposure is bounded
#: to this host; pass ``bind_host`` explicitly if the deployment terminates
#: authentication in front of this process.
DP_DEFAULT_BIND_HOST: str = "127.0.0.1"


# Type alias for an optional custom pipeline coroutine.
PipelineFn = Callable[
    [str, asyncio.StreamReader, asyncio.StreamWriter],
    Coroutine[Any, Any, None],
]


class DataPlaneServer:
    """Async TCP server that gates kernel connections by ownership.

    Args:
        self_node:      This worker's own address string (e.g. "localhost:9100").
                        Use ``CP_INPROCESS_SENTINEL`` for single-process mode.
        coordinator:    Local ``ShardCoordinator`` (single-process mode).
        cp_address:     Control-plane address for HTTP ownership queries
                        (multi-process mode).
        cp_api_key:     Bearer token for control-plane ownership queries.
                        Defaults to ``AZOS_BRAIN_CP_API_KEY`` from the
                        environment (see ``OwnershipChecker``).
        pipeline_fn:    Optional coroutine called with (robot_id, reader, writer)
                        after ownership is confirmed.  Defaults to a stub that
                        drains the connection.
        port:           TCP port to listen on.
        bind_host:      Interface to bind.  Defaults to loopback — see
                        ``DP_DEFAULT_BIND_HOST`` for why.
    """

    def __init__(
        self,
        *,
        self_node: str = CP_INPROCESS_SENTINEL,
        coordinator: Optional[ShardCoordinator] = None,
        cp_address: Optional[str] = None,
        cp_api_key: Optional[str] = None,
        pipeline_fn: Optional[PipelineFn] = None,
        port: int = DP_DEFAULT_PORT,
        bind_host: str = DP_DEFAULT_BIND_HOST,
    ) -> None:
        self._checker = OwnershipChecker(
            self_node,
            coordinator=coordinator,
            cp_address=cp_address,
            cp_api_key=cp_api_key,
        )
        self._pipeline_fn = pipeline_fn or _pipeline_stub
        self._port = port
        self._bind_host = bind_host
        self._self_node = self_node

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        server = await asyncio.start_server(self._handle, self._bind_host, self._port)
        logger.warning(
            "[DataPlane] node=%s listening on %s:%d — connections are NOT "
            "authenticated; robot_id is taken from the peer's TCP address",
            self._self_node,
            self._bind_host,
            self._port,
        )
        async with server:
            await server.serve_forever()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", ("?", 0))
        logger.debug("[DataPlane] new connection from %s:%s", peer[0], peer[1])
        try:
            raw = await asyncio.wait_for(
                reader.read(DP_FIRST_PACKET_MAX_BYTES),
                timeout=DP_FIRST_PACKET_TIMEOUT_S,
            )
            # NOTE: robot_id here is derived from the peer's TCP address, not
            # from anything the peer asserts about itself; check_ownership
            # below is shard routing, not authentication.
            # See the "Security status" section of the module docstring.
            robot_id = _robot_id_from_peer(raw, peer)
            owns = await self._checker.check_ownership(robot_id)
            if not owns:
                logger.info("[DataPlane] refusing robot_id=%s (not our shard)", robot_id)
                # Nothing is written to the peer: this socket is the same one
                # a real kernel speaks the framed protocol.py format on, and
                # there is no packet type for "not owner". A previous version
                # wrote raw json.dumps(...) bytes here with no MAGIC/LEN/CRC
                # framing at all — see the module docstring and
                # test_data_plane_main.py::test_ownership_rejection_writes_nothing_to_the_wire
                # for why that was reachable by a real kernel's connection but
                # harmless (the kernel's resync scan never finds "BR" in it,
                # and this `finally` closes the socket before any subsequent
                # frame could be misparsed). Close is still the right way to
                # signal "not for you" absent a defined error packet.
                return
            logger.info("[DataPlane] accepted robot_id=%s", robot_id)
            # Prepend the already-read bytes back for the pipeline.
            combined_reader = _PrefixReader(raw, reader)
            await self._pipeline_fn(robot_id, combined_reader, writer)  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            logger.warning("[DataPlane] timeout waiting for first packet from %s", peer)
        except Exception as exc:
            logger.error("[DataPlane] connection error from %s: %s", peer, exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _robot_id_from_peer(
    raw: bytes,
    peer: tuple[str, int],
) -> str:
    """Derive robot_id from the peer's TCP address.

    This is not a fallback for a rare malformed-packet case — it is the only
    source of a robot_id this protocol has.  ``protocol.py`` defines every
    packet type the kernel sends (SENSOR_PACKET, CAMERA_FRAME, STATUS,
    OTA_ACK, SENSOR_COMPACT, and the reserved CAPS handshake) and none of
    them, in any variant, carries a robot_id field: ``StatusPacket`` is
    ``mode, tasks_ok, canary_ok, uptime_s, robot_type``; every ``SensorPacket*``
    is timestamps, IMU, and chassis-specific telemetry.  There was never a
    JSON body to parse — a STATUS payload is 7-8 packed bytes, a SENSOR
    payload ~20-68 packed bytes, and ``json.loads`` on either raises every
    time.  ``BrainServer`` in ``server.py`` reaches the same conclusion
    independently: it provisionally IDs a connection from the peer address
    too (``f"{addr[0]}:{addr[1]}"``) and only ever refines the robot_type,
    never the id, from a later STATUS packet.

    The separator here is ``_``, not ``:`` like ``BrainServer`` uses, even
    though both name the same thing: this id is also sent verbatim as an
    HTTP path segment to the control plane's ``/v1/robots/{id}/route`` (see
    ``worker.py``), whose route regex is ``[A-Za-z0-9_.\\-]+`` — no colon.
    A colon-joined id would silently 404 every ownership query in
    multi-process mode.

    ``raw`` is accepted only so a well-formed first frame can be logged for
    diagnostics — it never contributes to the returned id.
    """
    parsed = protocol.parse_packet(raw)
    if parsed is not None:
        pkt_type, payload = parsed
        logger.debug(
            "[DataPlane] first packet from %s:%s is pkt_type=0x%02x (%d byte payload); "
            "robot_id is taken from the peer address regardless — the protocol "
            "carries no robot_id field",
            peer[0],
            peer[1],
            pkt_type,
            len(payload),
        )
    else:
        logger.debug(
            "[DataPlane] first %d byte(s) from %s:%s do not parse as a protocol.py "
            "frame; robot_id is taken from the peer address",
            len(raw),
            peer[0],
            peer[1],
        )
    return f"{peer[0]}_{peer[1]}"


async def _pipeline_stub(
    robot_id: str,
    reader: asyncio.StreamReader,  # type: ignore[type-arg]
    writer: asyncio.StreamWriter,
) -> None:
    """Default pipeline: drain the connection and log.

    Replace with a real pipeline coroutine in production deployments by
    supplying ``pipeline_fn`` to ``DataPlaneServer``.

    In single-process mode the caller wires the existing ``BrainServer``
    connection handler here so the full perception/planner/policy stack runs.
    """
    logger.info("[DataPlane/stub] pipeline for robot_id=%s (draining)", robot_id)
    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
    except Exception:
        pass


class _PrefixReader:
    """Wraps an asyncio.StreamReader, prepending already-read bytes.

    This lets us hand the pipeline the full byte stream including the first
    packet we already read while determining robot_id (see
    ``_robot_id_from_peer`` — the read is for diagnostics, the id itself
    comes from the peer address either way).
    """

    def __init__(self, prefix: bytes, inner: asyncio.StreamReader) -> None:
        self._buf = bytearray(prefix)
        self._inner = inner

    async def read(self, n: int = -1) -> bytes:
        if self._buf:
            if n > 0:
                chunk = bytes(self._buf[:n])
                del self._buf[:n]
            else:
                chunk = bytes(self._buf)
                del self._buf[:]
            return chunk
        return await self._inner.read(n)

    # Proxy other StreamReader methods the pipeline may need.
    async def readexactly(self, n: int) -> bytes:
        result = bytearray()
        while len(result) < n:
            need = n - len(result)
            chunk = await self.read(need)
            if not chunk:
                raise asyncio.IncompleteReadError(bytes(result), n)
            result.extend(chunk)
        return bytes(result)


# ---------------------------------------------------------------------------
# Standalone entry-point (``python -m data_plane.main``)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> "argparse.ArgumentParser":  # type: ignore[name-defined]
    import argparse

    p = argparse.ArgumentParser(description="AzOSRobotBrain — data plane worker")
    p.add_argument(
        "--port", type=int, default=DP_DEFAULT_PORT, help="TCP port for kernel connections"
    )
    p.add_argument(
        "--self-node",
        type=str,
        default=CP_INPROCESS_SENTINEL,
        help="This worker's address (host:port)",
    )
    p.add_argument(
        "--cp-address",
        type=str,
        default=None,
        help="Control-plane address (host:port) for ownership queries",
    )
    p.add_argument(
        "--bind-host",
        type=str,
        default=DP_DEFAULT_BIND_HOST,
        help=(
            "Interface to bind (default: %(default)s). Connections are NOT "
            "authenticated — only widen this if something in front of this "
            "process authenticates the peer."
        ),
    )
    return p


def main() -> None:
    import argparse

    args = _build_arg_parser().parse_args()
    coord: Optional[ShardCoordinator] = None
    if args.self_node == CP_INPROCESS_SENTINEL:
        coord = ShardCoordinator()
    server = DataPlaneServer(
        self_node=args.self_node,
        coordinator=coord,
        cp_address=args.cp_address,
        port=args.port,
        bind_host=args.bind_host,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
