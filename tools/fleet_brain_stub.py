#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Companion brain stub for fleet-scale soak testing.

Accepts up to NMAX concurrent fake-kernel connections, reads every packet,
and (optionally) replies with an ActuatorCmd zero on each SensorPacket so
the load generator can measure RTT.

NOT the production brain — no LLM, no vision, no policy.  This is the
harness used by fleet_soak.py and tests/test_fleet_soak.py.

Usage:
    python3 tools/fleet_brain_stub.py --port 9100 --nmax 5000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import struct
import sys
import time
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import protocol  # noqa: E402
from protocol import (  # noqa: E402
    MAGIC,
    SENSOR_PACKET,
    ACTUATOR_CMD,
    build_packet,
    ActuatorCmd,
)

# ── Constants (no magic numbers) ──────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
DEFAULT_NMAX = 5000
DEFAULT_REPLY_EVERY = 1  # send ActuatorCmd for every Nth SensorPacket
DEFAULT_REPORT_INTERVAL = 1.0  # seconds between stdout status lines
READ_CHUNK_BYTES = 65536
PACKET_HEADER_BYTES = 5  # MAGIC(2) + TYPE(1) + LEN(2)
PACKET_CRC_BYTES = 1

# Zero actuator cmd bytes (pre-built for speed: avoids re-encoding on every reply)
_ACTUATOR_ZERO_PAYLOAD = ActuatorCmd.wheeled(0, 0).to_bytes()
_ACTUATOR_ZERO_PACKET = build_packet(ACTUATOR_CMD, _ACTUATOR_ZERO_PAYLOAD)


class StubServerStats:
    """Shared stats object, updated by all connection handlers."""

    def __init__(self) -> None:
        self.connections_active: int = 0
        self.connections_total: int = 0
        self.rx_packets: int = 0
        self.rx_bytes: int = 0
        self.tx_packets: int = 0
        self.tx_bytes: int = 0
        self.parse_errors: int = 0
        self._start: float = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def summary(self) -> str:
        return (
            f"[FLEET_STUB] t={self.elapsed():.1f}s "
            f"active={self.connections_active} total={self.connections_total} "
            f"rx={self.rx_packets}pkt/{self.rx_bytes}B "
            f"tx={self.tx_packets}pkt "
            f"errors={self.parse_errors}"
        )


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    stats: StubServerStats,
    reply_every: int,
    quiet: bool,
) -> None:
    """Handle one robot connection until EOF or error."""
    stats.connections_active += 1
    stats.connections_total += 1
    sensor_count_local = 0
    buf = b""
    try:
        while True:
            chunk = await reader.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            stats.rx_bytes += len(chunk)
            buf += chunk

            # Drain all complete packets from the buffer.
            while True:
                idx = buf.find(MAGIC)
                if idx == -1:
                    buf = buf[-1:] if buf else b""
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < PACKET_HEADER_BYTES:
                    break
                pkt_type = buf[2]
                (length,) = struct.unpack_from("<H", buf, 3)
                total = PACKET_HEADER_BYTES + length + PACKET_CRC_BYTES
                if len(buf) < total:
                    break
                payload = buf[PACKET_HEADER_BYTES : PACKET_HEADER_BYTES + length]
                crc_rcv = buf[PACKET_HEADER_BYTES + length]
                crc_exp = protocol.crc8(buf[: PACKET_HEADER_BYTES + length])
                buf = buf[total:]
                if crc_rcv != crc_exp:
                    stats.parse_errors += 1
                    continue
                stats.rx_packets += 1

                # Reply path: ActuatorCmd zero on SensorPacket
                if pkt_type == SENSOR_PACKET:
                    sensor_count_local += 1
                    if reply_every > 0 and sensor_count_local % reply_every == 0:
                        writer.write(_ACTUATOR_ZERO_PACKET)
                        await writer.drain()
                        stats.tx_packets += 1
                        stats.tx_bytes += len(_ACTUATOR_ZERO_PACKET)

    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        stats.connections_active -= 1
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        if not quiet:
            pass  # individual close is too noisy; rely on periodic report


async def _report_loop(stats: StubServerStats, interval: float) -> None:
    """Print periodic status to stdout."""
    while True:
        await asyncio.sleep(interval)
        print(stats.summary(), flush=True)


async def amain(args: argparse.Namespace) -> int:
    stats = StubServerStats()

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _handle_connection(reader, writer, stats, args.reply_every, args.quiet)

    server = await asyncio.start_server(_handler, host=args.host, port=args.port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    if not args.quiet:
        print(f"[FLEET_STUB] listening on {addrs} nmax={args.nmax}", flush=True)

    report_task: Optional[asyncio.Task[None]] = None
    if not args.quiet:
        report_task = asyncio.create_task(_report_loop(stats, DEFAULT_REPORT_INTERVAL))

    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        if report_task is not None:
            report_task.cancel()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--nmax",
        type=int,
        default=DEFAULT_NMAX,
        help="max concurrent connections (informational only; OS limits apply)",
    )
    ap.add_argument(
        "--reply-every",
        type=int,
        default=DEFAULT_REPLY_EVERY,
        help="send ActuatorCmd for every Nth SensorPacket (0=never)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
