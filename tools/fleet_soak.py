#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Fleet-scale soak test: simulate N=10..1000 fake kernel robots connecting
to the brain concurrently, emitting sensor traffic, and measuring saturation.

Goal: find the N at which the brain starts dropping packets (ack-timeout) or
p99 latency spikes past 100 ms.

Usage examples:

    # Quick smoke test against fleet_brain_stub on port 9100
    python3 tools/fleet_soak.py --n 50 --duration-s 15 --port 9100

    # Full run with CSV output
    python3 tools/fleet_soak.py --n 100 --duration-s 60 --csv /tmp/soak.csv

    # Saturation sweep (doubles N every 30 s until breakdown)
    python3 tools/fleet_soak.py --sweep --port 9100

"Drop" definition: a SensorPacket was sent but no ActuatorCmd arrived within
ACK_TIMEOUT_S seconds, OR the connection was reset by the peer.  TCP itself
doesn't drop — we detect application-level non-responsiveness.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import struct
import sys
import time
from collections import deque
from typing import Deque, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import protocol
from protocol import (
    MAGIC,
    SENSOR_PACKET,
    STATUS,
    ACTUATOR_CMD,
    build_packet,
    SensorPacket,
    StatusPacket,
    ActuatorCmd,
)

# ── Constants (no magic numbers) ──────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100  # fleet_brain_stub default
DEFAULT_N = 100
DEFAULT_RAMP_S = 10  # ramp-up duration in seconds
DEFAULT_DURATION_S = 60
DEFAULT_SENSOR_HZ = 10  # packets per robot per second
REPORT_INTERVAL_S = 5.0  # how often to print aggregate stats
ACK_TIMEOUT_S = 2.0  # max seconds to wait for ActuatorCmd ack
READ_CHUNK_BYTES = 65536
CONNECT_RETRY_DELAY_S = 0.25  # wait between TCP retry on connect fail
MAX_CONNECT_RETRIES = 3

# Sweep mode constants
SWEEP_START_N = 10
SWEEP_STEP_FACTOR = 2  # double each step
SWEEP_STEP_S = 30  # seconds per step
SWEEP_P99_THRESHOLD_MS = 500.0  # stop sweep when p99 exceeds this
SWEEP_DROP_PCT_THRESH = 5.0  # stop sweep when drop% exceeds this
SWEEP_MAX_N = 2000  # hard cap to prevent runaway

# Latency tracking window: keep only the last N samples for rolling p50/p99
LATENCY_WINDOW_SIZE = 10_000

# Packet header layout
PACKET_HEADER_BYTES = 5  # MAGIC(2) + TYPE(1) + LEN(2)
PACKET_CRC_BYTES = 1


# ── Shared metrics (written by robot coroutines, read by reporter) ─────────────


class SoakMetrics:
    """Thread-safe-free shared metrics (all mutations happen in one event loop)."""

    def __init__(self) -> None:
        self.active_conns: int = 0
        self.tx_packets: int = 0
        self.rx_packets: int = 0
        self.tx_bytes: int = 0
        self.rx_bytes: int = 0
        self.connect_errors: int = 0
        self.parse_errors: int = 0
        self.disconnects: int = 0
        self.drops: int = 0  # ack-timeout drops
        # Rolling latency samples (ms); deque auto-trims to window size.
        self._latency_samples: Deque[float] = deque(maxlen=LATENCY_WINDOW_SIZE)
        self._start: float = time.monotonic()

    def record_latency(self, ms: float) -> None:
        self._latency_samples.append(ms)

    def latency_percentiles(self) -> tuple[float, float, float]:
        """Return (p50, p95, p99) in ms.  Returns (0,0,0) if no samples."""
        samples = list(self._latency_samples)
        if not samples:
            return (0.0, 0.0, 0.0)
        samples.sort()
        n = len(samples)

        def _p(pct: float) -> float:
            idx = int(n * pct / 100)
            return samples[min(idx, n - 1)]

        return (_p(50), _p(95), _p(99))

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def snapshot(self) -> dict:
        p50, p95, p99 = self.latency_percentiles()
        return {
            "t": round(self.elapsed(), 1),
            "active_conns": self.active_conns,
            "tx_packets": self.tx_packets,
            "rx_packets": self.rx_packets,
            "tx_bytes": self.tx_bytes,
            "rx_bytes": self.rx_bytes,
            "connect_errors": self.connect_errors,
            "parse_errors": self.parse_errors,
            "disconnects": self.disconnects,
            "drops": self.drops,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
        }


# ── Fake-kernel robot coroutine ───────────────────────────────────────────────


def _make_status_packet(robot_id: int) -> bytes:
    """Build a StatusPacket identifying this fake robot.

    Uses the V2 format from protocol.StatusPacket (mode, tasks_ok, canary_ok,
    uptime_s, robot_type).  We encode robot_id into the mode byte (lower 8 bits)
    so the brain can disambiguate log lines — purely informational.
    """
    pkt = StatusPacket(
        mode=robot_id & 0xFF,  # robot_id in mode byte for tracing
        tasks_ok=1,
        canary_ok=1,
        uptime_s=0,
        robot_type=protocol.ROBOT_WHEELED,
    )
    return build_packet(STATUS, pkt.to_bytes())


def _make_sensor_packet(robot_id: int, seq: int) -> bytes:
    """Build a SensorPacket with current timestamp embedded in timestamp_ms.

    We store `time.monotonic_ns() // 1_000_000` in timestamp_ms so that when
    the ack comes back we can compute RTT = now - timestamp_ms.
    """
    ts_ms = time.monotonic_ns() // 1_000_000
    pkt = SensorPacket(
        timestamp_ms=ts_ms,
        battery_mv=7400,
        accel_mg=(0, 0, 1000),
        gyro_mdps=(0, 0, 0),
        odom_dist_mm=seq,
        odom_hdg_cdeg=0,
        encoder_l=robot_id,
        encoder_r=seq & 0x7FFF_FFFF,
        range_front_mm=1500,
        range_right_mm=1500,
    )
    return build_packet(SENSOR_PACKET, pkt.to_bytes())


def _parse_one_packet(buf: bytes) -> Optional[tuple[int, bytes, bytes]]:
    """Try to extract one complete packet from buf.

    Returns (pkt_type, payload, remainder) or None if not enough data yet.
    Silently skips bytes before the MAGIC sync word.
    """
    idx = buf.find(MAGIC)
    if idx == -1:
        return None
    if idx > 0:
        buf = buf[idx:]
    if len(buf) < PACKET_HEADER_BYTES:
        return None
    pkt_type = buf[2]
    (length,) = struct.unpack_from("<H", buf, 3)
    total = PACKET_HEADER_BYTES + length + PACKET_CRC_BYTES
    if len(buf) < total:
        return None
    payload = buf[PACKET_HEADER_BYTES : PACKET_HEADER_BYTES + length]
    crc_rcv = buf[PACKET_HEADER_BYTES + length]
    crc_exp = protocol.crc8(buf[: PACKET_HEADER_BYTES + length])
    remainder = buf[total:]
    if crc_rcv != crc_exp:
        # Bad CRC — skip to after the MAGIC we found and let the next call resync.
        return None
    return (pkt_type, payload, remainder)


async def _robot_coroutine(
    robot_id: int,
    host: str,
    port: int,
    sensor_hz: float,
    end_time: float,
    metrics: SoakMetrics,
    stop_event: asyncio.Event,
) -> None:
    """Simulate one kernel robot: connect, send status, emit sensor packets,
    receive actuator acks."""

    # --- Connect with retry -----------------------------------------------
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    for attempt in range(MAX_CONNECT_RETRIES):
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=ACK_TIMEOUT_S,
            )
            break
        except (OSError, asyncio.TimeoutError) as exc:
            if attempt == MAX_CONNECT_RETRIES - 1:
                metrics.connect_errors += 1
                return
            await asyncio.sleep(CONNECT_RETRY_DELAY_S)
    else:
        metrics.connect_errors += 1
        return

    metrics.active_conns += 1

    # Pending send timestamps: FIFO (send time ms) so we can compute RTT
    # when an ack arrives.  We cap the deque so a non-replying stub doesn't
    # accumulate unbounded state.
    MAX_PENDING = int(sensor_hz * ACK_TIMEOUT_S * 4) + 1
    pending_ts: Deque[float] = deque(maxlen=MAX_PENDING)
    seq = 0
    buf = b""
    interval = 1.0 / sensor_hz
    drop_deadline: Optional[float] = None

    try:
        # Send status on connect
        status_pkt = _make_status_packet(robot_id)
        writer.write(status_pkt)
        await writer.drain()
        metrics.tx_packets += 1
        metrics.tx_bytes += len(status_pkt)

        next_send = time.monotonic()

        while not stop_event.is_set() and time.monotonic() < end_time:
            now = time.monotonic()

            # --- Ack-timeout drop detection --------------------------------
            if pending_ts and (now - (pending_ts[0] / 1000.0)) > ACK_TIMEOUT_S:
                # The oldest pending send has exceeded the ack timeout.
                metrics.drops += 1
                pending_ts.popleft()

            # --- Send next sensor packet -----------------------------------
            if now >= next_send:
                ts_ms_before = time.monotonic_ns() // 1_000_000
                pkt = _make_sensor_packet(robot_id, seq)
                try:
                    writer.write(pkt)
                    await writer.drain()
                except (BrokenPipeError, ConnectionResetError):
                    metrics.disconnects += 1
                    return
                metrics.tx_packets += 1
                metrics.tx_bytes += len(pkt)
                pending_ts.append(float(ts_ms_before))
                seq += 1
                next_send += interval

            # --- Read any available acks -----------------------------------
            sleep_until = min(next_send, time.monotonic() + interval)
            read_timeout = max(0.001, sleep_until - time.monotonic())
            try:
                chunk = await asyncio.wait_for(
                    reader.read(READ_CHUNK_BYTES),
                    timeout=read_timeout,
                )
                if not chunk:
                    metrics.disconnects += 1
                    return
                metrics.rx_bytes += len(chunk)
                buf += chunk

                # Drain all complete packets
                while True:
                    result = _parse_one_packet(buf)
                    if result is None:
                        break
                    pkt_type, payload, buf = result
                    metrics.rx_packets += 1

                    if pkt_type == ACTUATOR_CMD and pending_ts:
                        send_ts_ms = pending_ts.popleft()
                        rtt_ms = (time.monotonic_ns() // 1_000_000) - send_ts_ms
                        if rtt_ms > 0:
                            metrics.record_latency(float(rtt_ms))
                    elif pkt_type not in (ACTUATOR_CMD,):
                        pass  # other packet types: count but ignore
            except asyncio.TimeoutError:
                pass  # no data in window; loop back
            except (BrokenPipeError, ConnectionResetError):
                metrics.disconnects += 1
                return

    finally:
        metrics.active_conns -= 1
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ── Reporter + CSV ────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "t",
    "active_conns",
    "target_n",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "connect_errors",
    "parse_errors",
    "disconnects",
    "drops",
    "p50_ms",
    "p95_ms",
    "p99_ms",
]


async def _reporter(
    metrics: SoakMetrics,
    target_n: int,
    csv_path: Optional[str],
    stop_event: asyncio.Event,
) -> None:
    """Print aggregate stats every REPORT_INTERVAL_S and optionally write CSV."""
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        csv_file.flush()

    try:
        while not stop_event.is_set():
            await asyncio.sleep(REPORT_INTERVAL_S)
            snap = metrics.snapshot()
            snap["target_n"] = target_n
            print(
                f"[SOAK] t={snap['t']}s "
                f"active={snap['active_conns']}/{snap['target_n']} "
                f"tx={snap['tx_packets']}pkt rx={snap['rx_packets']}pkt "
                f"tx_bps={snap['tx_bytes']} "
                f"p50={snap['p50_ms']}ms p95={snap['p95_ms']}ms p99={snap['p99_ms']}ms "
                f"drops={snap['drops']} disconnects={snap['disconnects']} "
                f"conn_err={snap['connect_errors']}",
                flush=True,
            )
            if csv_writer is not None and csv_file is not None:
                row = {k: snap.get(k, target_n if k == "target_n" else 0) for k in CSV_COLUMNS}
                row["target_n"] = target_n
                csv_writer.writerow(row)
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()


# ── Main soak run ─────────────────────────────────────────────────────────────


async def _run_soak(
    host: str,
    port: int,
    n: int,
    ramp_s: float,
    duration_s: float,
    sensor_hz: float,
    csv_path: Optional[str],
) -> SoakMetrics:
    """Run one soak session with N robots.  Returns the final metrics."""
    metrics = SoakMetrics()
    stop_event = asyncio.Event()
    end_time = time.monotonic() + duration_s

    reporter_task = asyncio.create_task(_reporter(metrics, n, csv_path, stop_event))

    # Ramp up: spread connections over ramp_s seconds
    delay_between = ramp_s / n if n > 1 else 0.0

    robot_tasks: List[asyncio.Task] = []
    for robot_id in range(n):
        if delay_between > 0:
            await asyncio.sleep(delay_between)
        task = asyncio.create_task(
            _robot_coroutine(
                robot_id=robot_id,
                host=host,
                port=port,
                sensor_hz=sensor_hz,
                end_time=end_time,
                metrics=metrics,
                stop_event=stop_event,
            )
        )
        robot_tasks.append(task)

    # Wait for the soak duration to expire
    remaining = end_time - time.monotonic()
    if remaining > 0:
        await asyncio.sleep(remaining)

    # Signal stop and collect
    stop_event.set()
    await asyncio.gather(*robot_tasks, return_exceptions=True)
    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass

    return metrics


def _print_final_summary(
    n: int,
    duration_s: float,
    metrics: SoakMetrics,
    label: str = "SOAK",
) -> None:
    elapsed = metrics.elapsed()
    tx_rate = int(metrics.tx_packets / elapsed) if elapsed > 0 else 0
    rx_rate = int(metrics.rx_packets / elapsed) if elapsed > 0 else 0
    _, _, p99 = metrics.latency_percentiles()
    print(
        f"[{label}] N={n} duration={duration_s}s "
        f"tx={tx_rate}pkt/s rx={rx_rate}pkt/s "
        f"p99={round(p99, 1)}ms drops={metrics.drops} "
        f"disconnects={metrics.disconnects} conn_err={metrics.connect_errors}",
        flush=True,
    )


# ── Sweep mode ────────────────────────────────────────────────────────────────


async def _run_sweep(args: argparse.Namespace) -> int:
    """Sweep: double N every SWEEP_STEP_S until p99 > threshold or drop% > threshold.

    Prints the breakdown N and exits 0.
    """
    n = SWEEP_START_N
    breakdown_n: Optional[int] = None

    print(
        f"[SWEEP] starting sweep: step={SWEEP_STEP_S}s "
        f"p99_threshold={SWEEP_P99_THRESHOLD_MS}ms "
        f"drop_threshold={SWEEP_DROP_PCT_THRESH}% "
        f"max_n={SWEEP_MAX_N}",
        flush=True,
    )

    while n <= SWEEP_MAX_N:
        print(f"[SWEEP] step N={n} ...", flush=True)
        metrics = await _run_soak(
            host=args.host,
            port=args.port,
            n=n,
            ramp_s=min(float(args.ramp_s), float(SWEEP_STEP_S) / 2.0),
            duration_s=float(SWEEP_STEP_S),
            sensor_hz=float(args.sensor_hz),
            csv_path=args.csv if hasattr(args, "csv") else None,
        )

        _, _, p99 = metrics.latency_percentiles()
        total_sent = metrics.tx_packets
        drop_pct = (metrics.drops / total_sent * 100.0) if total_sent > 0 else 0.0

        _print_final_summary(n, float(SWEEP_STEP_S), metrics, label="SWEEP_STEP")

        if p99 > SWEEP_P99_THRESHOLD_MS or drop_pct > SWEEP_DROP_PCT_THRESH:
            breakdown_n = n
            print(
                f"[SWEEP] BREAKDOWN at N={n}: "
                f"p99={round(p99, 1)}ms drop%={round(drop_pct, 2)}%",
                flush=True,
            )
            break

        n = min(n * SWEEP_STEP_FACTOR, SWEEP_MAX_N + 1)

    if breakdown_n is None:
        print(
            f"[SWEEP] no breakdown found up to N={SWEEP_MAX_N} "
            f"(system is not saturated at this load)",
            flush=True,
        )
    else:
        print(f"[SWEEP] breakdown N = {breakdown_n}", flush=True)

    return 0


# ── CLI entry ─────────────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> int:
    if args.sweep:
        return await _run_sweep(args)

    metrics = await _run_soak(
        host=args.host,
        port=args.port,
        n=args.n,
        ramp_s=float(args.ramp_s),
        duration_s=float(args.duration_s),
        sensor_hz=float(args.sensor_hz),
        csv_path=getattr(args, "csv", None),
    )

    _print_final_summary(args.n, float(args.duration_s), metrics)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="target robot count")
    ap.add_argument(
        "--ramp-s", type=int, default=DEFAULT_RAMP_S, help="seconds to ramp up to N robots"
    )
    ap.add_argument(
        "--duration-s", type=int, default=DEFAULT_DURATION_S, help="total soak duration in seconds"
    )
    ap.add_argument(
        "--sensor-hz",
        type=float,
        default=DEFAULT_SENSOR_HZ,
        help="SensorPackets per robot per second",
    )
    ap.add_argument(
        "--csv", default=None, metavar="PATH", help="write per-interval CSV to this file"
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="saturation-discovery: double N every 30 s, stop at breakdown",
    )
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
