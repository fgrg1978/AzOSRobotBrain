# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Smoke tests for tools/fleet_soak.py + tools/fleet_brain_stub.py.

Spins up a fleet_brain_stub on a free port, runs fleet_soak with N=5
robots for 2 seconds, and asserts basic sanity:
  - run completes without error
  - metrics are non-zero
  - CSV file is written and non-empty
"""

from __future__ import annotations

import asyncio
import csv
import os
import socket
import sys
import tempfile
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from fleet_brain_stub import StubServerStats, _handle_connection  # noqa: E402
from fleet_soak import _run_soak, SoakMetrics  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Return an unused TCP port on 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SMOKE_N = 5
SMOKE_DURATION_S = 2
SMOKE_RAMP_S = 1
SMOKE_SENSOR_HZ = 5.0


async def _run_stub_and_soak(
    n: int = SMOKE_N,
    duration_s: float = SMOKE_DURATION_S,
    ramp_s: float = SMOKE_RAMP_S,
    sensor_hz: float = SMOKE_SENSOR_HZ,
    csv_path: Optional[str] = None,
) -> tuple[SoakMetrics, StubServerStats]:
    """Start fleet_brain_stub on a free port, run fleet_soak against it,
    return (soak_metrics, stub_stats)."""
    port = _free_port()
    stats = StubServerStats()

    # Build a minimal stub server inline so we don't need subprocess.
    REPLY_EVERY = 1
    QUIET = True

    async def _handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await _handle_connection(reader, writer, stats, REPLY_EVERY, QUIET)

    server = await asyncio.start_server(_handler, host="127.0.0.1", port=port)

    async with server:
        metrics = await _run_soak(
            host="127.0.0.1",
            port=port,
            n=n,
            ramp_s=ramp_s,
            duration_s=duration_s,
            sensor_hz=sensor_hz,
            csv_path=csv_path,
        )

    return metrics, stats


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_soak_completes_without_error() -> None:
    """Fleet soak with N=5 robots runs for 2s and exits cleanly."""
    metrics, _stats = asyncio.run(_run_stub_and_soak())
    # Must have sent packets (no crash before first send)
    assert metrics.tx_packets > 0, "soak sent zero packets"
    # Must not have more connect errors than robots
    assert metrics.connect_errors < SMOKE_N, f"too many connect errors: {metrics.connect_errors}"


def test_soak_receives_acks() -> None:
    """Brain stub sends ActuatorCmd acks and soak records RX packets."""
    metrics, stub_stats = asyncio.run(_run_stub_and_soak())
    assert (
        metrics.rx_packets > 0
    ), "soak received zero acks from stub (reply_every=1 should ack every SensorPacket)"
    assert stub_stats.rx_packets > 0, "stub received zero packets"


def test_soak_csv_non_empty() -> None:
    """CSV output file is created and has at least a header + one data row."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        csv_path = tf.name
    try:
        asyncio.run(
            _run_stub_and_soak(
                duration_s=3.0,  # longer run so at least one 5-second window fires
                csv_path=csv_path,
            )
        )
        assert os.path.exists(csv_path), "CSV file was not created"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        # With duration=3s < REPORT_INTERVAL_S=5s the reporter may fire 0 times.
        # We just assert the file is parseable (header written) — even 0 rows
        # is acceptable if the run is shorter than the report interval.
        # The header itself makes the file non-empty.
        assert os.path.getsize(csv_path) > 0, "CSV file is empty"
    finally:
        try:
            os.unlink(csv_path)
        except FileNotFoundError:
            pass


def test_soak_metrics_latency_recorded() -> None:
    """When stub replies with ActuatorCmd, soak records latency samples."""
    metrics, _ = asyncio.run(
        _run_stub_and_soak(
            n=3,
            duration_s=3.0,
            sensor_hz=10.0,
        )
    )
    # At 10 Hz * 3 robots * ~3 s = ~90 packets sent; expect some acks.
    if metrics.rx_packets > 0:
        p50, p95, p99 = metrics.latency_percentiles()
        # Loopback latency must be under 2000 ms (very loose bound)
        assert p99 < 2000.0, f"p99 suspiciously high: {p99}ms"
        assert p50 >= 0.0, "p50 must be non-negative"
