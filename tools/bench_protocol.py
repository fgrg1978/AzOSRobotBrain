#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Micro-benchmark for the brain wire-format hot paths.

Runs two workloads:
  1. `protocol.build_packet` — frame an ACTUATOR cmd.
  2. `protocol.parse_packet` — parse the framed bytes back.

Reports packets/sec for each. Use as a regression detector: drop ≥ 30%
between runs means something got slower. Not a pytest fixture — opt-in:

    python3 tools/bench_protocol.py
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import protocol  # noqa: E402

# Workload constants. Held outside the timed loops so each iteration of the
# loop is just the operation under test (no allocation, no scope changes).
BENCH_ITERS_FAST = 200_000  # build/parse_packet — pure Python, fast
TYPE_ACTUATOR = 0x80
INNER_PAYLOAD_BYTES = 20  # realistic ActuatorCmd payload size


def _wall_kpkt_per_s(label: str, op_count: int, elapsed_s: float) -> str:
    rate = op_count / elapsed_s if elapsed_s > 0 else float("inf")
    return f"  {label:40s} {rate / 1000:9.1f} kpkt/s  ({op_count} ops in {elapsed_s*1000:.1f} ms)"


def bench_build_packet() -> str:
    payload = b"\x00" * INNER_PAYLOAD_BYTES
    start = time.perf_counter()
    for _ in range(BENCH_ITERS_FAST):
        protocol.build_packet(TYPE_ACTUATOR, payload)
    elapsed = time.perf_counter() - start
    return _wall_kpkt_per_s("protocol.build_packet", BENCH_ITERS_FAST, elapsed)


def bench_parse_packet() -> str:
    frame = protocol.build_packet(TYPE_ACTUATOR, b"\x00" * INNER_PAYLOAD_BYTES)
    start = time.perf_counter()
    for _ in range(BENCH_ITERS_FAST):
        protocol.parse_packet(frame)
    elapsed = time.perf_counter() - start
    return _wall_kpkt_per_s("protocol.parse_packet", BENCH_ITERS_FAST, elapsed)


def main() -> int:
    print("== AzOSRobotBrain protocol micro-benchmark ==")
    print(
        "(rates are ops/s; compare against the baseline captured in the "
        "commit message to detect regressions)"
    )
    print()
    for line in (bench_build_packet(), bench_parse_packet()):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
