# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""GIL contention profile for the brain hot loops.

Measures three scenarios for the same work:
  S1 — sequential: do work_A then work_B in one thread.
  S2 — threaded:   do work_A and work_B in parallel threads.
  S3 — processes:  same but via multiprocessing (escapes GIL entirely).

If GIL is a real bottleneck:
  threaded ≈ sequential   (threading gives no speedup)
  processes ≈ sequential / 2  (parallelism actually realised)

If GIL is NOT the bottleneck (numpy releases it, work is I/O bound, etc.):
  threaded ≈ sequential / 2  (threads parallelise just fine)
  processes ≈ threaded       (no extra win from going multi-process)

Workloads:
  work_A — protocol build+parse loop (pure-Python struct, no I/O)
  work_B — GMM background update on synthetic frames (numpy-heavy)
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from typing import Callable, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, "/Users/azor/Library/Mobile Documents/com~apple~CloudDocs/Development/ia/AzOSRobotBrain"
)

import protocol
from protocol import SensorPacket, ActuatorCmd, build_packet, parse_packet, SENSOR_PACKET
from perception.gmm import build_gmm, GMMProfile

# ── Sizing (realistic robot load: ~30 Hz sensors, ~5 Hz vision per camera) ───
PROTO_ITERS = 2_000  # ≈ 60 s of 30 Hz sensor traffic
FRAME_W, FRAME_H = 320, 240  # low-res monitor camera (typical RTSP downscale)
GMM_FRAMES = 300  # ≈ 60 s of 5 Hz frames per camera
GMM_PROFILE = GMMProfile()  # defaults; matches what motion_detect.py uses

# Pre-build a sensor packet payload + frame data (shared, read-only).
SAMPLE_SENSOR = SensorPacket(
    timestamp_ms=1234567890,
    battery_mv=12000,
    accel_mg=(100, -200, 981),
    gyro_mdps=(1, -2, 3),
    odom_dist_mm=1234,
    odom_hdg_cdeg=4500,
    encoder_l=100,
    encoder_r=99,
    range_front_mm=2000,
    range_right_mm=1500,
)
SAMPLE_PACKET_BYTES = build_packet(SENSOR_PACKET, SAMPLE_SENSOR.to_bytes())

# Synthetic frame stream — same buffer each iter; GMM still does the work.
SAMPLE_FRAME = bytes((i * 7) & 0xFF for i in range(FRAME_W * FRAME_H))


def work_proto() -> None:
    """Build + parse ~40k packets. Pure-Python struct path."""
    for _ in range(PROTO_ITERS):
        _t, payload = parse_packet(SAMPLE_PACKET_BYTES)
        _ = SensorPacket.from_bytes(payload)
        _ = build_packet(SENSOR_PACKET, SAMPLE_SENSOR.to_bytes())


def work_gmm() -> None:
    """Run GMM updates on synthetic frames. numpy-heavy."""
    gmm = build_gmm(FRAME_W, FRAME_H, GMM_PROFILE)
    frame_list = list(SAMPLE_FRAME)
    for _ in range(GMM_FRAMES):
        _ = gmm.update(frame_list)


# ── Scenarios ─────────────────────────────────────────────────────────────────


def sequential(fns: Tuple[Callable[[], None], ...]) -> float:
    t0 = time.perf_counter()
    for f in fns:
        f()
    return time.perf_counter() - t0


def threaded(fns: Tuple[Callable[[], None], ...]) -> float:
    t0 = time.perf_counter()
    ts = [threading.Thread(target=f) for f in fns]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.perf_counter() - t0


def _proc_target(fn_name: str) -> None:
    g = globals()
    g[fn_name]()


def processed(fn_names: Tuple[str, ...]) -> float:
    t0 = time.perf_counter()
    ps = [mp.Process(target=_proc_target, args=(n,)) for n in fn_names]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return time.perf_counter() - t0


def main() -> None:
    # Warmup so first-touch import/numpy JIT cost doesn't skew the first run.
    work_proto()
    work_gmm()

    print(
        f"\n=== brain GIL profile  (proto={PROTO_ITERS} iters, gmm={GMM_FRAMES} frames @ {FRAME_W}x{FRAME_H}) ===\n"
    )

    # Single-task baselines
    t_a = sequential((work_proto,))
    t_b = sequential((work_gmm,))
    print(f"  proto alone    : {t_a:6.2f} s  ({PROTO_ITERS/t_a:,.0f} roundtrips/s)")
    print(f"  gmm alone      : {t_b:6.2f} s  ({GMM_FRAMES/t_b:,.0f} frames/s)")
    print()

    # Combined: sequential
    t_seq = sequential((work_proto, work_gmm))
    print(f"  S1 sequential  : {t_seq:6.2f} s")

    # Combined: threads
    t_thr = threaded((work_proto, work_gmm))
    print(f"  S2 threaded    : {t_thr:6.2f} s   speedup_vs_seq = {t_seq/t_thr:.2f}x")

    # Combined: processes
    t_proc = processed(("work_proto", "work_gmm"))
    print(f"  S3 processes   : {t_proc:6.2f} s   speedup_vs_seq = {t_seq/t_proc:.2f}x")

    # Two-camera scenario (LLM's specific claim): two GMM workers + proto in
    # parallel. This is the load they predicted would collapse the TCP loop.
    print()
    print("  --- two-camera scenario (proto + 2× gmm) ---")
    t_2cam_seq = sequential((work_proto, work_gmm, work_gmm))
    t_2cam_thr = threaded((work_proto, work_gmm, work_gmm))
    t_2cam_proc = processed(("work_proto", "work_gmm", "work_gmm"))
    print(f"  sequential     : {t_2cam_seq:6.2f} s")
    print(f"  threaded       : {t_2cam_thr:6.2f} s   speedup_vs_seq = {t_2cam_seq/t_2cam_thr:.2f}x")
    print(
        f"  processes      : {t_2cam_proc:6.2f} s   speedup_vs_seq = {t_2cam_seq/t_2cam_proc:.2f}x"
    )
    gil_loss_2cam = max(0.0, (t_2cam_thr - t_2cam_proc) / t_2cam_thr) * 100
    print(f"  GIL loss 2cam  : {gil_loss_2cam:5.1f}%")

    print()
    # GIL diagnosis
    thread_eff = (t_a + t_b) / t_thr  # >1.0 = some parallelism realised
    proc_eff = (t_a + t_b) / t_proc
    gil_loss_pct = max(0.0, (t_thr - t_proc) / t_thr) * 100
    print(f"  thread efficiency  : {thread_eff:.2f}  (1.0 = no parallelism, 2.0 = perfect)")
    print(f"  process efficiency : {proc_eff:.2f}")
    print(
        f"  GIL loss estimate  : {gil_loss_pct:5.1f}%  (= how much time threading loses vs processes)"
    )
    print()
    if gil_loss_pct < 10:
        print("  VERDICT: GIL is NOT the bottleneck for this workload mix.")
        print("           Multiproceso refactor would not improve throughput.")
    elif gil_loss_pct < 25:
        print("  VERDICT: Modest GIL pressure. Multiproceso plausible but marginal.")
    else:
        print("  VERDICT: GIL is a real bottleneck. Multiproceso is justified.")


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
