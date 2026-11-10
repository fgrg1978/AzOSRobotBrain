#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""bench_stub.py — Instrumented stub brain for bench_e2e.sh scenarios.

Extends stub_brain.py with:
  --scenario steady   measure TCP RTT over default duration (default)
  --scenario burst    send N actuator commands as fast as possible; report peak pkt/s
  --scenario boot     just wait for first packet and exit

RTT measurement:
  Sends an ActuatorCmd immediately after receiving each SENSOR packet, then
  waits for the next SENSOR.  Records delta between successive SENSOR arrivals
  as a proxy for round-trip latency.

  NOTE: under QEMU TCG, the sensor pump is often stalled (issue #39 in MEMORY),
  so RTT samples may be zero.  The collector handles this gracefully.

Burst measurement:
  After connecting, sends --burst-n ActuatorCmd packets as fast as TCP will
  accept them.  Records wall time for the full burst and prints
  "[BENCH-BURST-PEAK] <pkt/s>".

Boot timing:
  Logs "[BENCH-FIRST-PKT] <unix_ts>" on the first received packet.
  This is used by bench_e2e_collect.py to compute boot_ms.

All timing output goes to stdout so the harness can capture it in stub.log.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import protocol
from protocol import (
    build_packet,
    parse_packet,
    ActuatorCmd,
    SENSOR_PACKET,
    SENSOR_COMPACT,
    STATUS,
    OTA_ACK,
    CAMERA_FRAME,
)

# ── Named constants ───────────────────────────────────────────────────────────

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_DURATION_S = 30
DEFAULT_BURST_N = 100
DEFAULT_ACCEPT_GRACE_S = 10
RX_BUF_SIZE = 64 * 1024
RTT_TIMEOUT_S = 2.0  # max wait between SENSOR packets for RTT
EXIT_OK = 0
EXIT_NO_CONNECT = 2


# ── RFC-0019 transparent encryption wrappers ──────────────────────────────────
#
# When `AZOS_COMPANION_LINK_KEY` is set, the stub becomes the handshake RESPONDER
# (the robot dials it) and then wraps the raw asyncio reader/writer so the
# scenarios below run UNCHANGED on plaintext brain frames. On the wire each
# frame is Noise( brain_frame ) — matching the kernel pump.


class _EncryptedReader:
    """Drop-in for asyncio.StreamReader.read(): pulls RFC-0019 frames off the
    underlying stream (a rekey frame advances the key), opens each message,
    and returns the concatenated plaintext brain frames (which
    `_drain_packets` parses)."""

    def __init__(self, raw, sc) -> None:
        self._raw = raw
        self._sc = sc
        self._buf = b""

    def _extract(self) -> bytes:
        from secure_channel import FRAME_HEAD, LinkViolation, SecureChannel

        out = b""
        while len(self._buf) >= FRAME_HEAD:
            try:
                total = self._sc.frame_size(self._buf[:FRAME_HEAD])
                if len(self._buf) < total:
                    break
                frame, self._buf = self._buf[:total], self._buf[total:]
                kind, payload = self._sc.open_frame(frame)
            except LinkViolation:
                # A transport violation ends the session.
                raise ConnectionResetError("RFC-0019 session ended") from None
            if kind == SecureChannel.RECORD_REKEY:
                continue
            out += payload
        return out

    async def read(self, n: int) -> bytes:
        while True:
            out = self._extract()
            if out:
                return out
            chunk = await self._raw.read(n)
            if not chunk:
                return b""  # EOF
            self._buf += chunk


class _EncryptedWriter:
    """Drop-in for asyncio.StreamWriter: seals each plaintext brain frame as
    one RFC-0019 message before writing."""

    def __init__(self, raw, sc) -> None:
        self._raw = raw
        self._sc = sc

    def write(self, pkt: bytes) -> None:
        self._raw.write(self._sc.seal_message(pkt))

    async def drain(self) -> None:
        await self._raw.drain()

    def close(self) -> None:
        self._raw.close()

    async def wait_closed(self) -> None:
        await self._raw.wait_closed()

    def get_extra_info(self, key):  # type: ignore[no-untyped-def]
        return self._raw.get_extra_info(key)


class _MsReader:
    """Multi-stream (RFC-0021) outermost de-framing for the stub. Reads
    length-prefixed `[stream_id][len][inner]` frames, decodes the inner of
    STREAM_CONTROL frames through the Noise session when there is one, and
    returns concatenated plaintext brain frames so the
    scenarios' `_drain_packets` works unchanged. Non-control streams skipped."""

    def __init__(self, raw, sc=None) -> None:
        self._raw = raw
        self._sc = sc
        self._buf = b""

    def _decode_inner(self, inner: bytes):
        if self._sc is not None:
            from secure_channel import LinkViolation

            # One control-stream frame carries exactly one sealed message.
            try:
                return self._sc.open_message(inner)
            except LinkViolation:
                raise ConnectionResetError("RFC-0019 session ended") from None
        return inner

    def _extract(self) -> bytes:
        import multi_stream as _ms

        out = b""
        while len(self._buf) >= _ms.HEADER_LEN:
            plen = int.from_bytes(self._buf[_ms.STREAM_ID_BYTES : _ms.HEADER_LEN], "little")
            total = _ms.HEADER_LEN + plen
            if len(self._buf) < total:
                break
            sid = self._buf[0]
            inner = self._buf[_ms.HEADER_LEN : total]
            self._buf = self._buf[total:]
            if sid == _ms.STREAM_CONTROL:
                brain = self._decode_inner(inner)
                if brain:
                    out += brain
        return out

    async def read(self, n: int) -> bytes:
        while True:
            o = self._extract()
            if o:
                return o
            chunk = await self._raw.read(n)
            if not chunk:
                return b""
            self._buf += chunk


class _MsWriter:
    """Multi-stream outermost framing for the stub: seals into the Noise
    session when there is one, then wraps in `multi_stream(STREAM_CONTROL, ...)`."""

    def __init__(self, raw, sc=None) -> None:
        self._raw = raw
        self._sc = sc

    def write(self, pkt: bytes) -> None:
        import multi_stream as _ms

        b = pkt
        if self._sc is not None:
            b = self._sc.seal_message(b)
        self._raw.write(_ms.wrap(_ms.STREAM_CONTROL, b))

    async def drain(self) -> None:
        await self._raw.drain()

    def close(self) -> None:
        self._raw.close()

    async def wait_closed(self) -> None:
        await self._raw.wait_closed()

    def get_extra_info(self, key):  # type: ignore[no-untyped-def]
        return self._raw.get_extra_info(key)


def _label(pkt_type: int) -> str:
    return {
        SENSOR_PACKET: "SENSOR",
        SENSOR_COMPACT: "SENSOR_COMPACT",
        CAMERA_FRAME: "CAMERA",
        STATUS: "STATUS",
        OTA_ACK: "OTA_ACK",
    }.get(pkt_type, f"0x{pkt_type:02x}")


def _log(msg: str) -> None:
    print(msg, flush=True)


async def _send_actuator(
    writer: asyncio.StreamWriter,
    l: int = 0,
    r: int = 0,
) -> None:
    pkt = build_packet(protocol.ACTUATOR_CMD, ActuatorCmd.wheeled(l, r).to_bytes())
    writer.write(pkt)
    await writer.drain()


async def _drain_packets(buf: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """Parse as many complete packets as possible from buf; return (packets, remainder)."""
    packets = []
    while True:
        idx = buf.find(protocol.MAGIC)
        if idx == -1:
            buf = buf[-1:] if buf else buf
            break
        if idx > 0:
            buf = buf[idx:]
        parsed = parse_packet(buf)
        if parsed is None:
            break
        pkt_type, payload = parsed
        consumed = len(protocol.MAGIC) + 1 + 2 + len(payload) + 1
        buf = buf[consumed:]
        packets.append((pkt_type, payload))
    return packets, buf


# ── Scenario: steady-state RTT measurement ────────────────────────────────────


async def scenario_steady(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    duration_s: int,
) -> tuple[int, int, list[float]]:
    """
    Run steady-state scenario.

    Returns (rx_count, tx_count, rtt_samples_ms).
    Sends an ActuatorCmd for every SENSOR packet received.
    RTT sample = time between successive actuator-sends (crude proxy).
    """
    buf = b""
    rx_count = 0
    tx_count = 0
    rtt_samples: list[float] = []
    last_sensor_ts: Optional[float] = None
    first_pkt_logged = False
    start = time.monotonic()

    while time.monotonic() - start < duration_s:
        remaining = duration_s - (time.monotonic() - start)
        try:
            chunk = await asyncio.wait_for(
                reader.read(RX_BUF_SIZE),
                timeout=min(RTT_TIMEOUT_S, max(0.1, remaining)),
            )
        except asyncio.TimeoutError:
            continue
        if not chunk:
            _log("[BENCH-STUB] EOF from kernel")
            break
        buf += chunk
        pkts, buf = await _drain_packets(buf)

        for pkt_type, _payload in pkts:
            rx_count += 1
            now = time.monotonic()
            if not first_pkt_logged:
                first_pkt_logged = True
                _log(f"[BENCH-FIRST-PKT] {time.time():.6f}")
                _log(f"[STUB] first packet: type={_label(pkt_type)} len={len(_payload)}")

            if pkt_type in (SENSOR_PACKET, SENSOR_COMPACT):
                # RTT proxy: delta between successive sensor arrivals (both
                # directions combined).  Under QEMU TCG this is noisy.
                if last_sensor_ts is not None:
                    rtt_ms = (now - last_sensor_ts) * 1000.0
                    _log(f"[BENCH-RTT] {rtt_ms:.3f}")
                    rtt_samples.append(rtt_ms)
                last_sensor_ts = now

                # Reply with stop command (minimal load).
                await _send_actuator(writer, 0, 0)
                tx_count += 1

    # Final stop.
    await _send_actuator(writer, 0, 0)
    tx_count += 1
    return rx_count, tx_count, rtt_samples


# ── Scenario: throughput burst ────────────────────────────────────────────────


async def scenario_burst(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    duration_s: int,
    burst_n: int,
) -> tuple[int, int]:
    """
    Run burst scenario.

    Waits for first packet, then sends burst_n actuator commands as fast as
    possible.  Logs "[BENCH-BURST-PEAK] <pkt/s>".

    Returns (rx_count, tx_count).
    """
    buf = b""
    rx_count = 0
    tx_count = 0
    first_pkt_logged = False
    start = time.monotonic()

    # Wait for first inbound packet (gives the kernel a moment to settle).
    while time.monotonic() - start < duration_s:
        remaining = duration_s - (time.monotonic() - start)
        try:
            chunk = await asyncio.wait_for(
                reader.read(RX_BUF_SIZE),
                timeout=min(2.0, max(0.1, remaining)),
            )
        except asyncio.TimeoutError:
            # Kernel sensor pump may be stalled; try burst anyway.
            break
        if not chunk:
            break
        buf += chunk
        pkts, buf = await _drain_packets(buf)
        for pkt_type, _payload in pkts:
            rx_count += 1
            if not first_pkt_logged:
                first_pkt_logged = True
                _log(f"[BENCH-FIRST-PKT] {time.time():.6f}")
                _log(f"[STUB] first packet: type={_label(pkt_type)} len={len(_payload)}")
        if first_pkt_logged:
            break  # got one packet — start burst

    # Burst: send burst_n commands as fast as TCP will drain.
    pkt = build_packet(protocol.ACTUATOR_CMD, ActuatorCmd.wheeled(0, 0).to_bytes())
    burst_start = time.monotonic()
    for _ in range(burst_n):
        writer.write(pkt)
        tx_count += 1
    await writer.drain()
    burst_elapsed = time.monotonic() - burst_start
    if burst_elapsed > 0:
        peak = burst_n / burst_elapsed
    else:
        peak = float("inf")
    _log(f"[BENCH-BURST-PEAK] {peak:.1f}")

    # Drain any remaining inbound packets during rest of duration.
    while time.monotonic() - start < duration_s:
        remaining = duration_s - (time.monotonic() - start)
        try:
            chunk = await asyncio.wait_for(
                reader.read(RX_BUF_SIZE),
                timeout=min(1.0, max(0.1, remaining)),
            )
        except asyncio.TimeoutError:
            continue
        if not chunk:
            break
        buf += chunk
        pkts, buf = await _drain_packets(buf)
        rx_count += len(pkts)

    return rx_count, tx_count


# ── Scenario: boot timing only ────────────────────────────────────────────────


async def scenario_boot(
    reader: asyncio.StreamReader,
    duration_s: int,
) -> int:
    """
    Wait for any packet, log first-packet timestamp, then exit.
    Returns 1 if a packet was received, 0 otherwise.
    """
    buf = b""
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        remaining = duration_s - (time.monotonic() - start)
        try:
            chunk = await asyncio.wait_for(
                reader.read(RX_BUF_SIZE),
                timeout=min(2.0, max(0.1, remaining)),
            )
        except asyncio.TimeoutError:
            continue
        if not chunk:
            break
        buf += chunk
        pkts, buf = await _drain_packets(buf)
        for pkt_type, payload in pkts:
            _log(f"[BENCH-FIRST-PKT] {time.time():.6f}")
            _log(f"[STUB] first packet: type={_label(pkt_type)} len={len(payload)}")
            return 1
    return 0


# ── Server logic ──────────────────────────────────────────────────────────────


class BenchStub:
    def __init__(
        self,
        scenario: str,
        duration_s: int,
        burst_n: int,
    ) -> None:
        self.scenario = scenario
        self.duration_s = duration_s
        self.burst_n = burst_n
        self.connected = False
        self.rx_count = 0
        self.tx_count = 0

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        _log(f"[STUB] robot connected from {peer}")
        self.connected = True
        start_t = time.time()

        # Link layers (outermost → innermost): multi-stream (RFC-0021) →
        # Noise (RFC-0019). Build the active stack of wrappers so the
        # scenarios run unchanged on plaintext brain frames.
        enc_on = protocol.encrypt_link_armed()
        ms_on = protocol.multi_stream_armed()
        sc = None
        if enc_on:
            # Handshake runs on the RAW streams, before any wrapping.
            ok = await protocol.perform_handshake(reader, writer)
            if not ok:
                _log("[STUB] RFC-0019 handshake FAILED — closing")
                try:
                    writer.close()
                except Exception:
                    pass
                return
            link = protocol.link_for_writer(writer)
            sc = link.secure_channel if link is not None else None
            _log("[STUB] RFC-0019 encrypted link established")

        if ms_on:
            reader = _MsReader(reader, sc)  # type: ignore[assignment]
            writer = _MsWriter(writer, sc)  # type: ignore[assignment]
            _log("[STUB] RFC-0021 multi-stream framing active")
        elif enc_on:
            reader = _EncryptedReader(reader, sc)  # type: ignore[assignment]
            writer = _EncryptedWriter(writer, sc)  # type: ignore[assignment]

        try:
            if self.scenario == "steady":
                rx, tx, rtts = await scenario_steady(reader, writer, self.duration_s)
                self.rx_count = rx
                self.tx_count = tx

            elif self.scenario == "burst":
                rx, tx = await scenario_burst(reader, writer, self.duration_s, self.burst_n)
                self.rx_count = rx
                self.tx_count = tx

            elif self.scenario == "boot":
                self.rx_count = await scenario_boot(reader, self.duration_s)
                self.tx_count = 0

        except (ConnectionResetError, BrokenPipeError) as exc:
            _log(f"[STUB] connection error: {exc}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            duration = int(time.time() - start_t)
            _log(
                f"[STUB] disconnected — "
                f"rx[pkt={self.rx_count}] "
                f"tx[ACTUATOR={self.tx_count}] "
                f"duration={duration}s"
            )


async def amain(args: argparse.Namespace) -> int:
    # Arm the link layers from env (no-op when the vars are unset → plaintext
    # development link).
    if protocol.enable_encrypt_link():
        _log("[STUB] RFC-0019 encryption ARMED (stub is responder)")
    if protocol.enable_multi_stream():
        _log("[STUB] RFC-0021 multi-stream framing ARMED")

    stub = BenchStub(
        scenario=args.scenario,
        duration_s=args.duration_s,
        burst_n=args.burst_n,
    )

    server = await asyncio.start_server(stub.handle, host=args.host, port=args.port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    _log(f"[STUB] listening on {addrs} scenario={args.scenario}")

    async with server:
        try:
            await asyncio.wait_for(
                server.serve_forever(),
                timeout=args.duration_s + args.accept_grace_s,
            )
        except asyncio.TimeoutError:
            pass

    if not stub.connected:
        _log("[STUB] no robot ever connected — kernel didn't dial us")
        return EXIT_NO_CONNECT
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description="Instrumented bench stub brain")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--scenario",
        choices=["steady", "burst", "boot"],
        default="steady",
        help="measurement scenario",
    )
    ap.add_argument("--duration-s", type=int, default=DEFAULT_DURATION_S)
    ap.add_argument("--burst-n", type=int, default=DEFAULT_BURST_N)
    ap.add_argument("--accept-grace-s", type=int, default=DEFAULT_ACCEPT_GRACE_S)
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
