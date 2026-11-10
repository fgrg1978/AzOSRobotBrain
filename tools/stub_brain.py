#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Headless stub brain for end-to-end tests with a real kernel in QEMU.

Goal: drive the kernel↔brain TCP loop end-to-end without depending on
LM Studio, openai, or any vision/LLM stack. `server.py` is the full
production brain (vision → planner → policy → notifications); this
stub covers the minimum path: a wheeled robot working end to end under
QEMU, with no model server.

What it does:

    1. Listens on TCP 0.0.0.0:9000 (`DEFAULT_PORT`, matching
       `build/CONFIG.INI`'s `behavior_server_port`; QEMU forwards it
       from the host, which the guest reaches as 10.0.2.2).
    2. Accepts ONE robot connection (the kernel boots, reads
       CONFIG.INI, and connects out to 10.0.2.2:9000 = QEMU host).
    3. Decodes every packet using `protocol.parse_packet` so we get
       the full kernel-side wire-format coverage. Counts each kind.
    4. After receiving `--sensors-to-go` sensor packets (default 3),
       sends `ActuatorCmd.wheeled(50, 50)` — forward at 50 %.
    5. After `--duration-s` seconds total, sends a stop command and
       closes the connection cleanly.
    6. Exits 0 iff we received ≥ 1 sensor packet AND sent ≥ 1
       actuator command; exits 1 otherwise (lets a CI harness gate
       on "the loop actually closed").

The link (RFC-0019)
-------------------

The kernel's brain link is always a Noise session, so the stub speaks it
whenever `AZOS_COMPANION_LINK_KEY` is exported: it arms the encrypted link exactly
like `server.py` does (`protocol.enable_encrypt_link()`), runs the responder
handshake on the accepted connection and reads every packet through
`protocol.read_packet`. A configured key that fails to arm is a hard error,
never a silent downgrade to plaintext.

With the link armed, a handshake the robot cannot complete, a frame that does
not open and a message that does not parse are VERIFICATION FAILURES, not
noise: they are counted, logged, and they fail the final verdict (exit
`EXIT_AUTH_FAILURE`).

Without a key the channel stays plaintext, which only a development peer
speaks (the kernel never does); the MAGIC resync behaves as before — counted
and logged, but not fatal, because there is no cryptography to verify.

Usage:

    # Run in foreground:
    python3 tools/stub_brain.py

    # In another terminal (or use scripts/e2e_wheeled_qemu.sh):
    make qemu-full-smp     # in AzOS
    # → kernel boots, connects out, you see SENSOR packets here.

    # CI / scripted use:
    python3 tools/stub_brain.py --duration-s 20 --quiet | tee brain.log

    # Authenticated link (the kernel needs the same link key):
    export AZOS_COMPANION_LINK_KEY=$(xxd -p -c64 build/LINK.KEY)
    python3 tools/stub_brain.py --duration-s 20

    # NEGATIVE gate — proves the handshake actually rejects:
    #   corrupts the key on purpose; exits 0 ONLY if nothing verified.
    python3 tools/stub_brain.py --duration-s 10 --expect-auth-failure
"""

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

import protocol
from protocol import (
    parse_packet,
    SensorPacket,
    SensorPacketDrone,
    SensorPacketHumanoid,
    SensorCompact,
    StatusPacket,
    ActuatorCmd,
    SENSOR_PACKET,
    CAMERA_FRAME,
    STATUS,
    OTA_ACK,
    SENSOR_COMPACT,
    ROBOT_WHEELED,
)
from secure_channel import KEY_BYTES

# ── Defaults (named constants — no magic numbers) ────────────────────────────

DEFAULT_HOST = "0.0.0.0"
# Default port matches `build/CONFIG.INI`'s `behavior_server_port`.
# In QEMU user-mode networking the guest reaches the host at
# 10.0.2.2, so the kernel dials 10.0.2.2:9000 → our listener.
DEFAULT_PORT = 9000
DEFAULT_DURATION_S = 30
DEFAULT_SENSORS_TO_GO = 3
DEFAULT_FORWARD_PCT = 50
RX_BUF_SIZE = 64 * 1024
EXIT_OK = 0
EXIT_NO_LOOP = 1
EXIT_TIMEOUT = 2
#: Link verification failed (or was expected to fail and did not).
EXIT_AUTH_FAILURE = 3

# ── Frame geometry (derived from protocol.py, never hardcoded) ───────────────

#: MAGIC(2) + TYPE(1) + LEN(2)
FRAME_HEADER_BYTES = len(protocol.MAGIC) + 1 + 2
FRAME_CRC_BYTES = 1
FRAME_OVERHEAD_BYTES = FRAME_HEADER_BYTES + FRAME_CRC_BYTES
#: Largest brain-protocol frame.
MAX_FRAME_BYTES = FRAME_OVERHEAD_BYTES + protocol.MAX_PAYLOAD_BYTES

#: Upper bound on bytes we may discard while hunting for the FIRST frame on
#: the plaintext development link. QEMU slirp drops the first byte of the
#: first application segment; losing byte 0 destroys the whole first frame,
#: so the next frame boundary can be up to one maximum-size frame away.
MAX_RESYNC_SKEW_BYTES = MAX_FRAME_BYTES

#: Cap on per-event resync/verification log lines so a mis-keyed peer
#: cannot flood the CI log; the final summary always carries the totals.
MAX_NOISE_LOG_LINES = 5


def log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, flush=True)


def _corrupt_key_hex(key_hex: str) -> str:
    """Flip the low bit of the key's last byte — used by
    `--expect-auth-failure` so the negative gate needs no second key."""
    raw = bytearray(bytes.fromhex(key_hex))
    if len(raw) != KEY_BYTES:
        raise ValueError(f"link key must be {KEY_BYTES} bytes, got {len(raw)}")
    raw[-1] ^= 0x01
    return bytes(raw).hex()


class StubBrain:
    """A single-connection headless brain.

    Per-connection state lives on the instance so a future fork to
    handle multiple kernels would just spawn one per accept().
    """

    def __init__(
        self,
        duration_s: int,
        sensors_to_go: int,
        forward_pct: int,
        quiet: bool,
        expect_auth_failure: bool = False,
    ):
        self.duration_s = duration_s
        self.sensors_to_go = sensors_to_go
        self.forward_pct = forward_pct
        self.quiet = quiet
        self.expect_auth_failure = expect_auth_failure
        self.pkt_counts: dict[int, int] = {}
        self.cmds_sent = 0
        self.first_seen_at: Optional[float] = None
        self.start_t: Optional[float] = None
        self.connected = False
        self.sensors_seen = 0

        # ── Link accounting ──────────────────────────────────────────
        #: True when the encrypted link is armed for this run.
        self.encrypted = protocol.encrypt_link_armed()
        #: Every byte read off the socket (verified or not).
        self.bytes_rx = 0
        #: Frames that opened under the session.
        self.verified_frames = 0
        #: Handshakes the robot did not complete, and frames that did not
        #: open (forged, mis-keyed, replayed, reordered). Always fatal.
        self.auth_failures = 0
        #: A frame opened but its message does not parse → the two repos
        #: disagree on the wire format. Always fatal.
        self.bad_inner_frames = 0
        #: Plaintext development link only: bytes discarded before the first
        #: frame and within `MAX_RESYNC_SKEW_BYTES` (the slirp drop).
        #: Tolerated, but reported.
        self.skew_bytes = 0
        #: Plaintext development link only: bytes discarded that never
        #: parsed. Reported, not fatal: there is no cryptography to verify.
        self.noise_bytes = 0
        #: Bytes of a frame still in flight when the peer closed. Ordinary
        #: TCP teardown, NOT a verification failure — never fatal.
        self.truncated_bytes = 0

        self._synced = False
        self._sync_abandoned = False
        self._noise_logs = 0

    # ── Logging helpers ──────────────────────────────────────────────

    def _note(self, msg: str) -> None:
        """Bounded per-event logging (totals always land in the summary)."""
        if self._noise_logs >= MAX_NOISE_LOG_LINES:
            return
        self._noise_logs += 1
        log(self.quiet, msg)
        if self._noise_logs == MAX_NOISE_LOG_LINES:
            log(self.quiet, "[STUB] (further resync/verification events suppressed)")

    def _label(self, pkt_type: int) -> str:
        return {
            SENSOR_PACKET: "SENSOR",
            CAMERA_FRAME: "CAMERA",
            STATUS: "STATUS",
            OTA_ACK: "OTA_ACK",
            SENSOR_COMPACT: "SENSOR_COMPACT",
        }.get(pkt_type, f"0x{pkt_type:02x}")

    def _summary(self) -> str:
        # NOTE: the `rx[...] tx[ACTUATOR=N] duration=Ns` prefix is parsed by
        # tools/bench_e2e_collect.py (_RE_STUB_SUMMARY) — append, never reshape.
        kinds = ", ".join(f"{self._label(k)}={v}" for k, v in sorted(self.pkt_counts.items()))
        out = (
            f"rx[{kinds or 'none'}] tx[ACTUATOR={self.cmds_sent}] "
            f"duration={int(time.time() - (self.start_t or time.time()))}s"
        )
        if self.encrypted:
            out += (
                f" auth[verified={self.verified_frames}, failed={self.auth_failures}, "
                f"bad_inner={self.bad_inner_frames}]"
            )
        elif self.skew_bytes or self.noise_bytes:
            out += f" resync[skew={self.skew_bytes}B, dropped={self.noise_bytes}B]"
        return out

    # ── Byte accounting ──────────────────────────────────────────────

    def _discard(self, count: int, why: str) -> None:
        """Account for `count` bytes of the plaintext link we are about to
        throw away.

        Before the first good frame, and only within the slirp budget,
        the loss is tolerated (`skew_bytes`). Everything else lands in
        `noise_bytes`, reported but not fatal.
        """
        if count <= 0:
            return
        if not self._synced and self.skew_bytes + count <= MAX_RESYNC_SKEW_BYTES:
            self.skew_bytes += count
            self._note(f"[STUB] resync: tolerating {count}B before first frame ({why})")
            return
        self.noise_bytes += count
        self._note(f"[STUB] DISCARD {count}B that never verified/parsed ({why})")

    def _note_eof_remainder(self, buf: bytes) -> None:
        """Classify what is left of the plaintext stream when the peer closes.

        A stream cut mid-frame is ordinary TCP teardown, not a forgery.
        """
        if not buf:
            return
        self.truncated_bytes += len(buf)
        self._note(
            f"[STUB] EOF with {len(buf)}B of an incomplete frame "
            f"(connection cut mid-frame — not a verification failure)"
        )

    # ── Plaintext framing (development link, no key configured) ──────

    def _drain_plain(self, buf: bytes) -> tuple[bytes, list[tuple[int, bytes]]]:
        """Drain whole packets, resyncing on MAGIC.

        Defensive resync: QEMU slirp NAT has a reproducible bug that
        drops the FIRST byte of the FIRST application-data segment after
        the TCP handshake. We compensate by scanning forward to the next
        MAGIC instead of requiring it at offset 0. On real hardware
        (Ethernet, no slirp) the scan is a no-op. Discards are counted
        and logged; with no key there is nothing to verify, so they are
        reported but not fatal.
        """
        out: list[tuple[int, bytes]] = []
        while True:
            idx = buf.find(protocol.MAGIC)
            if idx == -1:
                # Keep the last byte in case MAGIC straddles two chunks.
                keep = buf[-1:]
                self._discard(len(buf) - len(keep), "no MAGIC in buffer")
                buf = keep
                break
            if idx > 0:
                self._discard(idx, "bytes before MAGIC")
                buf = buf[idx:]
            parsed = parse_packet(buf)
            if parsed is None:
                if len(buf) >= FRAME_HEADER_BYTES:
                    (length,) = struct.unpack_from("<H", buf, len(protocol.MAGIC) + 1)
                    if (
                        length <= protocol.MAX_PAYLOAD_BYTES
                        and len(buf) >= FRAME_OVERHEAD_BYTES + length
                    ):
                        # Complete frame that failed CRC → skip this MAGIC.
                        self._discard(len(protocol.MAGIC), "CRC mismatch at MAGIC")
                        buf = buf[len(protocol.MAGIC) :]
                        continue
                break  # incomplete — wait for more bytes
            pkt_type, payload = parsed
            buf = buf[FRAME_OVERHEAD_BYTES + len(payload) :]
            self._synced = True
            out.append((pkt_type, payload))
        return buf, out

    # ── Connection handling ──────────────────────────────────────────

    async def _send(self, writer: asyncio.StreamWriter, pkt_type: int, payload: bytes) -> None:
        # protocol.send_packet seals into the connection's Noise session when
        # a key is configured, so the stub emits byte-identical frames to the
        # production server.
        await protocol.send_packet(writer, pkt_type, payload)

    async def _on_packet(self, writer: asyncio.StreamWriter, pkt_type: int, payload: bytes) -> None:
        self.pkt_counts[pkt_type] = self.pkt_counts.get(pkt_type, 0) + 1
        if self.first_seen_at is None:
            self.first_seen_at = time.time()
            log(
                self.quiet,
                f"[STUB] first packet: type={self._label(pkt_type)} len={len(payload)}",
            )
        if pkt_type == SENSOR_PACKET:
            self.sensors_seen += 1
            if self.sensors_seen == self.sensors_to_go:
                log(
                    self.quiet,
                    f"[STUB] {self.sensors_seen} sensors received → "
                    f"sending wheeled forward {self.forward_pct}%",
                )
                await self._send(
                    writer,
                    protocol.ACTUATOR_CMD,
                    ActuatorCmd.wheeled(self.forward_pct, self.forward_pct).to_bytes(),
                )
                self.cmds_sent += 1

    def _remaining_s(self) -> float:
        return self.duration_s - (time.time() - (self.start_t or time.time()))

    async def _stop_command(self, writer: asyncio.StreamWriter) -> None:
        log(self.quiet, "[STUB] duration reached, sending stop")
        await self._send(writer, protocol.ACTUATOR_CMD, ActuatorCmd.wheeled(0, 0).to_bytes())
        self.cmds_sent += 1

    async def _serve_encrypted(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handshake, then read every packet through the connection's session."""
        # Count what the handshake and the frames read, so the negative gate
        # can tell "nothing was offered" from "everything was refused".
        inner_read = reader.readexactly

        async def counting(n: int) -> bytes:
            data = await inner_read(n)
            self.bytes_rx += len(data)
            return data

        reader.readexactly = counting  # type: ignore[method-assign]
        if not await protocol.perform_handshake(reader, writer):
            self.auth_failures += 1
            self._note("[STUB] AUTH FAILURE: the robot did not complete the handshake")
            return
        while True:
            left = self._remaining_s()
            if left <= 0:
                await self._stop_command(writer)
                return
            try:
                parsed = await asyncio.wait_for(protocol.read_packet(reader), timeout=max(0.5, left))
            except asyncio.TimeoutError:
                continue
            except asyncio.IncompleteReadError:
                log(self.quiet, "[STUB] EOF from robot")
                return
            if parsed is None:
                # A frame that did not open ends the session
                # (`read_packet` logs the violation); an opened message that
                # does not parse is a wire-format disagreement.
                link = protocol.link_for_reader(reader)
                if link is not None and not link.secure_channel.is_established:
                    self.auth_failures += 1
                else:
                    self.bad_inner_frames += 1
                    self._note(
                        "[STUB] frame opened but its message does not parse — "
                        "wire-format drift between AzOS and AzOSRobotBrain"
                    )
                return
            self.verified_frames += 1
            await self._on_packet(writer, *parsed)

    async def _serve_plain(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        buf = b""
        while True:
            left = self._remaining_s()
            if left <= 0:
                await self._stop_command(writer)
                return
            # Read whatever's available, bounded by remaining time.
            try:
                chunk = await asyncio.wait_for(reader.read(RX_BUF_SIZE), timeout=max(0.5, left))
            except asyncio.TimeoutError:
                continue
            if not chunk:
                log(self.quiet, "[STUB] EOF from robot")
                self._note_eof_remainder(buf)
                return
            buf += chunk
            self.bytes_rx += len(chunk)
            buf, packets = self._drain_plain(buf)
            for pkt_type, payload in packets:
                await self._on_packet(writer, pkt_type, payload)

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        self.encrypted = protocol.encrypt_link_armed()
        log(self.quiet, f"[STUB] robot connected from {peer}")
        log(
            self.quiet,
            "[STUB] link: Noise session per connection (handshake required)"
            if self.encrypted
            else "[STUB] link: PLAINTEXT (no AZOS_COMPANION_LINK_KEY; development only)",
        )
        self.connected = True
        self.start_t = time.time()
        self.sensors_seen = 0

        try:
            if self.encrypted:
                await self._serve_encrypted(reader, writer)
            else:
                await self._serve_plain(reader, writer)
        except (ConnectionResetError, BrokenPipeError) as e:
            log(self.quiet, f"[STUB] connection error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            log(self.quiet, f"[STUB] disconnected — {self._summary()}")


def compute_exit_code(brain: StubBrain, accept_tcp_only: bool = False) -> int:
    """The verdict, as a pure function of the run's accounting.

    Kept separate from `amain` so the gate itself is unit-testable — a
    gate nobody has watched reject something is trusted on faith.
    """
    quiet = brain.quiet
    if not brain.connected:
        log(quiet, "[STUB] no robot ever connected — kernel didn't dial us")
        return EXIT_TIMEOUT

    if brain.expect_auth_failure:
        # Negative gate: the key is wrong ON PURPOSE. Success means the
        # handshake rejected everything that arrived.
        if brain.bytes_rx == 0:
            log(
                quiet,
                "[STUB] FAIL (negative gate inconclusive): not one byte arrived, "
                "so nothing was ever offered for verification",
            )
            return EXIT_NO_LOOP
        if brain.verified_frames or brain.pkt_counts:
            log(
                quiet,
                f"[STUB] FAIL: {brain.verified_frames} frame(s) VERIFIED under a "
                f"deliberately WRONG key — the link is NOT being enforced. "
                f"{brain._summary()}",
            )
            return EXIT_AUTH_FAILURE
        log(
            quiet,
            f"[STUB] OK (expected auth failure): rejected {brain.auth_failures} frame(s) / "
            f"{brain.bytes_rx}B, verified none. {brain._summary()}",
        )
        return EXIT_OK

    if brain.encrypted:
        if brain.auth_failures or brain.bad_inner_frames:
            log(
                quiet,
                f"[STUB] FAIL: link verification failed — "
                f"{brain.auth_failures} refused (handshake or frame), "
                f"{brain.bad_inner_frames} unparseable message. {brain._summary()}",
            )
            return EXIT_AUTH_FAILURE
        if brain.bytes_rx and brain.verified_frames == 0:
            log(
                quiet,
                f"[STUB] FAIL: received {brain.bytes_rx}B but not one frame opened. "
                f"{brain._summary()}",
            )
            return EXIT_AUTH_FAILURE

    # `--accept-tcp-only` reports SUCCESS on a TCP handshake alone,
    # without requiring inbound sensor packets. Useful under QEMU
    # TCG on macOS where the host can't emulate the kernel timer
    # in real time — sensor-task `task_block(Timer(+100ms))` becomes
    # a multi-second wait, so the brain never sees a packet in the
    # test window even though the kernel TCP path is healthy.
    # It never masks a crypto failure: those are checked above.
    if accept_tcp_only:
        log(quiet, f"[STUB] OK (tcp-only): {brain._summary()}")
        return EXIT_OK
    if not (sum(brain.pkt_counts.values()) > 0 and brain.cmds_sent > 0):
        log(quiet, f"[STUB] loop incomplete: {brain._summary()}")
        return EXIT_NO_LOOP
    log(quiet, f"[STUB] OK: {brain._summary()}")
    return EXIT_OK


def arm_link_layer(quiet: bool, expect_auth_failure: bool) -> Optional[int]:
    """Arm the encrypted link from the environment.

    Returns None on success, or an exit code on a fatal misconfiguration.
    Never downgrades silently: if a key is configured it MUST arm.
    """
    key_hex = os.environ.get("AZOS_COMPANION_LINK_KEY", "").strip()

    if expect_auth_failure:
        if not key_hex:
            print(
                "[STUB] FATAL: --expect-auth-failure needs AZOS_COMPANION_LINK_KEY set "
                "(it corrupts that key on purpose); with no key there is no "
                "handshake to reject anything",
                flush=True,
            )
            return EXIT_AUTH_FAILURE
        try:
            key_hex = _corrupt_key_hex(key_hex)
        except ValueError as e:
            print(f"[STUB] FATAL: --expect-auth-failure: {e}", flush=True)
            return EXIT_AUTH_FAILURE
        os.environ["AZOS_COMPANION_LINK_KEY"] = key_hex
        print(
            "[STUB] --expect-auth-failure: link key deliberately CORRUPTED "
            "(last byte bit-flipped) — NOTHING must verify; exit 0 only if "
            "every frame is rejected",
            flush=True,
        )

    if not key_hex:
        log(quiet, "[STUB] link: PLAINTEXT (AZOS_COMPANION_LINK_KEY unset)")
        return None

    if not protocol.enable_encrypt_link():
        # Loud even under --quiet: a configured key that fails to load is
        # a security regression, not a log-level decision.
        print(
            "[STUB] FATAL: AZOS_COMPANION_LINK_KEY is set but the encrypted link "
            "refused to arm (bad hex or wrong length) — refusing to run "
            "unauthenticated",
            flush=True,
        )
        return EXIT_AUTH_FAILURE
    log(quiet, "[STUB] link: Noise session ARMED from AZOS_COMPANION_LINK_KEY")
    return None


async def amain(args: argparse.Namespace) -> int:
    armed = arm_link_layer(args.quiet, args.expect_auth_failure)
    if armed is not None:
        return armed

    brain = StubBrain(
        duration_s=args.duration_s,
        sensors_to_go=args.sensors_to_go,
        forward_pct=args.forward_pct,
        quiet=args.quiet,
        expect_auth_failure=args.expect_auth_failure,
    )

    server = await asyncio.start_server(brain.handle, host=args.host, port=args.port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    log(args.quiet, f"[STUB] listening on {addrs}")

    # Accept one connection's lifecycle, then return.
    async with server:
        try:
            await asyncio.wait_for(
                server.serve_forever(),
                timeout=args.duration_s + args.accept_grace_s,
            )
        except asyncio.TimeoutError:
            pass

    return compute_exit_code(brain, accept_tcp_only=args.accept_tcp_only)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--duration-s", type=int, default=DEFAULT_DURATION_S)
    ap.add_argument("--sensors-to-go", type=int, default=DEFAULT_SENSORS_TO_GO)
    ap.add_argument("--forward-pct", type=int, default=DEFAULT_FORWARD_PCT)
    ap.add_argument(
        "--accept-grace-s",
        type=int,
        default=5,
        help="extra seconds to wait for the kernel to dial us in",
    )
    ap.add_argument(
        "--accept-tcp-only",
        action="store_true",
        help="report success on TCP handshake alone (no rx needed) — "
        "use under QEMU TCG where the host can't keep up with "
        "the kernel timer in real time. Crypto failures still fail.",
    )
    ap.add_argument(
        "--expect-auth-failure",
        action="store_true",
        help="NEGATIVE gate: corrupt AZOS_COMPANION_LINK_KEY on purpose and "
        "exit 0 only if nothing verifies (proves the handshake rejects)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
