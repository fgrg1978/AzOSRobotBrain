# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""DEV04 — fleet-wide OTA orchestration on top of FleetManager.

`FleetManager` (E07) already tracks online robots over their persistent
brain↔robot TCP connection. This module layers OTA push on top: a
*separate* socket per robot (the kernel's `ota recv <port>` listener),
parallel fan-out, per-robot status tracking, REST-friendly job state,
and retry on transient failure.

Why a separate socket: the kernel-side `PKT_OTA_BEGIN / CHUNK / END`
packet types are declared (`protocol.OTA_BEGIN` etc.) but **not yet
wired** into the persistent stream handler. Until they are, the
production OTA path is the shell-spawned `ota recv` listener that
`tools/fleet_ota_deploy.py` already targets. We reuse the same
24-byte header layout so the kernel side requires zero changes.

Usage from `api.py`:

    fota = FleetOtaManager(fleet)
    job_id = await fota.start_job(
        image=open("kernel.bin", "rb").read(),
        sig=open("kernel.bin.sig", "rb").read(),
        platform="qemu",
        fw_version=2,
        robot_ids=["bot_1", "bot_2"],   # or None = all online
    )
    status = fota.get_job_status(job_id)
    # → {"state": "RUNNING", "per_robot": {"bot_1": "SENDING", ...}}

Tests can inject `open_connection_fn` so the socket call is mocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import struct
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from fleet import FleetManager, RobotRecord
import metrics as _metrics_mod

logger = logging.getLogger("brain.fleet_ota")


# ── Named constants (no magic numbers) ────────────────────────────────

OTA_MAGIC = b"ROTA"
OTA_HEADER_VERSION = 1
OTA_HEADER_SIZE = 24
OTA_MAX_IMAGE_SIZE = 2 * 1024 * 1024
OTA_DEFAULT_PORT = 8080
OTA_CONNECT_TIMEOUT_S = 10.0
OTA_TRANSFER_TIMEOUT_S = 120.0
OTA_RETRY_MAX_ATTEMPTS = 3
OTA_RETRY_BACKOFF_S = 2.0
OTA_FAN_OUT_CONCURRENCY = 8  # parallel robot uploads

PLATFORM_IDS = {"qemu": 0, "vf2": 1, "k1": 2, "esp32c3": 3}

JOB_ID_BYTES = 8  # 16-hex-char job ids — enough for human readability

# Ed25519 signature verification (brain-side, before any push to robots).
# The production public key lives at this path relative to this module.
# It is 32 raw bytes (the Ed25519 compressed public key point).
#
# This file is NOT in the repo and is not generated here: the keypair belongs
# to the kernel repo (AzOS), whose `crates/ota/build.rs` embeds the same
# prod_pub.bin into SECURE_BOOT_PUBKEY. Copy it across rather than minting a
# second one, or the brain and the kernel will verify against different keys.
OTA_PUBKEY_PATH = os.path.join(os.path.dirname(__file__), "tools", "keys", "prod_pub.bin")
OTA_ED25519_PUBKEY_BYTES = 32  # Ed25519 public key is always exactly 32 bytes

#: Where the OTA signing toolchain actually lives — the kernel repo, not this
#: one. The error path used to point at "tools/sign_ota.py", which does not
#: exist here, sending operators looking for a file they cannot find.
OTA_SIGNING_TOOLS_HINT = (
    "the kernel repo (AzOS): generate the keypair with tools/gen_prod_key.py, "
    "copy tools/keys/prod_pub.bin here, and sign images with tools/sign_ota.py"
)


# ── State enums ───────────────────────────────────────────────────────


class RobotOtaState(str, Enum):
    PENDING = "PENDING"
    CONNECTING = "CONNECTING"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED = "FAILED"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"  # all robots completed (mix of SENT/FAILED)
    CANCELLED = "CANCELLED"


# ── Wire format helpers ───────────────────────────────────────────────


def build_ota_header(image_size: int, image_crc32: int, fw_version: int, platform_id: int) -> bytes:
    """Build the 24-byte OTA header. Matches `tools/ota_send.py` exactly."""
    return struct.pack(
        "<4sIIIIBBH",
        OTA_MAGIC,
        OTA_HEADER_VERSION,
        image_size,
        image_crc32,
        fw_version,
        platform_id,
        0,  # flags
        0,  # reserved
    )


# ── Per-robot record + per-job record ─────────────────────────────────


@dataclass
class RobotOtaRecord:
    robot_id: str
    state: RobotOtaState = RobotOtaState.PENDING
    attempts: int = 0
    error: Optional[str] = None
    bytes_sent: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class OtaJob:
    job_id: str
    image_size: int
    fw_version: int
    platform: str
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    per_robot: dict[str, RobotOtaRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "image_size": self.image_size,
            "fw_version": self.fw_version,
            "platform": self.platform,
            "state": self.state.value,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "per_robot": {
                rid: {
                    "state": r.state.value,
                    "attempts": r.attempts,
                    "error": r.error,
                    "bytes_sent": r.bytes_sent,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                }
                for rid, r in self.per_robot.items()
            },
        }


# ── Ed25519 signature verification ───────────────────────────────────


def verify_ota_signature(image: bytes, sig: bytes) -> None:
    """Verify the Ed25519 signature of `image` against the production public key.

    Raises ValueError with a descriptive message on any failure so the caller
    can return a clean 400 Bad Request to the operator without crashing.

    We verify on the brain side — before any bytes reach a robot — because:
      1. A compromised operator workstation or API client must not be able to
         push an unsigned or wrongly-signed image to the fleet.
      2. The kernel already verifies on receive (secure_boot.rs), but that
         layer should be a backstop, not the only defence.
    """
    # Load public key from disk — fail loudly if missing so operators notice
    # during deployment rather than silently accepting unsigned images.
    if not os.path.isfile(OTA_PUBKEY_PATH):
        msg = (
            f"OTA public key not found at '{OTA_PUBKEY_PATH}'. "
            f"Provision it from {OTA_SIGNING_TOOLS_HINT}."
        )
        # This log line used to read "signature check skipped", which is the
        # opposite of what happens: the raise below is fail-closed and nothing
        # reaches the fleet. An operator reading the logs concluded an unsigned
        # image had gone out. Say what the code does.
        logger.error("[OTA] push REJECTED (no public key, cannot verify) — %s", msg)
        raise ValueError(msg)
    with open(OTA_PUBKEY_PATH, "rb") as f:
        pubkey_raw = f.read()
    if len(pubkey_raw) != OTA_ED25519_PUBKEY_BYTES:
        msg = f"OTA public key must be {OTA_ED25519_PUBKEY_BYTES} bytes, " f"got {len(pubkey_raw)}"
        logger.error("[OTA] push REJECTED (bad public key file) — %s", msg)
        raise ValueError(msg)
    if not sig:
        msg = "OTA signature is empty — cannot push unsigned image"
        logger.error("[OTA] push REJECTED (unsigned image) — %s", msg)
        raise ValueError(msg)

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature
    except ImportError as e:
        msg = (
            f"'cryptography' package not installed — cannot verify OTA "
            f"signature. Add it to requirements.txt: {e}"
        )
        logger.error("[OTA] push REJECTED (crypto dependency missing) — %s", msg)
        raise ValueError(msg) from e

    pub = Ed25519PublicKey.from_public_bytes(pubkey_raw)
    try:
        pub.verify(sig, image)
    except InvalidSignature as e:
        msg = (
            "OTA image Ed25519 signature verification FAILED — "
            "refusing to push unsigned or tampered image to fleet"
        )
        # ERROR level: a bad signature is evidence of an active attack or
        # tampering, not a configuration mistake.
        logger.error("[OTA] SIGNATURE INVALID — %s", msg)
        raise ValueError(msg) from e


# ── Connection-factory abstraction (so tests can mock) ────────────────

OpenConnectionFn = Callable[
    [str, int],
    Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
]


async def _default_open_connection(host: str, port: int):
    return await asyncio.wait_for(
        asyncio.open_connection(host, port),
        timeout=OTA_CONNECT_TIMEOUT_S,
    )


# ── FleetOtaManager ───────────────────────────────────────────────────


class FleetOtaManager:
    """Owns OTA jobs across the registered fleet.

    Stateful: holds a job table in memory. Restart loses in-flight jobs
    — that's acceptable for v1 (the kernel either booted the new image
    or rolled back via the existing OTA recovery slot).
    """

    def __init__(
        self,
        fleet: FleetManager,
        open_connection_fn: Optional[OpenConnectionFn] = None,
    ):
        self._fleet = fleet
        self._open_conn = open_connection_fn or _default_open_connection
        self._jobs: dict[str, OtaJob] = {}
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────

    async def start_job(
        self,
        image: bytes,
        sig: bytes,
        platform: str,
        fw_version: int,
        robot_ids: Optional[list[str]] = None,
    ) -> str:
        """Kick off an OTA deployment. Returns the new job_id immediately;
        the actual upload runs in the background.

        `robot_ids=None` ⇒ every currently-online robot.
        """
        if len(image) > OTA_MAX_IMAGE_SIZE:
            raise ValueError(
                f"image too large: {len(image)} > {OTA_MAX_IMAGE_SIZE}",
            )
        if platform not in PLATFORM_IDS:
            raise ValueError(
                f"unknown platform '{platform}'; " f"expected one of {sorted(PLATFORM_IDS)}",
            )
        # Verify Ed25519 signature on the brain side before any robot receives
        # bytes.  This is the first enforcement point; the kernel's secure_boot
        # is the second.  Raises ValueError with a human-readable message on
        # failure so api.py can return a clean 400.
        verify_ota_signature(image, sig)

        # Resolve target robots.
        if robot_ids is None:
            targets = [r.robot_id for r in self._fleet.online_robots()]
        else:
            targets = robot_ids
        if not targets:
            raise ValueError("no robots to target (fleet empty or offline)")

        job_id = secrets.token_hex(JOB_ID_BYTES)
        job = OtaJob(
            job_id=job_id,
            image_size=len(image),
            fw_version=fw_version,
            platform=platform,
            per_robot={rid: RobotOtaRecord(robot_id=rid) for rid in targets},
        )
        async with self._lock:
            self._jobs[job_id] = job

        # Fire-and-forget background task; status is queryable via job_id.
        asyncio.create_task(self._run_job(job, image, sig))
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    async def cancel_job(self, job_id: str) -> bool:
        """Mark a job cancelled. In-flight uploads will still run to
        completion (we don't yank sockets), but no new robots will start."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in (JobState.DONE, JobState.CANCELLED):
                return False
            job.state = JobState.CANCELLED
            return True

    # ── Internals ──────────────────────────────────────────────────────

    async def _run_job(self, job: OtaJob, image: bytes, sig: bytes) -> None:
        """Fan-out OTA push to all robots in `job.per_robot`."""
        job.state = JobState.RUNNING
        crc = zlib.crc32(image) & 0xFFFFFFFF
        platform_id = PLATFORM_IDS[job.platform]
        header = build_ota_header(len(image), crc, job.fw_version, platform_id)

        sem = asyncio.Semaphore(OTA_FAN_OUT_CONCURRENCY)

        async def _one(robot_id: str) -> None:
            async with sem:
                if job.state == JobState.CANCELLED:
                    return
                await self._push_to_robot(job, robot_id, header, image, sig)

        await asyncio.gather(*(_one(rid) for rid in job.per_robot))

        job.state = JobState.DONE
        job.finished_at = time.time()

    async def _push_to_robot(
        self,
        job: OtaJob,
        robot_id: str,
        header: bytes,
        image: bytes,
        sig: bytes,
    ) -> None:
        """Single-robot OTA push with retry. Updates job.per_robot[robot_id]
        in place."""
        rec = job.per_robot[robot_id]
        robot = self._fleet.get(robot_id)
        if robot is None or not robot.online:
            rec.state = RobotOtaState.FAILED
            rec.error = "robot not online"
            return

        host, port = self._resolve_endpoint(robot)
        if host is None:
            rec.state = RobotOtaState.FAILED
            rec.error = "could not resolve peer address"
            return

        rec.started_at = time.time()

        for attempt in range(1, OTA_RETRY_MAX_ATTEMPTS + 1):
            if job.state == JobState.CANCELLED:
                rec.state = RobotOtaState.FAILED
                rec.error = "job cancelled"
                return
            rec.attempts = attempt
            try:
                await self._push_once(rec, host, port, header, image, sig)
                rec.state = RobotOtaState.SENT
                rec.finished_at = time.time()
                _metrics_mod.M.ota_pushes_total.labels(status=DISPATCH_STATUS_OK).inc()
                logger.info(
                    "[FleetOTA] %s: sent %d bytes (attempt %d)",
                    robot_id,
                    rec.bytes_sent,
                    attempt,
                )
                return
            except Exception as e:  # noqa: BLE001
                rec.error = f"attempt {attempt}: {e}"
                logger.warning(
                    "[FleetOTA] %s: %s",
                    robot_id,
                    rec.error,
                )
                if attempt < OTA_RETRY_MAX_ATTEMPTS:
                    await asyncio.sleep(OTA_RETRY_BACKOFF_S * attempt)

        rec.state = RobotOtaState.FAILED
        rec.finished_at = time.time()
        _metrics_mod.M.ota_pushes_total.labels(status=DISPATCH_STATUS_ERROR).inc()

    async def _push_once(
        self,
        rec: RobotOtaRecord,
        host: str,
        port: int,
        header: bytes,
        image: bytes,
        sig: bytes,
    ) -> None:
        """One OTA upload attempt. Wire format: header (24B) + image + sig.

        Matches what `tools/ota_send.py` does over a fresh socket. We use
        the brain's injected `open_connection_fn` so tests can mock it.
        """
        rec.state = RobotOtaState.CONNECTING
        rec.bytes_sent = 0

        reader, writer = await self._open_conn(host, port)
        try:
            rec.state = RobotOtaState.SENDING

            async def _write_all(buf: bytes) -> None:
                writer.write(buf)
                await writer.drain()
                rec.bytes_sent += len(buf)

            await asyncio.wait_for(
                _write_all(header + image + sig),
                timeout=OTA_TRANSFER_TIMEOUT_S,
            )

            # Half-close so the kernel knows the stream is done. Drain any
            # final ack bytes (the kernel may emit a status word, may not).
            try:
                writer.write_eof()
            except (NotImplementedError, OSError):
                pass
            try:
                await asyncio.wait_for(reader.read(64), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _resolve_endpoint(robot: RobotRecord) -> tuple[Optional[str], int]:
        """Pick (host, port) for the OTA socket.

        The host is ALWAYS taken from the brain-side peer address of the
        established robot connection — never from robot.meta.  Allowing robots
        to set ota_host/ota_port via heartbeat would let a rogue robot redirect
        the OTA socket to an attacker-controlled host (fleet.py already strips
        those keys at the meta-update layer; this is the belt-and-suspenders
        defence so the OTA path can never be redirected even if that strip is
        somehow bypassed).

        The port is fixed at OTA_DEFAULT_PORT (brain config), not robot-supplied.
        """
        if robot.writer is None:
            return None, OTA_DEFAULT_PORT
        try:
            peer = robot.writer.get_extra_info("peername")
            if peer is None:
                return None, OTA_DEFAULT_PORT
            return peer[0], OTA_DEFAULT_PORT
        except Exception:  # noqa: BLE001
            return None, OTA_DEFAULT_PORT


# ── DEV04: synchronous group dispatch ────────────────────────────────
#
# The `FleetOtaManager.start_job` path is *asynchronous*: it returns a job
# id immediately and runs the fan-out in the background.  For the
# `POST /v1/fleet/{group}/ota` REST endpoint we want the opposite: a
# synchronous "push to every robot in this group, then return the
# per-robot results" call.  The helper below factors out the one-shot
# push so the endpoint handler stays thin and so unit tests can mock the
# socket layer independently of the job-tracking state machine.


# Meta key robots use to advertise their fleet group.  This is *not* in
# `_META_KEYS_FORBIDDEN` so robots may set it via heartbeat — the worst
# a rogue robot can do by lying about its group is opt itself into / out
# of a deployment, not redirect the OTA socket (host/port still come
# from `_resolve_endpoint`).
FLEET_GROUP_META_KEY: str = "group"

# Result-dict keys for `push_firmware_to_robot` and `_dispatch_ota_to_group`.
DISPATCH_KEY_ROBOT_ID: str = "robot_id"
DISPATCH_KEY_STATUS: str = "status"
DISPATCH_KEY_BYTES_SENT: str = "bytes_sent"
DISPATCH_KEY_ERROR: str = "error"

DISPATCH_STATUS_OK: str = "ok"
DISPATCH_STATUS_ERROR: str = "error"


def robots_in_group(fleet: FleetManager, group: str) -> list[RobotRecord]:
    """Return every registered robot whose `meta['group']` matches `group`.

    Offline robots are included intentionally — the dispatcher will still
    try to push and record a per-robot failure, which gives the operator
    visibility into who was missed (vs. silently filtering them out).
    """
    return [r for r in fleet.all_robots() if r.meta.get(FLEET_GROUP_META_KEY) == group]


async def push_firmware_to_robot(
    robot: RobotRecord,
    image: bytes,
    sig: bytes,
    platform: str,
    fw_version: int,
    open_conn_fn: Optional[OpenConnectionFn] = None,
) -> dict[str, Any]:
    """Push one OTA image to a single robot, synchronously.

    Returns a result dict with the schema documented at module top:
    ``{robot_id, status, bytes_sent, error}``.  Never raises — failures
    are reported via the ``status="error"`` field so the caller can
    aggregate mixed results without a try/except per robot.

    No retry: the REST endpoint is operator-driven, so the operator
    sees the failure and re-issues if needed.  The job-tracking path
    (`FleetOtaManager`) is the place for retry/backoff.
    """
    rid = robot.robot_id
    result: dict[str, Any] = {
        DISPATCH_KEY_ROBOT_ID: rid,
        DISPATCH_KEY_STATUS: DISPATCH_STATUS_ERROR,
        DISPATCH_KEY_BYTES_SENT: 0,
        DISPATCH_KEY_ERROR: None,
    }

    if platform not in PLATFORM_IDS:
        msg = f"unknown platform '{platform}'"
        logger.warning("[FleetOTA dispatch] %s: %s", rid, msg)
        result[DISPATCH_KEY_ERROR] = msg
        return result

    if not robot.online or robot.writer is None:
        msg = "robot not online"
        logger.warning("[FleetOTA dispatch] %s: %s", rid, msg)
        result[DISPATCH_KEY_ERROR] = msg
        return result

    host, port = FleetOtaManager._resolve_endpoint(robot)
    if host is None:
        msg = "could not resolve peer address"
        logger.warning("[FleetOTA dispatch] %s: %s", rid, msg)
        result[DISPATCH_KEY_ERROR] = msg
        return result

    crc = zlib.crc32(image) & 0xFFFFFFFF
    header = build_ota_header(len(image), crc, fw_version, PLATFORM_IDS[platform])
    open_fn: OpenConnectionFn = open_conn_fn or _default_open_connection

    try:
        reader, writer = await open_fn(host, port)
    except Exception as e:  # noqa: BLE001
        msg = f"connect failed: {e}"
        logger.warning("[FleetOTA dispatch] %s: %s", rid, msg)
        result[DISPATCH_KEY_ERROR] = msg
        _metrics_mod.M.ota_pushes_total.labels(status=DISPATCH_STATUS_ERROR).inc()
        return result

    try:
        payload = header + image + sig
        writer.write(payload)
        await writer.drain()
        result[DISPATCH_KEY_BYTES_SENT] = len(payload)
        try:
            writer.write_eof()
        except (NotImplementedError, OSError):
            pass
        try:
            await asyncio.wait_for(reader.read(64), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            pass
        result[DISPATCH_KEY_STATUS] = DISPATCH_STATUS_OK
        _metrics_mod.M.ota_pushes_total.labels(status=DISPATCH_STATUS_OK).inc()
    except Exception as e:  # noqa: BLE001
        msg = f"send failed: {e}"
        logger.warning("[FleetOTA dispatch] %s: %s", rid, msg)
        result[DISPATCH_KEY_ERROR] = msg
        _metrics_mod.M.ota_pushes_total.labels(status=DISPATCH_STATUS_ERROR).inc()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass

    return result


async def _dispatch_ota_to_group(
    fleet: FleetManager,
    group: str,
    fw_bytes: bytes,
    sig_bytes: bytes,
    platform: str,
    fw_version: int,
    open_conn_fn: Optional[OpenConnectionFn] = None,
) -> list[dict[str, Any]]:
    """Push `fw_bytes` (verified Ed25519-signed) to every robot in `group`.

    Returns one result dict per robot.  An empty group raises ValueError
    so the caller can map that to a 404.  Signature verification is the
    caller's responsibility; this function is the post-verify dispatcher.

    Note: spec hint was ``(group, fw_bytes, sig_bytes)`` but an OTA
    header cannot be built without `platform`+`fw_version`, so those are
    threaded through here too.
    """
    targets = robots_in_group(fleet, group)
    if not targets:
        msg = f"no robots registered in group '{group}'"
        logger.warning("[FleetOTA dispatch] %s", msg)
        raise ValueError(msg)

    sem = asyncio.Semaphore(OTA_FAN_OUT_CONCURRENCY)

    async def _one(robot: RobotRecord) -> dict[str, Any]:
        async with sem:
            return await push_firmware_to_robot(
                robot=robot,
                image=fw_bytes,
                sig=sig_bytes,
                platform=platform,
                fw_version=fw_version,
                open_conn_fn=open_conn_fn,
            )

    return list(await asyncio.gather(*(_one(r) for r in targets)))
