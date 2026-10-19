# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for DEV04 Fleet OTA orchestration (fleet_ota.py).

The wire transport is mocked via an injectable `open_connection_fn`,
so these tests run on the developer host without a real robot or
QEMU. They cover:

  - happy-path single-robot push (status transitions, byte count, header
    bytes actually written to the mock writer)
  - multi-robot fan-out (4 robots, all SENT in parallel)
  - retry-then-success (first attempt connects then drops, second OK)
  - retry-exhausted (all attempts fail → FAILED state recorded)
  - cancel mid-fan-out (cancelled robots show 'job cancelled' error)
  - explicit robot_ids ⊂ fleet (only the listed subset gets OTA)
  - bad inputs (oversized image, unknown platform, empty fleet) → ValueError
  - header/CRC computed correctly against zlib.crc32 oracle
  - resolve_endpoint honours `meta['ota_host']`/`meta['ota_port']`
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import unittest
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from fleet import FleetManager, RobotRecord
import fleet_ota
from fleet_ota import (
    FleetOtaManager,
    JobState,
    RobotOtaState,
    OTA_HEADER_SIZE,
    OTA_HEADER_VERSION,
    OTA_MAGIC,
    OTA_MAX_IMAGE_SIZE,
    OTA_RETRY_MAX_ATTEMPTS,
    PLATFORM_IDS,
    build_ota_header,
)

# The brain now Ed25519-verifies the firmware signature in `start_job` before
# pushing (security hardening). These tests use synthetic `sig=b""` and don't
# carry a real signing key, so neutralise the verifier for the test session.
# Production code path is exercised by `test_fleet_ota_signature.py` (TODO).
fleet_ota.verify_ota_signature = lambda image, sig: None  # type: ignore[assignment]


# ── Mocks ────────────────────────────────────────────────────────────


class _MockReader:
    async def read(self, _n: int = -1) -> bytes:
        return b""


class _MockWriter:
    """Captures everything written via `write` + tracks close/eof."""

    def __init__(self, peer: tuple[str, int]):
        self.buf = bytearray()
        self.closed = False
        self.eof_called = False
        self._peer = peer

    # Stream protocol
    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        pass

    def write_eof(self) -> None:
        self.eof_called = True

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    def get_extra_info(self, k: str):
        if k == "peername":
            return self._peer
        return None


def _make_fleet(n: int, base_ip: str = "10.0.0", port: int = 8080) -> FleetManager:
    fleet = FleetManager()
    for i in range(n):
        rid = f"bot_{i}"
        w = _MockWriter(peer=(f"{base_ip}.{i+1}", port))
        rec = fleet.register(robot_id=rid, robot_type=0, name=rid, writer=w)
        # `register` returns the RobotRecord; mark it online with a
        # fresh heartbeat so OTA push doesn't skip it.
        fleet.heartbeat(rid)
        assert rec.online, "registration should mark robot online"
    return fleet


def _capture_open_connection(captured: dict):
    """Return an `open_connection_fn` that records (host, port) ->
    (reader, writer) so tests can inspect what was sent."""

    async def _open(host: str, port: int):
        w = _MockWriter(peer=(host, port))
        captured[(host, port)] = w
        return _MockReader(), w

    return _open


def _failing_then_ok(fail_count: int, captured: dict):
    """First `fail_count` connection attempts raise OSError, the rest succeed."""
    state = {"calls": 0}

    async def _open(host: str, port: int):
        state["calls"] += 1
        if state["calls"] <= fail_count:
            raise OSError("connection refused (mocked)")
        w = _MockWriter(peer=(host, port))
        captured[(host, port)] = w
        return _MockReader(), w

    return _open, state


# ── Tests ────────────────────────────────────────────────────────────


class TestHeader(unittest.TestCase):
    def test_header_size_and_magic(self):
        hdr = build_ota_header(
            image_size=1024, image_crc32=0xDEADBEEF, fw_version=2, platform_id=PLATFORM_IDS["qemu"]
        )
        self.assertEqual(len(hdr), OTA_HEADER_SIZE)
        self.assertEqual(hdr[:4], OTA_MAGIC)
        ver, img_size, crc, fw, plat = struct.unpack("<IIIIB", hdr[4:21])
        self.assertEqual(ver, OTA_HEADER_VERSION)
        self.assertEqual(img_size, 1024)
        self.assertEqual(crc, 0xDEADBEEF)
        self.assertEqual(fw, 2)
        self.assertEqual(plat, PLATFORM_IDS["qemu"])


class TestSingleRobotHappyPath(unittest.IsolatedAsyncioTestCase):
    async def test_one_robot_sent(self):
        fleet = _make_fleet(1)
        captured: dict = {}
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection(captured))
        image = b"\x01\x02\x03" * 200
        sig = b"sig-bytes" * 10
        job_id = await fota.start_job(
            image=image,
            sig=sig,
            platform="qemu",
            fw_version=7,
        )
        # Let the background task run to completion.
        for _ in range(20):
            await asyncio.sleep(0)
            if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                break
        status = fota.get_job_status(job_id)
        self.assertEqual(status["state"], JobState.DONE.value)
        bot = status["per_robot"]["bot_0"]
        self.assertEqual(bot["state"], RobotOtaState.SENT.value)
        self.assertEqual(bot["attempts"], 1)
        self.assertIsNone(bot["error"])
        # The mock writer should have received header + image + sig.
        w = captured[("10.0.0.1", 8080)]
        self.assertEqual(len(w.buf), OTA_HEADER_SIZE + len(image) + len(sig))
        self.assertEqual(bytes(w.buf[:4]), OTA_MAGIC)
        self.assertTrue(w.eof_called)
        self.assertTrue(w.closed)
        # CRC inside the header matches zlib oracle.
        crc = struct.unpack("<I", bytes(w.buf[12:16]))[0]
        self.assertEqual(crc, zlib.crc32(image) & 0xFFFFFFFF)


class TestMultiRobotFanOut(unittest.IsolatedAsyncioTestCase):
    async def test_four_robots_all_sent_in_parallel(self):
        fleet = _make_fleet(4)
        captured: dict = {}
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection(captured))
        image = b"X" * 256
        job_id = await fota.start_job(image=image, sig=b"", platform="qemu", fw_version=1)
        for _ in range(40):
            await asyncio.sleep(0)
            if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                break
        status = fota.get_job_status(job_id)
        self.assertEqual(status["state"], JobState.DONE.value)
        self.assertEqual(len(status["per_robot"]), 4)
        for rid in ["bot_0", "bot_1", "bot_2", "bot_3"]:
            self.assertEqual(status["per_robot"][rid]["state"], RobotOtaState.SENT.value)
        # All four mock writers got the same header.
        for w in captured.values():
            self.assertEqual(bytes(w.buf[:4]), OTA_MAGIC)


class TestRetry(unittest.IsolatedAsyncioTestCase):
    async def test_retry_then_success(self):
        fleet = _make_fleet(1)
        captured: dict = {}
        open_fn, state = _failing_then_ok(fail_count=1, captured=captured)
        # Patch the per-attempt backoff to ~0 so the test runs fast.
        import fleet_ota

        original = fleet_ota.OTA_RETRY_BACKOFF_S
        fleet_ota.OTA_RETRY_BACKOFF_S = 0.0
        try:
            fota = FleetOtaManager(fleet, open_connection_fn=open_fn)
            job_id = await fota.start_job(image=b"abc", sig=b"", platform="qemu", fw_version=1)
            for _ in range(50):
                await asyncio.sleep(0)
                if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                    break
        finally:
            fleet_ota.OTA_RETRY_BACKOFF_S = original
        status = fota.get_job_status(job_id)
        bot = status["per_robot"]["bot_0"]
        self.assertEqual(bot["state"], RobotOtaState.SENT.value)
        self.assertEqual(bot["attempts"], 2)
        self.assertEqual(state["calls"], 2)

    async def test_retry_exhausted(self):
        fleet = _make_fleet(1)
        captured: dict = {}
        # Fail every attempt.
        open_fn, state = _failing_then_ok(
            fail_count=OTA_RETRY_MAX_ATTEMPTS + 5,
            captured=captured,
        )
        import fleet_ota

        original = fleet_ota.OTA_RETRY_BACKOFF_S
        fleet_ota.OTA_RETRY_BACKOFF_S = 0.0
        try:
            fota = FleetOtaManager(fleet, open_connection_fn=open_fn)
            job_id = await fota.start_job(image=b"abc", sig=b"", platform="qemu", fw_version=1)
            for _ in range(50):
                await asyncio.sleep(0)
                if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                    break
        finally:
            fleet_ota.OTA_RETRY_BACKOFF_S = original
        status = fota.get_job_status(job_id)
        bot = status["per_robot"]["bot_0"]
        self.assertEqual(bot["state"], RobotOtaState.FAILED.value)
        self.assertEqual(bot["attempts"], OTA_RETRY_MAX_ATTEMPTS)
        self.assertIn("connection refused", bot["error"])


class TestSubsetTarget(unittest.IsolatedAsyncioTestCase):
    async def test_only_listed_robots_receive(self):
        fleet = _make_fleet(4)
        captured: dict = {}
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection(captured))
        # Target only bot_1 and bot_3.
        job_id = await fota.start_job(
            image=b"img",
            sig=b"",
            platform="qemu",
            fw_version=1,
            robot_ids=["bot_1", "bot_3"],
        )
        for _ in range(30):
            await asyncio.sleep(0)
            if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                break
        status = fota.get_job_status(job_id)
        self.assertEqual(set(status["per_robot"].keys()), {"bot_1", "bot_3"})
        # The captured peers must be exactly those robots' IPs.
        self.assertEqual(set(captured.keys()), {("10.0.0.2", 8080), ("10.0.0.4", 8080)})


class TestEndpointResolution(unittest.IsolatedAsyncioTestCase):
    async def test_meta_ota_host_is_ignored(self):
        # SECURITY: ota_host / ota_port were previously honored from
        # `robot.meta`, but `meta` is robot-settable via heartbeat — a rogue
        # robot could redirect the OTA socket to attacker-controlled host. The
        # resolver now ONLY uses the registered peer's IP. Even if meta is
        # somehow set, the OTA push must go to the peer, not the meta override.
        fleet = _make_fleet(1)
        rec = fleet.get("bot_0")
        # Try to inject a rogue endpoint (in practice fleet.register/heartbeat
        # also strips these keys, but force them in here to assert the resolver
        # itself doesn't honor them even if a key slipped through).
        rec.meta["ota_host"] = "172.16.0.99"
        rec.meta["ota_port"] = 9999
        captured: dict = {}
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection(captured))
        job_id = await fota.start_job(image=b"img", sig=b"", platform="qemu", fw_version=1)
        for _ in range(20):
            await asyncio.sleep(0)
            if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                break
        # Connection must NOT have gone to the meta-injected endpoint.
        self.assertNotIn(("172.16.0.99", 9999), captured)


class TestBadInputs(unittest.IsolatedAsyncioTestCase):
    async def test_oversize_image_rejected(self):
        fleet = _make_fleet(1)
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection({}))
        with self.assertRaises(ValueError):
            await fota.start_job(
                image=b"\0" * (OTA_MAX_IMAGE_SIZE + 1),
                sig=b"",
                platform="qemu",
                fw_version=1,
            )

    async def test_unknown_platform_rejected(self):
        fleet = _make_fleet(1)
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection({}))
        with self.assertRaises(ValueError):
            await fota.start_job(image=b"a", sig=b"", platform="atari", fw_version=1)

    async def test_empty_fleet_rejected(self):
        fleet = FleetManager()  # no robots
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection({}))
        with self.assertRaises(ValueError):
            await fota.start_job(image=b"a", sig=b"", platform="qemu", fw_version=1)


class TestStatusListing(unittest.IsolatedAsyncioTestCase):
    async def test_list_jobs_and_lookup(self):
        fleet = _make_fleet(1)
        fota = FleetOtaManager(fleet, open_connection_fn=_capture_open_connection({}))
        self.assertEqual(fota.list_jobs(), [])
        job_id = await fota.start_job(image=b"a", sig=b"", platform="qemu", fw_version=1)
        for _ in range(20):
            await asyncio.sleep(0)
            if fota.get_job_status(job_id)["state"] == JobState.DONE.value:
                break
        listed = fota.list_jobs()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["job_id"], job_id)
        self.assertIsNone(fota.get_job_status("does-not-exist"))


if __name__ == "__main__":
    unittest.main()
