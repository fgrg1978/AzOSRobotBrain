# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for the S2 control-plane / data-plane split scaffolding.

Coverage:
  1. ShardCoordinator routes consistently to the same data plane.
  2. ShardCoordinator re-distributes after node removal.
  3. Control plane /v1/robots/{id}/route returns correct data plane (mock ring).
  4. Control plane /v1/fleet returns fleet status from injected callback.
  5. Control plane /v1/ota returns a job_id.
  6. Data-plane worker refuses a robot it doesn't own (in-process mode).
  7. Data-plane worker accepts a robot it DOES own (in-process mode).
  8. In single-process mode (empty ring) data plane accepts every robot.
  9. start_split.py with --data-planes 0 --duration 2 runs and exits cleanly.
  10. start_split.py with --data-planes > 0 without --allow-no-shared-state exits with error.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
import unittest
from typing import Any

import pytest

# Ensure the project root is importable from pytest regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from control_plane.discovery import ShardCoordinator
from control_plane.auth import BearerAuth, ENV_CP_API_KEY, ENV_ALLOW_INSECURE
from control_plane.ota_coordinator import OtaCoordinator, OtaJobState
from control_plane.main import ControlPlaneServer, _parse_request, _respond
from data_plane import CP_INPROCESS_SENTINEL
from data_plane.worker import OwnershipChecker

# ---------------------------------------------------------------------------
# Constants — no magic numbers in tests either
# ---------------------------------------------------------------------------

_CP_PORT_TEST: int = 0  # ephemeral / unused in unit tests
_DP_ADDR_A: str = "localhost:9100"
_DP_ADDR_B: str = "localhost:9101"
_ROBOT_ID_1: str = "bot_alpha"
_ROBOT_ID_2: str = "bot_beta"
_ROBOT_ID_3: str = "bot_gamma"
_DURATION_SMOKE: int = 2  # seconds for the smoke-start test
_PROBE_COUNT: int = 200  # robots to probe when searching for a shard assignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeWriter:
    """Minimal asyncio.StreamWriter stand-in that captures written bytes."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def get_extra_info(self, key: str, default: Any = None) -> Any:
        if key == "peername":
            return ("127.0.0.1", 9999)
        return default

    def response_body(self) -> dict[str, Any]:
        """Parse the JSON body from the buffered HTTP response."""
        sep = b"\r\n\r\n"
        idx = self.buf.find(sep)
        if idx == -1:
            return {}
        return json.loads(self.buf[idx + 4 :])

    def status_code(self) -> int:
        """Parse the HTTP status code from the first response line."""
        first_line = self.buf.split(b"\r\n")[0].decode(errors="replace")
        parts = first_line.split(" ", 2)
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0


def _make_http_request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    token: str | None = None,
) -> bytes:
    """Build a raw HTTP/1.1 request."""
    payload = json.dumps(body).encode() if body is not None else b""
    lines = [
        f"{method} {path} HTTP/1.1",
        "Host: localhost",
        f"Content-Length: {len(payload)}",
    ]
    if token:
        lines.append(f"Authorization: Bearer {token}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode() + payload


def _load_start_split() -> Any:
    """Import tools/start_split.py as a module (avoids polluting sys.modules)."""
    spec = importlib.util.spec_from_file_location(
        "start_split",
        os.path.join(_PROJECT_ROOT, "tools", "start_split.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Test 1: ShardCoordinator consistent routing
# ---------------------------------------------------------------------------


class TestShardCoordinatorConsistency(unittest.TestCase):
    def test_routes_consistently(self) -> None:
        """Same robot_id always maps to the same data plane."""
        coord = ShardCoordinator(nodes=[_DP_ADDR_A, _DP_ADDR_B])
        first = coord.assign(_ROBOT_ID_1)
        for _ in range(10):
            self.assertEqual(
                coord.assign(_ROBOT_ID_1),
                first,
                "Routing must be deterministic",
            )


# ---------------------------------------------------------------------------
# Test 2: ShardCoordinator redistribution after node removal
# ---------------------------------------------------------------------------


class TestShardCoordinatorRemoval(unittest.TestCase):
    def test_redistributes_after_removal(self) -> None:
        """After removing a node, robots that mapped to it are reassigned."""
        coord = ShardCoordinator(nodes=[_DP_ADDR_A, _DP_ADDR_B])
        robot_ids = [f"robot_{i}" for i in range(20)]
        before: dict[str, str | None] = {r: coord.assign(r) for r in robot_ids}

        coord.remove_node(_DP_ADDR_B)
        after: dict[str, str | None] = {r: coord.assign(r) for r in robot_ids}

        self.assertTrue(
            all(v == _DP_ADDR_A for v in after.values()),
            "After removing _DP_ADDR_B, all robots must route to _DP_ADDR_A",
        )
        changed = [r for r in robot_ids if before[r] != after[r]]
        self.assertGreater(
            len(changed),
            0,
            "Some robots should have moved after node removal",
        )


# ---------------------------------------------------------------------------
# Test 3: Control plane /v1/robots/{id}/route
# ---------------------------------------------------------------------------


class TestCpRoute(unittest.IsolatedAsyncioTestCase):
    async def test_route_returns_correct_data_plane(self) -> None:
        """POST /v1/robots/{id}/route returns the expected data-plane address."""
        os.environ[ENV_ALLOW_INSECURE] = "1"
        try:
            coord = ShardCoordinator(nodes=[_DP_ADDR_A, _DP_ADDR_B])
            auth = BearerAuth()
            ota = OtaCoordinator()
            server = ControlPlaneServer(
                coordinator=coord,
                ota_coordinator=ota,
                auth=auth,
                port=_CP_PORT_TEST,
            )
            expected_dp = coord.assign(_ROBOT_ID_1)
            self.assertIsNotNone(expected_dp)

            raw = _make_http_request("POST", f"/v1/robots/{_ROBOT_ID_1}/route")
            writer = _FakeWriter()
            reader = asyncio.StreamReader()
            reader.feed_data(raw)
            reader.feed_eof()
            await server._handle(reader, writer)  # type: ignore[arg-type]

            self.assertEqual(writer.status_code(), 200)
            body = writer.response_body()
            self.assertEqual(body["robot_id"], _ROBOT_ID_1)
            self.assertEqual(body["data_plane"], expected_dp)
        finally:
            os.environ.pop(ENV_ALLOW_INSECURE, None)


# ---------------------------------------------------------------------------
# Test 4: Control plane /v1/fleet
# ---------------------------------------------------------------------------


class TestCpFleet(unittest.IsolatedAsyncioTestCase):
    async def test_fleet_returns_status(self) -> None:
        """GET /v1/fleet returns fleet status from the injected callback."""
        os.environ[ENV_ALLOW_INSECURE] = "1"
        try:
            coord = ShardCoordinator(nodes=[_DP_ADDR_A])
            auth = BearerAuth()
            ota = OtaCoordinator()
            expected_status: dict[str, Any] = {"robots": [{"id": _ROBOT_ID_1, "online": True}]}
            server = ControlPlaneServer(
                coordinator=coord,
                ota_coordinator=ota,
                auth=auth,
                fleet_status_fn=lambda: expected_status,
                port=_CP_PORT_TEST,
            )

            raw = _make_http_request("GET", "/v1/fleet")
            writer = _FakeWriter()
            reader = asyncio.StreamReader()
            reader.feed_data(raw)
            reader.feed_eof()
            await server._handle(reader, writer)  # type: ignore[arg-type]

            self.assertEqual(writer.status_code(), 200)
            self.assertEqual(writer.response_body(), expected_status)
        finally:
            os.environ.pop(ENV_ALLOW_INSECURE, None)


# ---------------------------------------------------------------------------
# Test 5: Control plane /v1/ota returns job_id
# ---------------------------------------------------------------------------


class TestCpOta(unittest.IsolatedAsyncioTestCase):
    async def test_ota_returns_job_id(self) -> None:
        """POST /v1/ota returns a job_id with state RUNNING."""
        os.environ[ENV_ALLOW_INSECURE] = "1"
        try:
            coord = ShardCoordinator(nodes=[_DP_ADDR_A])
            auth = BearerAuth()
            ota = OtaCoordinator()
            server = ControlPlaneServer(
                coordinator=coord,
                ota_coordinator=ota,
                auth=auth,
                port=_CP_PORT_TEST,
            )

            ota_payload: dict[str, Any] = {
                "robot_ids": [_ROBOT_ID_1],
                "platform": "qemu",
                "fw_version": 2,
            }
            raw = _make_http_request("POST", "/v1/ota", body=ota_payload)
            writer = _FakeWriter()
            reader = asyncio.StreamReader()
            reader.feed_data(raw)
            reader.feed_eof()
            await server._handle(reader, writer)  # type: ignore[arg-type]

            self.assertEqual(writer.status_code(), 200)
            body = writer.response_body()
            self.assertIn("job_id", body)
            self.assertEqual(body["state"], "RUNNING")
        finally:
            os.environ.pop(ENV_ALLOW_INSECURE, None)


# ---------------------------------------------------------------------------
# Test 6: Data-plane worker refuses unowned robot (in-process mode)
# ---------------------------------------------------------------------------


class TestDpOwnershipRefuse(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_unowned_robot(self) -> None:
        """Worker with self_node A refuses robot owned by node B."""
        coord = ShardCoordinator(nodes=[_DP_ADDR_A, _DP_ADDR_B])
        robot_on_b: str | None = None
        for i in range(_PROBE_COUNT):
            r = f"probe_{i}"
            if coord.assign(r) == _DP_ADDR_B:
                robot_on_b = r
                break

        if robot_on_b is None:
            self.skipTest(
                "Could not find a robot assigned to _DP_ADDR_B — " "ring distribution edge case"
            )

        checker = OwnershipChecker(_DP_ADDR_A, coordinator=coord)
        result = await checker.check_ownership(robot_on_b)
        self.assertFalse(
            result,
            f"{_DP_ADDR_A} must NOT own robot assigned to {_DP_ADDR_B}",
        )


# ---------------------------------------------------------------------------
# Test 7: Data-plane worker accepts owned robot (in-process mode)
# ---------------------------------------------------------------------------


class TestDpOwnershipAccept(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_owned_robot(self) -> None:
        """Worker with self_node A accepts a robot the ring assigns to A."""
        coord = ShardCoordinator(nodes=[_DP_ADDR_A, _DP_ADDR_B])
        robot_on_a: str | None = None
        for i in range(_PROBE_COUNT):
            r = f"probe_{i}"
            if coord.assign(r) == _DP_ADDR_A:
                robot_on_a = r
                break

        if robot_on_a is None:
            self.skipTest(
                "Could not find a robot assigned to _DP_ADDR_A — " "ring distribution edge case"
            )

        checker = OwnershipChecker(_DP_ADDR_A, coordinator=coord)
        result = await checker.check_ownership(robot_on_a)
        self.assertTrue(
            result,
            f"{_DP_ADDR_A} must own robot assigned to it",
        )


# ---------------------------------------------------------------------------
# Test 8: Single-process mode (empty ring) accepts every robot
# ---------------------------------------------------------------------------


class TestDpInprocessEmptyRing(unittest.IsolatedAsyncioTestCase):
    async def test_empty_ring_accepts_all(self) -> None:
        """In single-process mode with empty ring, all robots are accepted."""
        coord = ShardCoordinator()  # empty ring → assign() returns None → accept
        checker = OwnershipChecker(CP_INPROCESS_SENTINEL, coordinator=coord)
        for rid in [_ROBOT_ID_1, _ROBOT_ID_2, _ROBOT_ID_3]:
            result = await checker.check_ownership(rid)
            self.assertTrue(
                result,
                f"Empty ring should accept all robots; {rid} was rejected",
            )


# ---------------------------------------------------------------------------
# Test 9: start_split.py --data-planes 0 --duration 2 runs and exits cleanly
# ---------------------------------------------------------------------------


class TestStartSplitSingleProcess(unittest.TestCase):
    def test_smoke_run(self) -> None:
        """start_split.py with --data-planes 0 --duration 2 runs and exits cleanly."""
        os.environ["AZOS_BRAIN_ALLOW_INSECURE"] = "1"
        try:
            mod = _load_start_split()
            start = time.monotonic()
            mod.run(["--data-planes", "0", "--duration", str(_DURATION_SMOKE)])
            elapsed = time.monotonic() - start
            self.assertGreaterEqual(
                elapsed,
                _DURATION_SMOKE - 0.5,
                f"Expected run to last ~{_DURATION_SMOKE}s, got {elapsed:.2f}s",
            )
            self.assertLess(
                elapsed,
                _DURATION_SMOKE + 5.0,
                f"Run took too long: {elapsed:.2f}s",
            )
        finally:
            os.environ.pop("AZOS_BRAIN_ALLOW_INSECURE", None)


# ---------------------------------------------------------------------------
# Test 10: start_split.py --data-planes > 0 without --allow-no-shared-state
# ---------------------------------------------------------------------------


class TestStartSplitGuard(unittest.TestCase):
    def test_multiprocess_requires_guard(self) -> None:
        """Multi-process mode without --allow-no-shared-state must exit with error."""
        mod = _load_start_split()
        with self.assertRaises(SystemExit) as ctx:
            mod.run(["--data-planes", "2"])
        self.assertNotEqual(
            ctx.exception.code,
            0,
            "--data-planes > 0 without --allow-no-shared-state must exit non-zero",
        )
