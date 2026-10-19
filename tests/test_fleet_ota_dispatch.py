# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""DEV04 — tests for POST /v1/fleet/{group}/ota.

Covers:
  - happy path: 3-robot group, all robots receive the firmware.
  - auth rejection: no bearer token -> 401.
  - empty / unknown group -> 404.
  - bad Ed25519 signature -> 400.
  - partial failure: 1 of 3 connect attempts raises -> 200 with mixed results.

The kernel TCP layer is mocked via ``open_connection_fn`` on a
``FleetOtaManager``-independent code path (we exercise the new
``_dispatch_ota_to_group`` helper directly + the endpoint via an
in-process HTTP client).
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from typing import Any, Optional
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import api as api_module
import fleet_ota
from api import APIServer
from fleet import FleetManager

# ── Constants used by the tests (no magic numbers) ────────────────────

TEST_API_KEY: str = "test-token-dev04"
TEST_PEER_PORT: int = 8080
TEST_GROUP_NAME: str = "field"
TEST_OTHER_GROUP: str = "lab"
TEST_FIRMWARE_BYTES: bytes = b"\xaa\xbb\xcc" * 64
TEST_SIGNATURE_BYTES: bytes = b"sig-bytes-stub-not-real-just-non-empty"
TEST_BOUNDARY: str = "----PHANES-TEST-BOUNDARY-9c1e7a"
TEST_BAD_BOUNDARY: bytes = b"----BAD-SIG-BOUNDARY"

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORISED = 401
HTTP_NOT_FOUND = 404


# ── Mocks ─────────────────────────────────────────────────────────────


class _MockWriter:
    """Bare-bones asyncio.StreamWriter compatible mock."""

    def __init__(self, peer: tuple[str, int]) -> None:
        self.buf: bytearray = bytearray()
        self.closed: bool = False
        self.eof_called: bool = False
        self._peer: tuple[str, int] = peer

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

    def get_extra_info(self, key: str) -> Any:
        if key == "peername":
            return self._peer
        return None


class _MockReader:
    async def read(self, _n: int = -1) -> bytes:
        return b""


class _StubBrain:
    """Minimal brain stub: only the bits APIServer touches at construction."""

    def __init__(self, fleet: FleetManager) -> None:
        self.fleet_manager: FleetManager = fleet
        self.config: dict[str, Any] = {}


def _make_fleet_with_group(n_in_group: int, group: str, n_outside: int = 0) -> FleetManager:
    """Build a FleetManager with `n_in_group` robots tagged `meta['group']=group`."""
    fleet = FleetManager()
    for i in range(n_in_group):
        rid = f"bot_{i}"
        peer_ip = f"10.0.0.{i + 1}"
        w = _MockWriter(peer=(peer_ip, TEST_PEER_PORT))
        fleet.register(robot_id=rid, robot_type=0, name=rid, writer=w, meta={"group": group})
        fleet.heartbeat(rid)
    for i in range(n_outside):
        rid = f"out_{i}"
        peer_ip = f"10.0.1.{i + 1}"
        w = _MockWriter(peer=(peer_ip, TEST_PEER_PORT))
        fleet.register(
            robot_id=rid, robot_type=0, name=rid, writer=w, meta={"group": TEST_OTHER_GROUP}
        )
        fleet.heartbeat(rid)
    return fleet


def _capture_open_connection(captured: dict[tuple[str, int], _MockWriter]):
    async def _open(host: str, port: int) -> tuple[Any, Any]:
        w = _MockWriter(peer=(host, port))
        captured[(host, port)] = w
        return _MockReader(), w

    return _open


def _failing_then_ok_open(fail_hosts: set[str], captured: dict[tuple[str, int], _MockWriter]):
    """An open_connection_fn that raises for hosts in `fail_hosts`."""

    async def _open(host: str, port: int) -> tuple[Any, Any]:
        if host in fail_hosts:
            raise OSError(f"refused (mock) for {host}")
        w = _MockWriter(peer=(host, port))
        captured[(host, port)] = w
        return _MockReader(), w

    return _open


def _build_multipart(
    firmware: bytes = TEST_FIRMWARE_BYTES,
    signature: bytes = TEST_SIGNATURE_BYTES,
    boundary: str = TEST_BOUNDARY,
    include_firmware: bool = True,
    include_signature: bool = True,
) -> bytes:
    """Build a minimal multipart/form-data body for testing."""
    parts: list[bytes] = []
    if include_firmware:
        parts.append(
            f"--{boundary}\r\n".encode()
            + b'Content-Disposition: form-data; name="firmware"; '
            + b'filename="kernel.bin"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + firmware
            + b"\r\n"
        )
    if include_signature:
        parts.append(
            f"--{boundary}\r\n".encode()
            + b'Content-Disposition: form-data; name="signature"; '
            + b'filename="kernel.sig"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + signature
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


# ── Test: helper-level dispatch (no HTTP) ─────────────────────────────


class TestDispatchHelper(unittest.IsolatedAsyncioTestCase):
    """Direct tests of fleet_ota._dispatch_ota_to_group()."""

    async def test_three_robots_all_receive_firmware(self) -> None:
        fleet = _make_fleet_with_group(3, TEST_GROUP_NAME, n_outside=2)
        captured: dict[tuple[str, int], _MockWriter] = {}
        results = await fleet_ota._dispatch_ota_to_group(
            fleet=fleet,
            group=TEST_GROUP_NAME,
            fw_bytes=TEST_FIRMWARE_BYTES,
            sig_bytes=TEST_SIGNATURE_BYTES,
            platform="qemu",
            fw_version=1,
            open_conn_fn=_capture_open_connection(captured),
        )
        # 3 in-group robots dispatched; the 2 outsiders ignored.
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["status"], "ok")
            self.assertIsNone(r["error"])
            # Header (24) + image + sig actually written.
            self.assertEqual(
                r["bytes_sent"],
                fleet_ota.OTA_HEADER_SIZE + len(TEST_FIRMWARE_BYTES) + len(TEST_SIGNATURE_BYTES),
            )
        # 2 outsiders were NOT contacted.
        self.assertEqual(len(captured), 3)

    async def test_empty_group_raises_value_error(self) -> None:
        fleet = _make_fleet_with_group(0, TEST_GROUP_NAME)
        with self.assertRaises(ValueError):
            await fleet_ota._dispatch_ota_to_group(
                fleet=fleet,
                group=TEST_GROUP_NAME,
                fw_bytes=TEST_FIRMWARE_BYTES,
                sig_bytes=TEST_SIGNATURE_BYTES,
                platform="qemu",
                fw_version=1,
            )

    async def test_partial_failure_mixed_results(self) -> None:
        fleet = _make_fleet_with_group(3, TEST_GROUP_NAME)
        captured: dict[tuple[str, int], _MockWriter] = {}
        # Make the second robot's host refuse the connection.
        results = await fleet_ota._dispatch_ota_to_group(
            fleet=fleet,
            group=TEST_GROUP_NAME,
            fw_bytes=TEST_FIRMWARE_BYTES,
            sig_bytes=TEST_SIGNATURE_BYTES,
            platform="qemu",
            fw_version=1,
            open_conn_fn=_failing_then_ok_open({"10.0.0.2"}, captured),
        )
        statuses = sorted(r["status"] for r in results)
        self.assertEqual(statuses, ["error", "ok", "ok"])
        # The error result has bytes_sent==0 and a populated error field.
        bad = next(r for r in results if r["status"] == "error")
        self.assertEqual(bad["bytes_sent"], 0)
        self.assertIsNotNone(bad["error"])
        self.assertIn("refused", bad["error"])


# ── Test: full HTTP round-trip through APIServer ──────────────────────


class _InMemoryWriter:
    """Captures bytes written by APIServer's response path."""

    def __init__(self) -> None:
        self.buf: bytearray = bytearray()
        self.closed: bool = False

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def get_extra_info(self, key: str) -> Any:
        return ("127.0.0.1", 65000) if key == "peername" else None


def _parse_http_response(raw: bytes) -> tuple[int, bytes]:
    """Return (status_code, body) from a raw HTTP response."""
    head_end = raw.find(b"\r\n\r\n")
    if head_end == -1:
        raise AssertionError(f"no header terminator in: {raw[:80]!r}")
    head = raw[:head_end].decode("latin-1", errors="replace")
    body = raw[head_end + 4 :]
    status_line = head.split("\r\n", 1)[0]
    status_code = int(status_line.split(" ", 2)[1])
    return status_code, body


async def _post_via_api(
    api: APIServer,
    path: str,
    body: bytes,
    *,
    bearer: Optional[str],
    content_type: str = "multipart/form-data; boundary=" + TEST_BOUNDARY,
) -> tuple[int, bytes]:
    """Exercise APIServer._route by calling it directly with a stub writer."""
    headers: dict[str, str] = {
        "host": "test",
        "content-type": content_type,
        "content-length": str(len(body)),
    }
    if bearer is not None:
        headers["authorization"] = f"Bearer {bearer}"
    writer = _InMemoryWriter()
    # Authorise check first (matches what _handle does).
    if not api._is_authorised(path, headers):
        # Mirror the 401 path in _handle.
        from api import _response

        _response(writer, 401, {"error": "unauthorised"})
        return _parse_http_response(bytes(writer.buf))
    await api._route("POST", path, body, writer, peer_str="127.0.0.1:65000")
    return _parse_http_response(bytes(writer.buf))


def _make_api() -> tuple[APIServer, FleetManager]:
    fleet = _make_fleet_with_group(3, TEST_GROUP_NAME, n_outside=1)
    brain = _StubBrain(fleet)
    api = APIServer(brain, port=0, api_key=TEST_API_KEY)  # type: ignore[arg-type]
    return api, fleet


class TestV1FleetGroupOtaEndpoint(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        # Bypass Ed25519 verification for happy-path tests by stubbing the
        # function imported inside the handler.  The dedicated bad-sig
        # test re-enables a stub that ALWAYS raises.
        self._orig_verify = fleet_ota.verify_ota_signature
        fleet_ota.verify_ota_signature = lambda image, sig: None  # type: ignore[assignment]
        # Patch the connection factory used by the synchronous push path.
        self._captured: dict[tuple[str, int], _MockWriter] = {}
        self._patch_open = patch.object(
            fleet_ota,
            "_default_open_connection",
            _capture_open_connection(self._captured),
        )
        self._patch_open.start()

    async def asyncTearDown(self) -> None:
        fleet_ota.verify_ota_signature = self._orig_verify  # type: ignore[assignment]
        self._patch_open.stop()

    async def test_happy_path_three_robot_group(self) -> None:
        api, _ = _make_api()
        body = _build_multipart()
        status, resp_body = await _post_via_api(
            api,
            f"/v1/fleet/{TEST_GROUP_NAME}/ota",
            body,
            bearer=TEST_API_KEY,
        )
        self.assertEqual(status, HTTP_OK, msg=resp_body)
        import json

        parsed = json.loads(resp_body)
        self.assertIn("results", parsed)
        self.assertEqual(len(parsed["results"]), 3)
        for r in parsed["results"]:
            self.assertEqual(r["status"], "ok")
            self.assertIsNone(r["error"])
            self.assertGreater(r["bytes_sent"], 0)
        # All 3 in-group robots got the firmware; outsiders did not.
        self.assertEqual(len(self._captured), 3)

    async def test_missing_auth_token_returns_401(self) -> None:
        api, _ = _make_api()
        body = _build_multipart()
        status, _resp = await _post_via_api(
            api,
            f"/v1/fleet/{TEST_GROUP_NAME}/ota",
            body,
            bearer=None,
        )
        self.assertEqual(status, HTTP_UNAUTHORISED)

    async def test_unknown_group_returns_404(self) -> None:
        api, _ = _make_api()
        body = _build_multipart()
        status, _resp = await _post_via_api(
            api,
            "/v1/fleet/nonexistent/ota",
            body,
            bearer=TEST_API_KEY,
        )
        self.assertEqual(status, HTTP_NOT_FOUND)

    async def test_bad_signature_returns_400(self) -> None:
        # Re-install a verify_ota_signature that raises.
        def _always_bad(image: bytes, sig: bytes) -> None:
            raise ValueError("signature invalid (test stub)")

        fleet_ota.verify_ota_signature = _always_bad  # type: ignore[assignment]
        api, _ = _make_api()
        body = _build_multipart()
        status, resp_body = await _post_via_api(
            api,
            f"/v1/fleet/{TEST_GROUP_NAME}/ota",
            body,
            bearer=TEST_API_KEY,
        )
        self.assertEqual(status, HTTP_BAD_REQUEST)
        import json

        parsed = json.loads(resp_body)
        self.assertIn("signature", parsed["error"].lower())

    async def test_partial_failure_returns_200_with_mixed_results(self) -> None:
        # Patch the connection factory so one specific host refuses.
        self._patch_open.stop()
        captured: dict[tuple[str, int], _MockWriter] = {}
        self._patch_open = patch.object(
            fleet_ota,
            "_default_open_connection",
            _failing_then_ok_open({"10.0.0.2"}, captured),
        )
        self._patch_open.start()
        api, _ = _make_api()
        body = _build_multipart()
        status, resp_body = await _post_via_api(
            api,
            f"/v1/fleet/{TEST_GROUP_NAME}/ota",
            body,
            bearer=TEST_API_KEY,
        )
        # Partial failure still returns 200 — operator inspects results.
        self.assertEqual(status, HTTP_OK)
        import json

        parsed = json.loads(resp_body)
        statuses = sorted(r["status"] for r in parsed["results"])
        self.assertEqual(statuses, ["error", "ok", "ok"])

    async def test_missing_firmware_field_returns_400(self) -> None:
        api, _ = _make_api()
        body = _build_multipart(include_firmware=False)
        status, _resp = await _post_via_api(
            api,
            f"/v1/fleet/{TEST_GROUP_NAME}/ota",
            body,
            bearer=TEST_API_KEY,
        )
        self.assertEqual(status, HTTP_BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
