# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Control-plane HTTP server.

Exposes three endpoints consumed by operators and data-plane workers:

  POST /v1/robots/{id}/route
    Returns which data-plane worker owns this robot_id.
    Body: {} (robot_id taken from URL path)
    Response: {"robot_id": str, "data_plane": str | null}
              data_plane is null when the ring is empty (single-process mode
              with no registered workers).

  GET /v1/fleet
    Fleet-wide read: returns a JSON list of robot summaries from the backend.
    In single-process mode delegates to the injected ``fleet_status_fn``.

  POST /v1/ota
    Accepts a fleet OTA push request and fans it out to the relevant data
    planes via ``OtaCoordinator``.
    Body JSON: {
        "robot_ids": [...],     // optional; [] or absent = all online robots
        "platform":  str,       // e.g. "qemu"
        "fw_version": int,
    }
    Response: {"job_id": str, "state": "RUNNING"}

  GET /health
    Liveness check — always 200 OK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Coroutine, Optional

from control_plane.auth import BearerAuth
from control_plane.discovery import ShardCoordinator
from control_plane.ota_coordinator import OtaCoordinator

logger = logging.getLogger("brain.control_plane")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Default TCP port for the control-plane HTTP API.
CP_DEFAULT_PORT: int = 8090

#: Maximum bytes read from a single HTTP request body (Content-Length cap).
CP_MAX_REQUEST_BYTES: int = 1 * 1024 * 1024  # 1 MiB

#: Maximum bytes accepted for the request-line + headers block.
CP_MAX_HEADER_BYTES: int = 8 * 1024

#: Timeout (seconds) waiting for the request headers to arrive.
CP_HEADER_TIMEOUT_S: float = 10.0

#: Timeout (seconds) waiting for the remainder of the body once
#: Content-Length is known.
CP_BODY_TIMEOUT_S: float = 30.0

#: Regex that matches /v1/robots/{id}/route — id may contain alphanum + _-.
_ROUTE_PATH_RE: re.Pattern[str] = re.compile(r"^/v1/robots/(?P<robot_id>[A-Za-z0-9_.\-]+)/route$")

#: Path for the fleet read endpoint.
_FLEET_PATH: str = "/v1/fleet"

#: Path for the OTA dispatch endpoint.
_OTA_PATH: str = "/v1/ota"

#: Path for the liveness probe.
_HEALTH_PATH: str = "/health"

#: Bind address used when a bearer token is configured (workers are remote).
CP_BIND_ANY: str = "0.0.0.0"

#: Bind address used in explicit insecure mode.  Unlike api.py, this server
#: used to bind 0.0.0.0 unconditionally — including with auth disabled, which
#: published POST /v1/ota (fleet-wide firmware push) and GET /v1/fleet to the
#: whole LAN with no token.  Insecure mode is for local development; keep it
#: reachable only from this machine, exactly as APIServer.run does.
CP_BIND_LOOPBACK: str = "127.0.0.1"


# Type alias for the optional fleet-status provider.
FleetStatusFn = Callable[[], Any]


class ControlPlaneServer:
    """Async HTTP server exposing the control-plane API.

    Args:
        coordinator:     Consistent-hash ring for robot → data-plane routing.
        ota_coordinator: OTA fan-out coordinator.
        auth:            Bearer-token authenticator.
        fleet_status_fn: Callable returning fleet status JSON (dict or list).
                         Used for the ``GET /v1/fleet`` endpoint.  Supply the
                         ``FleetManager.get_fleet_status`` method in
                         single-process mode.
        port:            TCP port to bind.
    """

    def __init__(
        self,
        *,
        coordinator: ShardCoordinator,
        ota_coordinator: OtaCoordinator,
        auth: BearerAuth,
        fleet_status_fn: Optional[FleetStatusFn] = None,
        port: int = CP_DEFAULT_PORT,
    ) -> None:
        self._coord = coordinator
        self._ota = ota_coordinator
        self._auth = auth
        self._fleet_status_fn = fleet_status_fn
        self._port = port

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        insecure = self._auth.is_insecure()
        bind_host = CP_BIND_LOOPBACK if insecure else CP_BIND_ANY
        server = await asyncio.start_server(self._handle, bind_host, self._port)
        if insecure:
            logger.warning(
                "[ControlPlane] Listening on %s:%d (insecure mode — no token "
                "required; bind restricted to loopback)",
                bind_host,
                self._port,
            )
        else:
            logger.info("[ControlPlane] Listening on %s:%d", bind_host, self._port)
        async with server:
            await server.serve_forever()

    # ------------------------------------------------------------------
    # Internal request handling
    # ------------------------------------------------------------------

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # A single `reader.read(N)` silently truncates any body that
            # doesn't arrive in the first TCP read — for a POST /v1/ota with
            # a real robot_ids list that means the JSON parse below either
            # fails or (worse) succeeds on a truncated payload. Mirror
            # api.py's `_handle`: read the header block up to the blank
            # line, then read exactly Content-Length more for the body.
            header_blob = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=CP_HEADER_TIMEOUT_S,
            )
            if len(header_blob) > CP_MAX_HEADER_BYTES:
                _respond(writer, 431, {"error": "headers too large"})
                return
            method, path, headers, leftover = _parse_request(header_blob)
            if not self._auth.is_authorised(path, headers):
                _respond(writer, 401, {"error": "Unauthorized"})
                return
            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError:
                _respond(writer, 400, {"error": "bad content-length"})
                return
            if content_length < 0 or content_length > CP_MAX_REQUEST_BYTES:
                _respond(writer, 413, {"error": "payload too large"})
                return
            body = leftover
            still_needed = content_length - len(body)
            if still_needed > 0:
                body += await asyncio.wait_for(
                    reader.readexactly(still_needed),
                    timeout=CP_BODY_TIMEOUT_S,
                )
            await self._route(writer, method, path, body)
        except asyncio.IncompleteReadError:
            _respond(writer, 400, {"error": "truncated request"})
        except asyncio.LimitOverrunError:
            _respond(writer, 431, {"error": "headers too large"})
        except asyncio.TimeoutError:
            _respond(writer, 400, {"error": "request timeout"})
        except Exception as exc:
            logger.error("[ControlPlane] handler error: %s", exc)
            try:
                _respond(writer, 500, {"error": "internal error"})
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _route(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        path: str,
        body: bytes,
    ) -> None:
        bare = path.split("?")[0].rstrip("/") or "/"

        if bare == _HEALTH_PATH:
            _respond(writer, 200, {"status": "ok"})
            return

        # POST /v1/robots/{id}/route
        m = _ROUTE_PATH_RE.match(bare)
        if m and method == "POST":
            robot_id = m.group("robot_id")
            dp = self._coord.assign(robot_id)
            _respond(writer, 200, {"robot_id": robot_id, "data_plane": dp})
            return

        # GET /v1/fleet
        if bare == _FLEET_PATH and method == "GET":
            if self._fleet_status_fn is not None:
                result = self._fleet_status_fn()
                _respond(writer, 200, result)
            else:
                # TODO: query shared backend (Redis) for fleet state in
                #       multi-process mode.  For now, return empty.
                _respond(writer, 200, {"robots": []})
            return

        # POST /v1/ota
        if bare == _OTA_PATH and method == "POST":
            await self._handle_ota(writer, body)
            return

        _respond(writer, 404, {"error": "not found"})

    async def _handle_ota(self, writer: asyncio.StreamWriter, body: bytes) -> None:
        try:
            payload: dict[str, Any] = json.loads(body) if body else {}
        except json.JSONDecodeError:
            _respond(writer, 400, {"error": "invalid JSON"})
            return

        robot_ids: list[str] = payload.get("robot_ids") or []
        platform: str = payload.get("platform") or "qemu"
        fw_version: int = int(payload.get("fw_version") or 1)

        # Build per-data-plane robot map.
        dp_robot_map: dict[str, list[str]] = {}
        for rid in robot_ids:
            dp = self._coord.assign(rid)
            if dp is not None:
                dp_robot_map.setdefault(dp, []).append(rid)

        job_id = await self._ota.start_job(
            robot_ids=robot_ids,
            platform=platform,
            fw_version=fw_version,
            dp_addresses=self._coord.nodes(),
            dp_robot_map=dp_robot_map,
        )
        _respond(writer, 200, {"job_id": job_id, "state": "RUNNING"})


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib-only, mirrors api.py style)
# ---------------------------------------------------------------------------


def _parse_request(raw: bytes) -> tuple[str, str, dict[str, str], bytes]:
    """Return (method, path, headers, body) from raw HTTP bytes."""
    sep = b"\r\n\r\n"
    sep_idx = raw.find(sep)
    if sep_idx == -1:
        return "", "", {}, b""
    header_raw = raw[:sep_idx].decode(errors="replace")
    body = raw[sep_idx + 4 :]
    lines = header_raw.split("\r\n")
    parts = lines[0].split(" ", 2) if lines else []
    method = parts[0].upper() if len(parts) > 0 else ""
    path = parts[1] if len(parts) > 1 else "/"
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return method, path, headers, body


_STATUS_TEXTS: dict[int, str] = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    500: "Internal Server Error",
}


def _respond(
    writer: asyncio.StreamWriter,
    status: int,
    body: dict[str, Any] | list[Any],
) -> None:
    """Write a JSON HTTP response."""
    payload = json.dumps(body).encode()
    status_text = _STATUS_TEXTS.get(status, "Unknown")
    header = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode()
    writer.write(header + payload)


# ---------------------------------------------------------------------------
# Standalone entry-point (``python -m control_plane.main``)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> "argparse.ArgumentParser":  # type: ignore[name-defined]
    import argparse

    p = argparse.ArgumentParser(description="AzOSRobotBrain — control plane")
    p.add_argument(
        "--port", type=int, default=CP_DEFAULT_PORT, help="TCP port for the control-plane HTTP API"
    )
    p.add_argument("--data-planes", nargs="*", default=[], help="data-plane addresses (host:port)")
    return p


def main() -> None:
    import argparse

    args = _build_arg_parser().parse_args()

    auth = BearerAuth()
    coord = ShardCoordinator(nodes=args.data_planes or [])
    ota = OtaCoordinator()
    server = ControlPlaneServer(
        coordinator=coord,
        ota_coordinator=ota,
        auth=auth,
        port=args.port,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
