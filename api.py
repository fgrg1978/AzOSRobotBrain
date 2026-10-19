# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Minimal HTTP API for the AzOS Robot Brain.

Pure asyncio — no extra dependencies. Implements a subset of HTTP/1.1
sufficient for REST calls from scripts, curl, and dashboards.

Endpoints:
  GET  /health          — 200 OK, {"status": "ok"}
  GET  /status          — full robot state JSON
  GET  /mode            — current mode name
  POST /mode            — {"mode": "patrulla"} — switch mode
  POST /stop            — emergency stop
  POST /task            — {"task": "patrol room A then return"} — queue task
  POST /cmd             — {"skill": "FORWARD", "args": {"speed": 60}} — run one skill
  GET  /topics          — list available data topics
  GET  /topics/{name}   — latest data for a topic
  GET  /config/{key}    — read a config value (dot notation)
  POST /config/{key}    — set a config value (dot notation)
  GET  /fleet/robots    — list all registered robots + aggregated status
  POST /fleet/command   — {"id": "bot_1", "pkt_type": 128, "payload_hex": ".."}
  POST /fleet/broadcast — {"pkt_type": 128, "payload_hex": "..", "type": 0}
  GET  /dashboard[/...] — static dashboard files (B02)

Usage:
    api = APIServer(brain, port=8080)
    asyncio.create_task(api.run())
"""

import asyncio
import hmac
import json
import logging
import os
import time
from typing import Any, Optional, TYPE_CHECKING, Union

import metrics as _metrics_mod
import protocol

if TYPE_CHECKING:
    from server import BrainServer
    from fleet import FleetManager
    from fleet_ota import FleetOtaManager

logger = logging.getLogger("brain.api")


# ── Dashboard static files ────────────────────────────────────────────────────
# B02 Fleet Dashboard: vanilla HTML/JS/CSS served from ./dashboard/.

DASHBOARD_DIR_NAME = "dashboard"
DASHBOARD_ROUTE_PREFIX = "/dashboard"
DASHBOARD_INDEX_FILE = "index.html"

# Content-type mapping for static files served under /dashboard.
_STATIC_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}
_DEFAULT_STATIC_MIME = "application/octet-stream"

# Absolute path to the dashboard directory (resolved once at import time).
DASHBOARD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), DASHBOARD_DIR_NAME))


def _guess_content_type(path: str) -> str:
    """Return a content-type header value for a static file path."""
    _, ext = os.path.splitext(path)
    return _STATIC_MIME_TYPES.get(ext.lower(), _DEFAULT_STATIC_MIME)


def _resolve_dashboard_path(url_path: str) -> Optional[str]:
    """Translate a /dashboard[/...] URL to an absolute file path.

    Returns the absolute path if it is a readable file inside DASHBOARD_ROOT,
    or None if the path is missing / escapes the dashboard root / is a dir.
    Index file is served for the bare /dashboard prefix.
    """
    if not url_path.startswith(DASHBOARD_ROUTE_PREFIX):
        return None

    rel = url_path[len(DASHBOARD_ROUTE_PREFIX) :]
    if rel in ("", "/"):
        rel = "/" + DASHBOARD_INDEX_FILE
    # Strip leading slash before joining so os.path.join doesn't reset root.
    rel = rel.lstrip("/")

    candidate = os.path.abspath(os.path.join(DASHBOARD_ROOT, rel))

    # Guard against path traversal: must stay inside DASHBOARD_ROOT.
    if os.path.commonpath([candidate, DASHBOARD_ROOT]) != DASHBOARD_ROOT:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def _serve_static(writer: asyncio.StreamWriter, file_path: str) -> None:
    """Write a 200 response with the file body and appropriate content-type."""
    with open(file_path, "rb") as f:
        payload = f.read()
    ctype = _guess_content_type(file_path)
    header = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: {ctype}\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Cache-Control: no-cache\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode()
    writer.write(header + payload)


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _response(
    writer: asyncio.StreamWriter,
    status: int,
    body: Union[dict[str, Any], list[Any], str],
    status_text: str = "",
    _count_bytes: bool = True,
) -> None:
    _STATUS_TEXTS = {
        200: "OK",
        201: "Created",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
    }
    if not status_text:
        status_text = _STATUS_TEXTS.get(status, "Unknown")

    if isinstance(body, (dict, list)):
        payload = json.dumps(body).encode()
        ctype = "application/json"
    else:
        payload = body.encode()
        ctype = "text/plain"

    header = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: {ctype}\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode()
    wire = header + payload
    writer.write(wire)
    if _count_bytes:
        _metrics_mod.M.http_bytes_total.labels(direction="out").inc(len(wire))


def _metrics_response(writer: asyncio.StreamWriter) -> None:
    """Write the OpenMetrics text payload for ``GET /metrics``.

    Uses _count_bytes=False so the /metrics response does not count itself
    in phanes_brain_http_bytes_total (avoids a feedback loop on each scrape).
    """
    body_str = _metrics_mod.M.render_text()
    payload = body_str.encode()
    header = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: {_metrics_mod.OPENMETRICS_CONTENT_TYPE}\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode()
    writer.write(header + payload)


def _parse_request(raw: bytes) -> tuple[str, str, dict[str, str], bytes]:
    """Returns (method, path, headers, body)."""
    header_end = raw.find(b"\r\n\r\n")
    if header_end == -1:
        return "", "", {}, b""
    header_raw = raw[:header_end].decode(errors="replace")
    body = raw[header_end + 4 :]

    lines = header_raw.split("\r\n")
    if not lines:
        return "", "", {}, body

    parts = lines[0].split(" ", 2)
    method = parts[0].upper() if len(parts) > 0 else ""
    path = parts[1] if len(parts) > 1 else "/"

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    return method, path, headers, body


# ── DEV04 v1 fleet group OTA — multipart endpoint constants ───────────────────

#: Path prefix and suffix for the v1 group OTA endpoint.
#: Full path shape: ``/v1/fleet/{group}/ota``.
V1_FLEET_GROUP_PREFIX: str = "/v1/fleet/"
V1_FLEET_GROUP_SUFFIX: str = "/ota"

#: Form field names accepted in the multipart body.
V1_OTA_FIELD_FIRMWARE: str = "firmware"
V1_OTA_FIELD_SIGNATURE: str = "signature"
V1_OTA_FIELD_PLATFORM: str = "platform"
V1_OTA_FIELD_FWVER: str = "fw_version"

#: Defaults when the corresponding form fields are missing.
V1_OTA_DEFAULT_PLATFORM: str = "qemu"
V1_OTA_DEFAULT_FW_VERSION: int = 1

#: Cap on how many parts we'll parse out of a multipart body.  Guards
#: against pathological boundary-spam payloads.
V1_OTA_MAX_PARTS: int = 16

#: Marker prefix every multipart part starts with (per RFC 7578).
_MULTIPART_BOUNDARY_PREFIX: bytes = b"--"
_MULTIPART_CRLF: bytes = b"\r\n"
_MULTIPART_HEADER_SEP: bytes = b"\r\n\r\n"


def _parse_multipart_body(body: bytes) -> dict[str, bytes]:
    """Parse a ``multipart/form-data`` body into a ``{field_name: value_bytes}``.

    We auto-detect the boundary from the body's first line so the caller
    doesn't have to thread the Content-Type header through — multipart
    bodies are required to start with ``--<boundary>\\r\\n`` per RFC 7578.

    Returns an empty dict if the body doesn't look like multipart; the
    handler treats that as a 400.  Raises ``ValueError`` on malformed
    input (e.g. boundary line longer than the body, missing terminator).
    """
    if not body:
        raise ValueError("empty multipart body")

    first_line_end = body.find(_MULTIPART_CRLF)
    if first_line_end == -1 or first_line_end < len(_MULTIPART_BOUNDARY_PREFIX):
        raise ValueError("multipart body missing initial boundary line")
    boundary_line = body[:first_line_end]
    if not boundary_line.startswith(_MULTIPART_BOUNDARY_PREFIX):
        raise ValueError("multipart body first line is not a boundary")
    boundary = boundary_line[len(_MULTIPART_BOUNDARY_PREFIX) :]
    if not boundary:
        raise ValueError("multipart boundary is empty")

    delim = _MULTIPART_CRLF + _MULTIPART_BOUNDARY_PREFIX + boundary
    # Strip the leading "--<boundary>" so split() yields the parts cleanly.
    head_strip = _MULTIPART_BOUNDARY_PREFIX + boundary
    if body.startswith(head_strip):
        rest = body[len(head_strip) :]
    else:
        rest = body

    fields: dict[str, bytes] = {}
    pieces = rest.split(delim)
    if len(pieces) > V1_OTA_MAX_PARTS + 1:
        raise ValueError(f"multipart body has too many parts (max {V1_OTA_MAX_PARTS})")
    for piece in pieces:
        # A well-formed body starts each part with CRLF (after the
        # boundary).  Strip it; ignore the final "--\r\n" closing marker.
        if piece in (b"", _MULTIPART_BOUNDARY_PREFIX, _MULTIPART_BOUNDARY_PREFIX + _MULTIPART_CRLF):
            continue
        if piece.startswith(_MULTIPART_BOUNDARY_PREFIX):
            # Final closing boundary marker — done.
            break
        if piece.startswith(_MULTIPART_CRLF):
            piece = piece[len(_MULTIPART_CRLF) :]
        hdr_end = piece.find(_MULTIPART_HEADER_SEP)
        if hdr_end == -1:
            continue
        header_block = piece[:hdr_end].decode("latin-1", errors="replace")
        value = piece[hdr_end + len(_MULTIPART_HEADER_SEP) :]
        # Strip the trailing CRLF that precedes the next boundary.
        if value.endswith(_MULTIPART_CRLF):
            value = value[: -len(_MULTIPART_CRLF)]
        name: Optional[str] = None
        for hline in header_block.split("\r\n"):
            lower = hline.lower()
            if lower.startswith("content-disposition:"):
                # Look for name="...".
                marker = 'name="'
                idx = hline.find(marker)
                if idx != -1:
                    end = hline.find('"', idx + len(marker))
                    if end != -1:
                        name = hline[idx + len(marker) : end]
        if name is not None:
            fields[name] = value
    return fields


# ── Router ────────────────────────────────────────────────────────────────────


class APIServer:
    """Async HTTP API server."""

    # ── Authentication ────────────────────────────────────────────────────
    # Security policy (hardened):
    #
    #   Case 1 — AZOS_BRAIN_API_KEY is set:
    #     Every non-public request must carry "Authorization: Bearer <key>".
    #
    #   Case 2 — AZOS_BRAIN_ALLOW_INSECURE=1 (and no key):
    #     Explicit operator opt-in to open/unauthenticated mode.  The warning
    #     is printed so the intent is visible in logs.
    #
    #   Case 3 — neither variable is set:
    #     run() refuses to bind.  Allowing silent open mode would let anyone
    #     on the LAN POST /stop, /cmd, /fleet/broadcast without any credential.
    #     Operators must make an explicit choice before the server accepts
    #     connections.
    #
    # Why refuse in run() rather than __init__?  __init__ is called during
    # BrainServer construction before the event loop starts; raising there
    # makes the server silently exit without a useful error context.  Raising
    # in run() gives asyncio.gather a chance to print the traceback cleanly.

    #: Env var name that holds the shared API bearer token.
    ENV_API_KEY = "AZOS_BRAIN_API_KEY"
    #: Env var name for the explicit insecure-mode opt-in.
    ENV_ALLOW_INSECURE = "AZOS_BRAIN_ALLOW_INSECURE"
    #: Value of ENV_ALLOW_INSECURE that enables unauthenticated mode.
    ALLOW_INSECURE_VALUE = "1"

    #: Routes that are reachable without auth (health checks, dashboard).
    _PUBLIC_ROUTES = {"/health", "/", ""}

    def __init__(self, brain: "BrainServer", port: int = 8080, api_key: Optional[str] = None):
        self.brain = brain
        self.port = port
        self._start = time.time()
        # Env wins over constructor arg so deployment can override config.
        self.api_key: Optional[str] = os.environ.get(self.ENV_API_KEY) or api_key or None
        self._insecure_mode: bool = (
            os.environ.get(self.ENV_ALLOW_INSECURE) == self.ALLOW_INSECURE_VALUE
        )

    async def run(self) -> None:
        # Refuse to bind if neither a key nor an explicit insecure opt-in is set.
        # This prevents the server from silently opening to the LAN.
        if self.api_key is None and not self._insecure_mode:
            raise RuntimeError(
                "[API] Cannot start: no authentication configured. "
                f"Set {self.ENV_API_KEY} to a secret token, or set "
                f"{self.ENV_ALLOW_INSECURE}={self.ALLOW_INSECURE_VALUE} "
                "to explicitly allow unauthenticated access."
            )
        # B-A10: with no bearer key, ANY request succeeds — binding that to
        # every interface (0.0.0.0) means anyone on the LAN gets unauthenticated
        # /stop, /cmd, /fleet/broadcast, etc. Insecure mode is for local
        # development; keep it reachable only from this machine by default.
        bind_host = "127.0.0.1" if self._insecure_mode and not self.api_key else "0.0.0.0"
        server = await asyncio.start_server(self._handle, bind_host, self.port)
        if self.api_key:
            print(f"[API] Listening on port {self.port} (auth: API key required)")
        else:
            print(
                f"[API] Listening on {bind_host}:{self.port} "
                f"(WARNING: insecure mode — {self.ENV_ALLOW_INSECURE}=1)"
            )
        async with server:
            await server.serve_forever()

    def _is_authorised(self, path: str, headers: dict[str, str]) -> bool:
        """Authorise a request: public routes always pass; everything else
        must carry a matching bearer token if `api_key` is configured.
        In explicit insecure mode (no key, ALLOW_INSECURE=1) all paths pass."""
        if self.api_key is None:
            # Insecure mode was validated in run(); return True here.
            return True
        # Strip query string + trailing slash for the public-route check.
        bare = path.split("?")[0].rstrip("/") or "/"
        if bare in self._PUBLIC_ROUTES or bare.startswith(DASHBOARD_ROUTE_PREFIX):
            return True
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return False
        provided = auth[7:].strip()
        # Constant-time comparison to avoid timing-attack key recovery.
        return hmac.compare_digest(provided, self.api_key)

    # Request body cap. Large enough for an OTA push (~2 MB firmware,
    # base64-expanded to ~2.7 MB + JSON envelope), small enough to block
    # blind upload DoS. Anything legitimately larger should stream through
    # a dedicated chunked endpoint, not the JSON router.
    MAX_REQUEST_BODY_BYTES = 4 * 1024 * 1024
    MAX_REQUEST_HEADERS_BYTES = 8 * 1024

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            peer = writer.get_extra_info("peername")
            peer_str: str = f"{peer[0]}:{peer[1]}" if peer else "unknown"

            # Read headers, then body separately. The previous code did
            # `reader.read(8192)` and parsed both at once — that silently
            # truncated any body over 8 KiB, including every real OTA push
            # (a 2 MB firmware base64-expanded to ~2.7 MB), making the OTA
            # signature verification dead code for production payloads.
            header_blob = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=5,
            )
            if len(header_blob) > self.MAX_REQUEST_HEADERS_BYTES:
                _response(writer, 431, {"error": "headers too large"})
                return
            method, path, headers, leftover = _parse_request(header_blob)
            if not method:
                _response(writer, 400, {"error": "bad request"})
                return
            if not self._is_authorised(path, headers):
                logger.warning(
                    "[API] 401 unauthorised — peer=%s path=%s",
                    peer_str,
                    path,
                )
                _response(writer, 401, {"error": "unauthorised"})
                return
            # Content-Length-driven body read (HTTP/1.0-style; we don't
            # implement chunked transfer-encoding — clients should use
            # plain Content-Length, which is what our own `_response`
            # always emits).
            content_length = 0
            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError:
                _response(writer, 400, {"error": "bad content-length"})
                return
            if content_length < 0 or content_length > self.MAX_REQUEST_BODY_BYTES:
                _response(writer, 413, {"error": "payload too large"})
                return
            body = leftover
            still_needed = content_length - len(body)
            if still_needed > 0:
                body += await asyncio.wait_for(
                    reader.readexactly(still_needed),
                    timeout=30,
                )
            await self._route(method, path, body, writer, peer_str)
        except asyncio.IncompleteReadError:
            _response(writer, 400, {"error": "truncated request"})
        except asyncio.LimitOverrunError:
            _response(writer, 431, {"error": "headers too large"})
        except asyncio.TimeoutError:
            _response(writer, 400, {"error": "request timeout"})
        except Exception as e:
            _response(writer, 500, {"error": str(e)})
        finally:
            await writer.drain()
            writer.close()

    async def _route(
        self,
        method: str,
        path: str,
        body: bytes,
        writer: asyncio.StreamWriter,
        peer_str: str = "unknown",
    ) -> None:
        # Strip query string
        path = path.split("?")[0].rstrip("/") or "/"

        # ── GET routes ─────────────────────────────────────────────────────────
        if method == "GET":
            # Dashboard static files — handled before the normal routes so
            # that /dashboard/app.js etc. don't fall through to 404.
            if path == DASHBOARD_ROUTE_PREFIX or path.startswith(DASHBOARD_ROUTE_PREFIX + "/"):
                file_path = _resolve_dashboard_path(path)
                if file_path is None:
                    _response(writer, 404, {"error": "dashboard file not found"})
                else:
                    _serve_static(writer, file_path)
                return

            if path == "/metrics":
                _metrics_response(writer)

            elif path in ("/health", ""):
                _response(
                    writer,
                    200,
                    {
                        "status": "ok",
                        "uptime_s": round(time.time() - self._start, 1),
                    },
                )
            elif path == "/status":
                _response(writer, 200, self._full_status())
            elif path == "/mode":
                _response(writer, 200, {"mode": self._current_mode()})
            elif path == "/topics":
                _response(writer, 200, self._list_topics())
            elif path.startswith("/topics/"):
                topic_name = path[len("/topics/") :]
                _response(writer, 200, self._topic_data(topic_name))
            elif path.startswith("/config/"):
                key = path[len("/config/") :]
                _response(writer, 200, self._config_get(key))
            elif path == "/fleet/robots":
                _response(writer, 200, self._fleet_status())
            elif path == "/fleet/ota/jobs":
                _response(writer, 200, self._fleet_ota_list())
            elif path.startswith("/fleet/ota/status/"):
                job_id = path[len("/fleet/ota/status/") :]
                result, http_status = self._fleet_ota_status(job_id)
                _response(writer, http_status, result)
            else:
                _response(writer, 404, {"error": "not found"})

        # ── POST routes ────────────────────────────────────────────────────────
        elif method == "POST":
            # DEV04 — /v1/fleet/{group}/ota is multipart/form-data and must be
            # routed BEFORE the JSON parse (which would crash on the binary
            # multipart body).  All other POSTs continue to use JSON below.
            if path.startswith(V1_FLEET_GROUP_PREFIX) and path.endswith(V1_FLEET_GROUP_SUFFIX):
                group = path[len(V1_FLEET_GROUP_PREFIX) : -len(V1_FLEET_GROUP_SUFFIX)]
                result_v1, http_status_v1 = await self._v1_fleet_group_ota(
                    group,
                    body,
                    peer_str,
                )
                _response(writer, http_status_v1, result_v1)
                return

            try:
                payload = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                _response(writer, 400, {"error": "invalid JSON"})
                return

            if path == "/stop":
                await self._do_stop()
                _response(writer, 200, {"status": "stop sent"})

            elif path == "/mode":
                name = payload.get("mode", "")
                if not name:
                    _response(writer, 400, {"error": "missing 'mode' field"})
                    return
                ok = self._set_mode(name)
                if ok:
                    _response(writer, 200, {"mode": name})
                else:
                    modes = self._available_modes()
                    _response(
                        writer,
                        400,
                        {
                            "error": f"unknown mode '{name}'",
                            "available": modes,
                        },
                    )

            elif path == "/task":
                task = payload.get("task", "")
                if not task:
                    _response(writer, 400, {"error": "missing 'task' field"})
                    return
                await self._queue_task(task)
                _response(writer, 201, {"task": task, "status": "queued"})

            elif path == "/cmd":
                skill = payload.get("skill", "")
                args = payload.get("args", {})
                if not skill:
                    _response(writer, 400, {"error": "missing 'skill' field"})
                    return
                result = await self._execute_skill(skill, args)
                _response(writer, 200, result)

            elif path.startswith("/config/"):
                key = path[len("/config/") :]
                _response(writer, 200, self._config_set(key, payload))

            elif path == "/fleet/command":
                result, http_status = await self._fleet_command(payload)
                _response(writer, http_status, result)

            elif path == "/fleet/broadcast":
                result, http_status = await self._fleet_broadcast(payload)
                _response(writer, http_status, result)

            elif path == "/fleet/ota/push":
                result, http_status = await self._fleet_ota_push(payload, peer_str)
                _response(writer, http_status, result)

            elif path.startswith("/fleet/ota/cancel/"):
                job_id = path[len("/fleet/ota/cancel/") :]
                result, http_status = await self._fleet_ota_cancel(job_id)
                _response(writer, http_status, result)

            else:
                _response(writer, 404, {"error": "not found"})

        else:
            _response(writer, 405, {"error": "method not allowed"})

    # ── Brain interactions ─────────────────────────────────────────────────────

    def _full_status(self) -> dict[str, Any]:
        b = self.brain
        s = b.state
        return {
            "connected": s.connected,
            "robot_type": b.robot_type,
            "mode": self._current_mode(),
            "sensors": s.sensors,
            "odom": s.odom,
            "status": s.status,
            "last_sensor_age_s": (
                round(time.time() - s.last_sensor_time, 2) if s.last_sensor_time else None
            ),
            "has_image": len(s.last_image) > 0,
        }

    def _current_mode(self) -> str:
        mm = getattr(self.brain, "mode_manager", None)
        return mm.current_name if mm else "unknown"

    def _available_modes(self) -> list[str]:
        mm = getattr(self.brain, "mode_manager", None)
        return list(mm.modes.keys()) if mm else []

    def _set_mode(self, name: str) -> bool:
        mm = getattr(self.brain, "mode_manager", None)
        if mm:
            return bool(mm.set_mode(name))
        return False

    async def _do_stop(self) -> None:
        # Robot-type-aware central stop (a drone must HOVER, not receive a
        # hardcoded diff-drive n_channels=2 zero frame) — see
        # BrainServer.emergency_stop(). Also replaces the previous
        # hand-rolled protocol.send_packet(writer, 0x80, ...) call, where
        # 0x80 was a magic literal for ACTUATOR_CMD.
        await self.brain.emergency_stop("API /stop")

    async def _queue_task(self, task: str) -> None:
        q = getattr(self.brain, "task_queue", None)
        if q:
            await q.put(task)

    async def _execute_skill(self, skill: str, args: dict[str, Any]) -> dict[str, Any]:
        runner = getattr(self.brain, "runner", None)
        if runner is None:
            return {"error": "skill runner not available"}
        try:
            cmd = await runner.execute_one(skill, args)
            return {
                "skill": skill,
                "args": args,
                "actuator_type": cmd.actuator_type,
                "channels": cmd.channels,
                "flags": cmd.flags,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Topics ────────────────────────────────────────────────────────────────

    DEFAULT_SENSOR_RATE_HZ = 20
    DEFAULT_CAMERA_RATE_HZ = 2
    STATUS_RATE_HZ = 1
    CMD_MOTOR_RATE_HZ = 0  # on-demand, not periodic

    def _list_topics(self) -> list[dict[str, Any]]:
        """Return available data topics with their update rates."""
        robot_cfg = self.brain.config.get("robot", {})
        return [
            {
                "name": "/sensors/imu",
                "rate_hz": robot_cfg.get("sensor_rate_hz", self.DEFAULT_SENSOR_RATE_HZ),
            },
            {
                "name": "/sensors/camera",
                "rate_hz": robot_cfg.get("camera_rate_hz", self.DEFAULT_CAMERA_RATE_HZ),
            },
            {"name": "/cmd/motor", "rate_hz": self.CMD_MOTOR_RATE_HZ},
            {"name": "/status", "rate_hz": self.STATUS_RATE_HZ},
        ]

    ZERO_ACCEL = [0, 0, 0]
    ZERO_GYRO = [0, 0, 0]

    def _topic_data(self, topic_name: str) -> dict[str, Any]:
        """Return latest data for a given topic."""
        state = self.brain.state
        if topic_name == "sensors/imu" and state.sensors:
            return {
                "accel_mg": state.sensors.get("accel_mg", self.ZERO_ACCEL),
                "gyro_mdps": state.sensors.get("gyro_mdps", self.ZERO_GYRO),
                "battery_mv": state.sensors.get("battery_mv", 0),
            }
        elif topic_name == "sensors/camera":
            return {
                "has_image": len(state.last_image) > 0,
                "image_age_s": (
                    round(time.time() - state.last_image_time, 2) if state.last_image_time else None
                ),
            }
        elif topic_name == "cmd/motor":
            return {"info": "write-only topic, POST /cmd to send commands"}
        elif topic_name == "status":
            return self._full_status()
        return {"error": f"unknown topic: {topic_name}"}

    # ── Fleet endpoints ───────────────────────────────────────────────────────

    # HTTP status codes — no magic numbers in route handlers
    HTTP_OK = 200
    HTTP_CREATED = 201
    HTTP_BAD_REQUEST = 400
    HTTP_FORBIDDEN = 403
    HTTP_NOT_FOUND = 404
    HTTP_SERVICE_UNAVAILABLE = 503

    # Field names accepted by fleet POST bodies
    FLEET_FIELD_ID = "id"
    FLEET_FIELD_PKT_TYPE = "pkt_type"
    FLEET_FIELD_PAYLOAD_HEX = "payload_hex"
    FLEET_FIELD_ROBOT_TYPE = "type"

    # B-A10: /fleet/command and /fleet/broadcast send `payload_hex` as raw
    # bytes straight to the robot's writer — bypassing the policy layer
    # (speed clamps, SafetyProfile) that every other command path (/cmd,
    # /task, the reactive LLM loop) goes through. Allowlist only packet
    # types that cannot themselves command motion: MODE (operator mode
    # switch — also used to clear ESTOP) and the safety-tightening ones
    # (ESTOP, DEGRADE, SEMANTIC_LEVEL only ever *reduce* capability, never
    # move the robot). Everything that carries a raw motor/actuator value
    # (ACTUATOR, PREDICT, PAYLOAD), a navigation target (WAYPOINT), a
    # runtime safety threshold (CONFIG), or firmware bytes (OTA_*) is
    # blocked here — those must go through the brain's own policy/OTA
    # flows, not this generic raw-packet passthrough.
    #
    # Owner decision, 2026-09-06: a MODE payload only clears ESTOP / degraded
    # mode on the robot when the byte is `protocol.MODE_ID_ESTOP_RESET`
    # (0xFF) — see its doc comment. An operator posting `payload_hex` here
    # for anything else changes mode and leaves any armed e-stop exactly as
    # armed as they found it; this endpoint does not (and should not) enforce
    # that itself, since MODE is also legitimately used for ordinary mode
    # switches that must NOT rearm anything.
    FLEET_ALLOWED_PKT_TYPES = frozenset({
        protocol.MODE_CMD,
        protocol.ESTOP_CMD,
        protocol.DEGRADE_CMD,
        protocol.SEMANTIC_LEVEL_CMD,
    })

    def _fleet_manager(self) -> Optional["FleetManager"]:
        """Return the brain's FleetManager or None."""
        fm: Optional["FleetManager"] = getattr(self.brain, "fleet_manager", None)
        return fm

    def _fleet_status(self) -> dict[str, Any]:
        fm = self._fleet_manager()
        if fm is None:
            return {"error": "fleet manager not enabled"}
        # Run timeout sweep before serving status so data is fresh.
        fm.check_timeouts()
        status: dict[str, Any] = fm.get_fleet_status()
        return status

    @staticmethod
    def _decode_payload_hex(payload_hex: str) -> bytes:
        """Decode hex payload. Accepts empty string as empty bytes."""
        if not payload_hex:
            return b""
        return bytes.fromhex(payload_hex)

    async def _fleet_command(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        fm = self._fleet_manager()
        if fm is None:
            return {"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE

        robot_id = body.get(self.FLEET_FIELD_ID, "")
        pkt_type = body.get(self.FLEET_FIELD_PKT_TYPE)
        payload_hex = body.get(self.FLEET_FIELD_PAYLOAD_HEX, "")

        if not robot_id:
            return {"error": f"missing '{self.FLEET_FIELD_ID}' field"}, self.HTTP_BAD_REQUEST
        if pkt_type is None:
            return {"error": f"missing '{self.FLEET_FIELD_PKT_TYPE}' field"}, self.HTTP_BAD_REQUEST
        if int(pkt_type) not in self.FLEET_ALLOWED_PKT_TYPES:
            # B-A10: this endpoint bypasses policy/safety clamps entirely.
            return {
                "error": f"pkt_type {int(pkt_type)} not allowed via raw fleet command "
                          f"(bypasses policy/safety clamps) — use /cmd or /task instead",
            }, self.HTTP_FORBIDDEN
        try:
            payload = self._decode_payload_hex(payload_hex)
        except ValueError:
            return {"error": "invalid hex in 'payload_hex'"}, self.HTTP_BAD_REQUEST

        ok = await fm.send_targeted(robot_id, int(pkt_type), payload)
        return {
            "id": robot_id,
            "pkt_type": int(pkt_type),
            "delivered": ok,
        }, (self.HTTP_OK if ok else self.HTTP_NOT_FOUND)

    async def _fleet_broadcast(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        fm = self._fleet_manager()
        if fm is None:
            return {"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE

        pkt_type = body.get(self.FLEET_FIELD_PKT_TYPE)
        payload_hex = body.get(self.FLEET_FIELD_PAYLOAD_HEX, "")
        robot_type = body.get(self.FLEET_FIELD_ROBOT_TYPE)

        if pkt_type is None:
            return {"error": f"missing '{self.FLEET_FIELD_PKT_TYPE}' field"}, self.HTTP_BAD_REQUEST
        if int(pkt_type) not in self.FLEET_ALLOWED_PKT_TYPES:
            # B-A10: this endpoint bypasses policy/safety clamps entirely.
            return {
                "error": f"pkt_type {int(pkt_type)} not allowed via raw fleet broadcast "
                          f"(bypasses policy/safety clamps) — use /cmd or /task instead",
            }, self.HTTP_FORBIDDEN
        try:
            payload = self._decode_payload_hex(payload_hex)
        except ValueError:
            return {"error": "invalid hex in 'payload_hex'"}, self.HTTP_BAD_REQUEST

        results = await fm.broadcast(
            int(pkt_type),
            payload,
            robot_type=int(robot_type) if robot_type is not None else None,
        )
        return {
            "pkt_type": int(pkt_type),
            "delivered": sum(1 for ok in results.values() if ok),
            "attempted": len(results),
            "results": results,
        }, self.HTTP_OK

    # ── Fleet OTA (DEV04) ──────────────────────────────────────────────────────

    def _fleet_ota_manager(self) -> Optional["FleetOtaManager"]:
        """Lazy-cached FleetOtaManager bound to the brain's FleetManager.

        Built on first use so the brain doesn't import fleet_ota at startup
        if the operator never touches the OTA endpoints — keeps tests of
        api.py decoupled from the OTA module."""
        cached: Optional["FleetOtaManager"] = getattr(self, "_fota_cache", None)
        if cached is not None:
            return cached
        fm = self._fleet_manager()
        if fm is None:
            return None
        from fleet_ota import FleetOtaManager

        self._fota_cache: Optional["FleetOtaManager"] = FleetOtaManager(fm)
        return self._fota_cache

    async def _fleet_ota_push(
        self, body: dict[str, Any], peer_str: str = "unknown"
    ) -> tuple[dict[str, Any], int]:
        """Kick off a fleet OTA deployment job.

        Body: {
          "image_b64": str,        # base64 of the kernel image bytes
          "sig_b64":   str,        # base64 of the .SIG sidecar
          "platform":  str,        # qemu / vf2 / k1 / esp32c3
          "fw_version": int,
          "robot_ids": list[str],  # optional — defaults to all online
        }
        """
        import base64

        fota = self._fleet_ota_manager()
        if fota is None:
            return {"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE
        try:
            image = base64.b64decode(body.get("image_b64", ""))
            sig = base64.b64decode(body.get("sig_b64", ""))
            platform = body.get("platform", "")
            fw_version = int(body.get("fw_version", 0))
            robot_ids = body.get("robot_ids")  # list[str] | None
        except (ValueError, TypeError) as e:
            return {"error": f"bad body: {e}"}, self.HTTP_BAD_REQUEST
        if not image:
            return {"error": "missing 'image_b64'"}, self.HTTP_BAD_REQUEST
        if not platform:
            return {"error": "missing 'platform'"}, self.HTTP_BAD_REQUEST
        try:
            job_id = await fota.start_job(
                image=image,
                sig=sig,
                platform=platform,
                fw_version=fw_version,
                robot_ids=robot_ids,
            )
        except ValueError as e:
            logger.warning(
                "[API] OTA push rejected — peer=%s platform=%s reason=%s",
                peer_str,
                platform,
                e,
            )
            return {"error": str(e)}, self.HTTP_BAD_REQUEST
        return {"job_id": job_id, "status_url": f"/fleet/ota/status/{job_id}"}, self.HTTP_CREATED

    def _fleet_ota_status(self, job_id: str) -> tuple[dict[str, Any], int]:
        fota = self._fleet_ota_manager()
        if fota is None:
            return {"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE
        status = fota.get_job_status(job_id)
        if status is None:
            return {"error": f"job '{job_id}' not found"}, self.HTTP_NOT_FOUND
        return status, self.HTTP_OK

    def _fleet_ota_list(self) -> dict[str, Any]:
        fota = self._fleet_ota_manager()
        if fota is None:
            return {"error": "fleet manager not enabled"}
        return {"jobs": fota.list_jobs()}

    async def _fleet_ota_cancel(self, job_id: str) -> tuple[dict[str, Any], int]:
        fota = self._fleet_ota_manager()
        if fota is None:
            return {"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE
        ok = await fota.cancel_job(job_id)
        if not ok:
            return {"error": f"job '{job_id}' not cancellable"}, self.HTTP_NOT_FOUND
        return {"job_id": job_id, "cancelled": True}, self.HTTP_OK

    # ── Config read/write ─────────────────────────────────────────────────────

    CONFIG_SEPARATOR = "."

    def _config_get(self, key: str) -> dict[str, Any]:
        """Get a config value by dot-notation key (e.g. 'robot.type')."""
        parts = key.split(self.CONFIG_SEPARATOR)
        node = self.brain.config
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return {"error": f"key not found: {key}"}
        return {"key": key, "value": node}

    def _config_set(self, key: str, body: dict[str, Any]) -> dict[str, Any]:
        """Set a config value by dot-notation key. Body must have 'value'."""
        if "value" not in body:
            return {"error": "missing 'value' field in body"}
        parts = key.split(self.CONFIG_SEPARATOR)
        node = self.brain.config
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return {"error": f"key not found: {key}"}
        last = parts[-1]
        if not isinstance(node, dict):
            return {"error": f"cannot set on non-dict: {key}"}
        old_value = node.get(last)
        node[last] = body["value"]
        return {"key": key, "old_value": old_value, "value": body["value"]}

    # ── DEV04: POST /v1/fleet/{group}/ota ─────────────────────────────────────

    async def _v1_fleet_group_ota(
        self,
        group: str,
        body: bytes,
        peer_str: str,
    ) -> tuple[dict[str, Any], int]:
        """Handle ``POST /v1/fleet/{group}/ota`` — synchronous group OTA push.

        Multipart fields (RFC 7578 ``multipart/form-data``):
            firmware:   raw ``.bin`` bytes (required)
            signature:  raw Ed25519 ``.sig`` bytes (required)
            platform:   "qemu" | "vf2" | "k1" | "esp32c3" (optional)
            fw_version: integer (optional)

        Returns ``{"results": [{"robot_id", "status", "bytes_sent", "error"}, ...]}``
        on 200.  Auth is checked upstream in ``_handle`` via the standard
        ``Authorization: Bearer`` header — by the time we get here the
        caller is already authorised.
        """
        from fleet_ota import (
            DISPATCH_KEY_STATUS,
            DISPATCH_STATUS_ERROR,
            _dispatch_ota_to_group,
            verify_ota_signature,
        )

        if not group:
            logger.warning(
                "[API v1 OTA] empty group in path — peer=%s",
                peer_str,
            )
            return {"error": "missing group name"}, self.HTTP_BAD_REQUEST

        fm = self._fleet_manager()
        if fm is None:
            return ({"error": "fleet manager not enabled"}, self.HTTP_SERVICE_UNAVAILABLE)

        try:
            fields = _parse_multipart_body(body)
        except ValueError as e:
            logger.warning(
                "[API v1 OTA] multipart parse failed — peer=%s group=%s reason=%s",
                peer_str,
                group,
                e,
            )
            return {"error": f"bad multipart body: {e}"}, self.HTTP_BAD_REQUEST

        firmware = fields.get(V1_OTA_FIELD_FIRMWARE)
        signature = fields.get(V1_OTA_FIELD_SIGNATURE)
        if not firmware:
            logger.warning(
                "[API v1 OTA] missing firmware field — peer=%s group=%s",
                peer_str,
                group,
            )
            return ({"error": f"missing '{V1_OTA_FIELD_FIRMWARE}' part"}, self.HTTP_BAD_REQUEST)
        if not signature:
            logger.warning(
                "[API v1 OTA] missing signature field — peer=%s group=%s",
                peer_str,
                group,
            )
            return ({"error": f"missing '{V1_OTA_FIELD_SIGNATURE}' part"}, self.HTTP_BAD_REQUEST)

        # Parse optional platform / fw_version (defaults below).
        platform_raw = fields.get(V1_OTA_FIELD_PLATFORM)
        platform: str = (
            platform_raw.decode("latin-1", errors="replace")
            if platform_raw
            else V1_OTA_DEFAULT_PLATFORM
        ).strip()
        fwver_raw = fields.get(V1_OTA_FIELD_FWVER)
        fw_version: int = V1_OTA_DEFAULT_FW_VERSION
        if fwver_raw:
            try:
                fw_version = int(fwver_raw.decode("latin-1", errors="replace").strip())
            except ValueError:
                logger.warning(
                    "[API v1 OTA] bad fw_version — peer=%s group=%s raw=%r",
                    peer_str,
                    group,
                    fwver_raw,
                )
                return (
                    {"error": f"invalid '{V1_OTA_FIELD_FWVER}': not an integer"},
                    self.HTTP_BAD_REQUEST,
                )

        # Verify Ed25519 signature on the brain side BEFORE any robot
        # receives bytes.  verify_ota_signature raises ValueError on
        # any failure (missing key file, bad key length, invalid sig).
        try:
            verify_ota_signature(firmware, signature)
        except ValueError as e:
            logger.warning(
                "[API v1 OTA] signature rejected — peer=%s group=%s reason=%s",
                peer_str,
                group,
                e,
            )
            return ({"error": f"signature verification failed: {e}"}, self.HTTP_BAD_REQUEST)

        try:
            results = await _dispatch_ota_to_group(
                fleet=fm,
                group=group,
                fw_bytes=firmware,
                sig_bytes=signature,
                platform=platform,
                fw_version=fw_version,
            )
        except ValueError as e:
            # Empty / unknown group.
            logger.warning(
                "[API v1 OTA] unknown group — peer=%s group=%s reason=%s",
                peer_str,
                group,
                e,
            )
            return {"error": str(e)}, self.HTTP_NOT_FOUND

        errors = sum(1 for r in results if r[DISPATCH_KEY_STATUS] == DISPATCH_STATUS_ERROR)
        logger.info(
            "[API v1 OTA] group=%s pushed to %d robots (%d errors)",
            group,
            len(results),
            errors,
        )
        return {"results": results}, self.HTTP_OK
