# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for B02 Fleet Dashboard (static files + /dashboard route).

The dashboard is a plain HTML/JS/CSS app served by api.py. We can't run
a real browser here, so the sanity checks are:

  1. Dashboard files exist, are non-empty, and decode as UTF-8.
  2. The /dashboard route returns 200 for index.html and nested assets.
  3. Path traversal is blocked (../ in URL must not escape DASHBOARD_ROOT).
  4. Content types are correct for .html / .js / .css.
"""

import asyncio
import os
import sys
from typing import Optional

import pytest

# Make the project root importable when running pytest from a subdir.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import api  # noqa: E402
from api import (  # noqa: E402
    APIServer,
    DASHBOARD_DIR_NAME,
    DASHBOARD_INDEX_FILE,
    DASHBOARD_ROOT,
    DASHBOARD_ROUTE_PREFIX,
    _guess_content_type,
    _resolve_dashboard_path,
)

# ---------------------------------------------------------------------------
# Constants — keep the tests free of magic numbers.
# ---------------------------------------------------------------------------

HTTP_STATUS_OK = "200"
HTTP_STATUS_NOT_FOUND = "404"

EXPECTED_DASHBOARD_FILES = (
    DASHBOARD_INDEX_FILE,
    "app.js",
    "style.css",
)


# ---------------------------------------------------------------------------
# Stub stream writer — collects bytes, mimics asyncio.StreamWriter API.
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.closed = False

    # asyncio.StreamWriter subset used by APIServer._response / _serve_static
    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def get_extra_info(self, _name: str) -> None:  # pragma: no cover
        return None


class _FakeBrain:
    """Minimal brain stand-in for APIServer wiring."""

    def __init__(self) -> None:
        self.state = type(
            "S",
            (),
            {
                "connected": False,
                "sensors": {},
                "odom": {},
                "last_image": b"",
                "last_sensor_time": 0,
                "last_image_time": 0,
                "status": {},
            },
        )()
        self.robot_type = 0
        self.config = {}
        self.fleet_manager = None


def _status_line(buf: bytes) -> str:
    """Return the HTTP status code from a raw response buffer."""
    # b"HTTP/1.1 200 OK\r\n..." -> "200"
    first_line = buf.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
    parts = first_line.split(" ")
    return parts[1] if len(parts) >= 2 else ""


def _content_type(buf: bytes) -> Optional[str]:
    for line in buf.split(b"\r\n"):
        if line.lower().startswith(b"content-type:"):
            return line.split(b":", 1)[1].strip().decode("ascii", errors="replace")
    return None


def _body(buf: bytes) -> bytes:
    sep = b"\r\n\r\n"
    idx = buf.find(sep)
    return buf[idx + len(sep) :] if idx >= 0 else b""


async def _get(server: APIServer, path: str) -> bytes:
    """Invoke the router for a GET request and return the raw response."""
    writer = _FakeWriter()
    await server._route("GET", path, b"", writer)  # noqa: SLF001
    return bytes(writer.buf)


# ---------------------------------------------------------------------------
# File-level sanity checks
# ---------------------------------------------------------------------------


class TestDashboardFiles:
    def test_dashboard_root_exists(self) -> None:
        assert os.path.isdir(DASHBOARD_ROOT), f"dashboard directory missing: {DASHBOARD_ROOT}"

    @pytest.mark.parametrize("filename", EXPECTED_DASHBOARD_FILES)
    def test_expected_files_present_and_nonempty(self, filename: str) -> None:
        path = os.path.join(DASHBOARD_ROOT, filename)
        assert os.path.isfile(path), f"missing: {path}"
        assert os.path.getsize(path) > 0, f"empty: {path}"

    def test_index_is_valid_utf8(self) -> None:
        path = os.path.join(DASHBOARD_ROOT, DASHBOARD_INDEX_FILE)
        with open(path, "rb") as f:
            data = f.read()
        # Must decode as UTF-8 and look like HTML.
        text = data.decode("utf-8")
        lowered = text.lower()
        assert "<!doctype html>" in lowered
        assert "</html>" in lowered
        # Dashboard references its JS entrypoint.
        assert "/dashboard/app.js" in text

    def test_app_js_is_valid_utf8(self) -> None:
        path = os.path.join(DASHBOARD_ROOT, "app.js")
        with open(path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8")
        # Must be vanilla JS (no bundler markers, no imports of deps).
        assert "require(" not in text
        assert 'from "react' not in text
        assert "pollFleet" in text

    def test_style_css_is_valid_utf8(self) -> None:
        path = os.path.join(DASHBOARD_ROOT, "style.css")
        with open(path, "rb") as f:
            data = f.read()
        data.decode("utf-8")  # raises if not UTF-8


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_bare_prefix_serves_index(self) -> None:
        resolved = _resolve_dashboard_path(DASHBOARD_ROUTE_PREFIX)
        assert resolved is not None
        assert resolved.endswith(DASHBOARD_INDEX_FILE)

    def test_trailing_slash_serves_index(self) -> None:
        resolved = _resolve_dashboard_path(DASHBOARD_ROUTE_PREFIX + "/")
        assert resolved is not None
        assert resolved.endswith(DASHBOARD_INDEX_FILE)

    def test_named_asset(self) -> None:
        resolved = _resolve_dashboard_path(DASHBOARD_ROUTE_PREFIX + "/app.js")
        assert resolved is not None
        assert resolved.endswith("app.js")

    def test_missing_file_returns_none(self) -> None:
        resolved = _resolve_dashboard_path(DASHBOARD_ROUTE_PREFIX + "/does-not-exist.xyz")
        assert resolved is None

    def test_path_traversal_blocked(self) -> None:
        # /dashboard/../api.py must not resolve outside the dashboard dir.
        resolved = _resolve_dashboard_path(DASHBOARD_ROUTE_PREFIX + "/../api.py")
        assert resolved is None

    def test_unrelated_path_returns_none(self) -> None:
        assert _resolve_dashboard_path("/other") is None


class TestContentTypes:
    def test_html(self) -> None:
        assert "text/html" in _guess_content_type("index.html")

    def test_js(self) -> None:
        assert "javascript" in _guess_content_type("app.js")

    def test_css(self) -> None:
        assert "text/css" in _guess_content_type("style.css")

    def test_unknown_extension(self) -> None:
        assert _guess_content_type("weird.xyz") == api._DEFAULT_STATIC_MIME


# ---------------------------------------------------------------------------
# Integration: exercise the real router with a stub brain
# ---------------------------------------------------------------------------


class TestDashboardRoute:
    def _server(self) -> APIServer:
        return APIServer(_FakeBrain())

    def test_get_dashboard_returns_200(self) -> None:
        server = self._server()
        buf = asyncio.run(_get(server, DASHBOARD_ROUTE_PREFIX))
        assert _status_line(buf) == HTTP_STATUS_OK
        ctype = _content_type(buf)
        assert ctype is not None and "text/html" in ctype
        assert b"<!DOCTYPE html>" in _body(buf)

    def test_get_app_js_returns_200(self) -> None:
        server = self._server()
        buf = asyncio.run(_get(server, DASHBOARD_ROUTE_PREFIX + "/app.js"))
        assert _status_line(buf) == HTTP_STATUS_OK
        ctype = _content_type(buf)
        assert ctype is not None and "javascript" in ctype

    def test_get_style_css_returns_200(self) -> None:
        server = self._server()
        buf = asyncio.run(_get(server, DASHBOARD_ROUTE_PREFIX + "/style.css"))
        assert _status_line(buf) == HTTP_STATUS_OK
        ctype = _content_type(buf)
        assert ctype is not None and "text/css" in ctype

    def test_missing_dashboard_asset_returns_404(self) -> None:
        server = self._server()
        buf = asyncio.run(_get(server, DASHBOARD_ROUTE_PREFIX + "/missing.txt"))
        assert _status_line(buf) == HTTP_STATUS_NOT_FOUND

    def test_path_traversal_returns_404(self) -> None:
        server = self._server()
        buf = asyncio.run(_get(server, DASHBOARD_ROUTE_PREFIX + "/../api.py"))
        assert _status_line(buf) == HTTP_STATUS_NOT_FOUND

    def test_dashboard_prefix_confirms_directory_name(self) -> None:
        # Sanity: prefix stays aligned with the physical directory name.
        assert DASHBOARD_ROUTE_PREFIX.endswith(DASHBOARD_DIR_NAME)
