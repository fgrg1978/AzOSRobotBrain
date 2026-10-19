# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Data-plane ownership checker and pipeline gate.

Responsibilities:
  1. Given a ``robot_id`` extracted from the first STATUS or SENSOR packet,
     determine whether this data-plane node is the owner.
  2. In single-process mode: ask the injected ``ShardCoordinator`` directly.
  3. In multi-process mode: make a lightweight HTTP POST to the control-plane
     ``/v1/robots/{id}/route`` endpoint and compare the returned
     ``data_plane`` field against our own ``self_node`` address.
  4. If not owner → reject the connection (return False from ``check_ownership``).
  5. If owner → the caller (DataPlaneServer) can proceed with the full
     perception / planner / policy pipeline.

The actual pipeline invocation is intentionally NOT done inside this module.
``DataPlaneServer.main`` imports perception/planner/policy lazily so that
the module-level import of ``data_plane.worker`` does not trigger LM Studio
connections in tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from control_plane.auth import ENV_CP_API_KEY
from control_plane.discovery import ShardCoordinator
from data_plane import CP_INPROCESS_SENTINEL

logger = logging.getLogger("brain.data_plane.worker")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Timeout in seconds for the ownership query to the control plane.
OWNERSHIP_QUERY_TIMEOUT_S: float = 3.0

#: Maximum bytes read from the control-plane response.
OWNERSHIP_RESPONSE_MAX_BYTES: int = 4096

#: HTTP path pattern for the ownership query (formatted with robot_id).
_CP_ROUTE_PATH_TEMPLATE: str = "/v1/robots/{robot_id}/route"

#: HTTP method used for the ownership query.
_CP_ROUTE_METHOD: str = "POST"

#: HTTP status returned by the control plane when the bearer token is missing
#: or wrong.  Called out by name because a 401 and a genuine "not our shard"
#: answer used to be indistinguishable here: the 401 body carries no
#: ``data_plane`` key, so ``assigned`` came back None, ``owned`` came back
#: False, and every robot was silently refused with a debug line blaming the
#: hash ring.  Turning auth on therefore looked like the ring was broken.
_CP_HTTP_UNAUTHORIZED: int = 401


class OwnershipChecker:
    """Determines if this data-plane node owns a given robot_id.

    Args:
        self_node:      Address of THIS data-plane (e.g. "localhost:9100").
                        Use ``CP_INPROCESS_SENTINEL`` for single-process mode.
        coordinator:    Local ``ShardCoordinator`` (single-process mode only).
        cp_address:     Control-plane host:port (multi-process mode only).
        cp_api_key:     Bearer token for the control plane.  Defaults to
                        ``AZOS_BRAIN_CP_API_KEY`` from the environment so a
                        worker inherits the same secret the control plane was
                        started with.  ``/v1/robots/{id}/route`` is not a
                        public path, so without this the ownership query gets
                        a 401 and every robot is refused — which is why the
                        stack only appeared to work with auth disabled.
    """

    def __init__(
        self,
        self_node: str,
        *,
        coordinator: Optional[ShardCoordinator] = None,
        cp_address: Optional[str] = None,
        cp_api_key: Optional[str] = None,
    ) -> None:
        self._self_node = self_node
        self._coordinator = coordinator
        self._cp_address = cp_address
        # Env wins over the constructor arg, matching APIServer/BearerAuth.
        self._cp_api_key: Optional[str] = os.environ.get(ENV_CP_API_KEY) or cp_api_key or None
        self._inprocess = self_node == CP_INPROCESS_SENTINEL

    def is_inprocess(self) -> bool:
        return self._inprocess

    async def check_ownership(self, robot_id: str) -> bool:
        """Return True if this data-plane owns *robot_id*.

        Uses the local coordinator (O(log N), no I/O) when one is injected,
        regardless of whether running in single-process or multi-process mode.
        Falls back to HTTP when no local coordinator is available.
        """
        if self._coordinator is not None:
            return self._check_inprocess(robot_id)
        if self._inprocess:
            # In-process mode with no coordinator → accept all.
            return True
        return await self._check_http(robot_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_inprocess(self, robot_id: str) -> bool:
        if self._coordinator is None:
            # No ring configured → accept all (degenerate single-node case).
            return True
        owner = self._coordinator.assign(robot_id)
        if owner is None:
            # Empty ring → accept (only one data plane running).
            return True
        owned = owner == self._self_node
        if not owned:
            logger.debug(
                "[DP/worker] %s: ring says owner=%s, we are %s — refusing",
                robot_id,
                owner,
                self._self_node,
            )
        return owned

    async def _check_http(self, robot_id: str) -> bool:
        if self._cp_address is None:
            logger.warning(
                "[DP/worker] no cp_address configured — accepting %s by default",
                robot_id,
            )
            return True
        path = _CP_ROUTE_PATH_TEMPLATE.format(robot_id=robot_id)
        try:
            host, port_str = self._cp_address.rsplit(":", 1)
            port = int(port_str)
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=OWNERSHIP_QUERY_TIMEOUT_S,
            )
            auth_header = (
                f"Authorization: Bearer {self._cp_api_key}\r\n" if self._cp_api_key else ""
            )
            request = (
                f"{_CP_ROUTE_METHOD} {path} HTTP/1.1\r\n"
                f"Host: {self._cp_address}\r\n"
                f"{auth_header}"
                f"Content-Length: 0\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode()
            writer.write(request)
            await writer.drain()
            resp = await asyncio.wait_for(
                reader.read(OWNERSHIP_RESPONSE_MAX_BYTES),
                timeout=OWNERSHIP_QUERY_TIMEOUT_S,
            )
            writer.close()
            body_start = resp.find(b"\r\n\r\n")
            if body_start == -1:
                logger.error("[DP/worker] malformed CP response for %s", robot_id)
                return False
            # Report a 401 as an auth failure rather than letting it fall
            # through as "the ring says someone else owns this robot" — those
            # two produced identical output before, and the misdiagnosis is
            # what made operators turn control-plane auth back off.
            if _http_status(resp[:body_start]) == _CP_HTTP_UNAUTHORIZED:
                logger.error(
                    "[DP/worker] control plane rejected the ownership query for "
                    "%s: 401 Unauthorized. Set %s on this worker to the same "
                    "token the control plane runs with. Refusing the robot.",
                    robot_id,
                    ENV_CP_API_KEY,
                )
                return False
            body = resp[body_start + 4 :]
            data: dict[str, object] = json.loads(body)
            assigned = data.get("data_plane")
            owned = assigned == self._self_node
            if not owned:
                logger.debug(
                    "[DP/worker] %s: CP says owner=%s, we are %s — refusing",
                    robot_id,
                    assigned,
                    self._self_node,
                )
            return owned
        except Exception as exc:
            logger.error("[DP/worker] ownership query failed for %s: %s", robot_id, exc)
            # Fail-closed: if we cannot reach the control plane, refuse the
            # connection rather than silently duplicating pipeline work.
            return False


def _http_status(head: bytes) -> int:
    """Parse the numeric status code out of an HTTP response head.

    Returns 0 when the status line is absent or unparseable — callers treat
    that as "not a status we special-case", never as success.
    """
    try:
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        return int(status_line.split(" ")[1])
    except (IndexError, ValueError, UnicodeDecodeError):
        return 0
