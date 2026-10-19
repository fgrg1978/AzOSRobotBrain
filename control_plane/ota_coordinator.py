# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""OTA dispatch coordination for the control plane.

The control plane receives a fleet-wide OTA request and fans it out to each
data-plane worker that owns at least one of the targeted robots.  Each
data-plane worker is responsible for pushing the firmware to the robots it
manages.

In single-process mode (no external data planes), the coordinator calls back
into the local ``FleetOtaManager`` directly so no HTTP round-trips are needed.

In multi-process mode the coordinator sends an HTTP POST to each data-plane's
``/internal/ota`` endpoint with the robot list and a signed job descriptor.

State is entirely in-memory (Python dict).  Multi-process deployments REQUIRE
a shared backend (Redis) for job state to be consistent across the control
plane and all data planes.  A TODO marker is left at the persistence point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("brain.control_plane.ota_coordinator")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Maximum number of concurrent per-data-plane HTTP dispatch calls.
OTA_MAX_CONCURRENT_DISPATCHES: int = 8

#: Timeout in seconds for a single HTTP dispatch to a data-plane worker.
OTA_DISPATCH_TIMEOUT_S: float = 10.0

#: Maximum number of OTA jobs retained in memory before the oldest is evicted.
OTA_MAX_JOB_HISTORY: int = 64

#: HTTP path used to notify a data-plane worker of an OTA job.
_DP_OTA_ENDPOINT: str = "/internal/ota"

#: HTTP method for the dispatch notification.
_DP_OTA_METHOD: str = "POST"


class OtaJobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class OtaJob:
    job_id: str
    robot_ids: list[str]  # target robots (empty = all online)
    platform: str
    fw_version: int
    state: OtaJobState = OtaJobState.PENDING
    created_at: float = field(default_factory=time.time)
    per_dp: dict[str, str] = field(default_factory=dict)  # dp_addr -> status


# Type alias for the optional local-dispatch callback (single-process mode).
LocalDispatchFn = Callable[[OtaJob], Coroutine[Any, Any, None]]


class OtaCoordinator:
    """Fan-out OTA requests from the control plane to data-plane workers."""

    def __init__(
        self,
        *,
        local_dispatch: Optional[LocalDispatchFn] = None,
    ) -> None:
        """
        Args:
            local_dispatch: If provided, used instead of HTTP calls.  Pass a
                coroutine function ``async def dispatch(job) -> None`` that
                delegates to the local ``FleetOtaManager``.  This is the
                single-process path.
        """
        self._jobs: dict[str, OtaJob] = {}
        self._local_dispatch = local_dispatch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_job(
        self,
        *,
        robot_ids: list[str],
        platform: str,
        fw_version: int,
        dp_addresses: list[str],
        dp_robot_map: dict[str, list[str]],  # dp_addr -> [robot_id, ...]
    ) -> str:
        """Create and start a fleet OTA job.  Returns the job_id.

        Args:
            robot_ids:    Robots targeted by this job.
            platform:     Platform tag forwarded to data planes.
            fw_version:   Firmware version number.
            dp_addresses: All data-plane addresses (for routing).
            dp_robot_map: Pre-computed mapping of dp_addr → robot_ids it owns.
        """
        job_id = secrets.token_hex(8)
        job = OtaJob(
            job_id=job_id,
            robot_ids=robot_ids,
            platform=platform,
            fw_version=fw_version,
            state=OtaJobState.RUNNING,
        )
        self._evict_if_needed()
        self._jobs[job_id] = job
        logger.info("[OtaCoordinator] job=%s robots=%s", job_id, robot_ids)

        asyncio.create_task(self._dispatch_to_data_planes(job, dp_robot_map))
        return job_id

    def get_job(self, job_id: str) -> Optional[OtaJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "job_id": j.job_id,
                "state": j.state.value,
                "robots": j.robot_ids,
                "platform": j.platform,
                "fw_version": j.fw_version,
                "per_dp": j.per_dp,
            }
            for j in self._jobs.values()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _dispatch_to_data_planes(
        self,
        job: OtaJob,
        dp_robot_map: dict[str, list[str]],
    ) -> None:
        """Fan-out to each relevant data plane concurrently."""
        sem = asyncio.Semaphore(OTA_MAX_CONCURRENT_DISPATCHES)
        tasks = []
        for dp_addr, robots in dp_robot_map.items():
            if robots:
                tasks.append(asyncio.create_task(self._dispatch_one(sem, job, dp_addr, robots)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Mark job done/failed based on per_dp statuses.
        failed = any(v == "FAILED" for v in job.per_dp.values())
        job.state = OtaJobState.FAILED if failed else OtaJobState.DONE
        logger.info("[OtaCoordinator] job=%s final_state=%s", job.job_id, job.state.value)

    async def _dispatch_one(
        self,
        sem: asyncio.Semaphore,
        job: OtaJob,
        dp_addr: str,
        robots: list[str],
    ) -> None:
        async with sem:
            if self._local_dispatch is not None:
                # Single-process path: call local handler directly.
                try:
                    await asyncio.wait_for(
                        self._local_dispatch(job),
                        timeout=OTA_DISPATCH_TIMEOUT_S,
                    )
                    job.per_dp[dp_addr] = "OK"
                except Exception as exc:
                    logger.error("[OtaCoordinator] local dispatch error: %s", exc)
                    job.per_dp[dp_addr] = "FAILED"
                return

            # Multi-process path: HTTP POST to data-plane worker.
            # TODO: replace with shared-state backend call once Redis backend
            #       lands.  Currently this is a best-effort fire-and-forget
            #       HTTP POST; no persistent state is written.
            try:
                host, port_str = dp_addr.rsplit(":", 1)
                port = int(port_str)
                payload = json.dumps(
                    {
                        "job_id": job.job_id,
                        "robot_ids": robots,
                        "platform": job.platform,
                        "fw_version": job.fw_version,
                    }
                ).encode()
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=OTA_DISPATCH_TIMEOUT_S,
                )
                request = (
                    f"{_DP_OTA_METHOD} {_DP_OTA_ENDPOINT} HTTP/1.1\r\n"
                    f"Host: {dp_addr}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                ).encode() + payload
                writer.write(request)
                await writer.drain()
                resp_bytes = await asyncio.wait_for(
                    reader.read(4096), timeout=OTA_DISPATCH_TIMEOUT_S
                )
                writer.close()
                status_line = resp_bytes.split(b"\r\n")[0].decode(errors="replace")
                if "200" in status_line:
                    job.per_dp[dp_addr] = "OK"
                else:
                    job.per_dp[dp_addr] = f"HTTP_ERROR:{status_line}"
                    logger.warning("[OtaCoordinator] dp=%s responded: %s", dp_addr, status_line)
            except Exception as exc:
                logger.error("[OtaCoordinator] dp=%s dispatch failed: %s", dp_addr, exc)
                job.per_dp[dp_addr] = "FAILED"

    def _evict_if_needed(self) -> None:
        """Remove oldest job when history cap is reached."""
        if len(self._jobs) >= OTA_MAX_JOB_HISTORY:
            oldest_key = next(iter(self._jobs))
            del self._jobs[oldest_key]
