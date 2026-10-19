# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Launcher for the split control-plane + data-plane deployment.

Usage:

    # Single-process (backward-compatible, no multi-process overhead):
    python tools/start_split.py --data-planes 0

    # Multi-process (2 data-plane workers on ports 9100 and 9101):
    python tools/start_split.py --data-planes 2 --allow-no-shared-state

    # With custom ports:
    python tools/start_split.py --data-planes 2 --dp-base-port 9200 \\
        --cp-port 8090 --allow-no-shared-state

Authentication (required — the launcher no longer decides for you):
  Export ``AZOS_BRAIN_CP_API_KEY=<secret>`` before launching.  Data-plane
  workers pick the same value up automatically for their ownership queries to
  the control plane, so one variable covers both sides.  For local work with no
  secret at all, export ``AZOS_BRAIN_ALLOW_INSECURE=1`` *explicitly* — the
  control plane then accepts every request and binds to 127.0.0.1 only.
  Setting neither makes ``BearerAuth`` raise at start-up, on purpose.

Guards:
  - ``--data-planes > 0`` with InMemoryBackend requires ``--allow-no-shared-state``
    or the script exits with an error.  Each subprocess would have its own
    in-memory state; robots registered on data-plane-0 would be invisible to
    data-plane-1 and the control plane.
    # TODO: replace InMemoryBackend with RedisBackend once the shared-state
    #       backend is implemented.  At that point remove the guard or keep it
    #       as a "requires --redis-url" check.

Logs:
  Each subprocess writes its stdout/stderr to
  ``build/split_logs/{role}_{idx}.log`` (created if missing).

Duration:
  ``--duration N`` runs all processes for N seconds then terminates them.
  Useful for smoke tests.  Default: run forever (0 = forever).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("split_launcher")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

#: Default TCP port for the control-plane HTTP API.
DEFAULT_CP_PORT: int = 8090

#: Base port for data-plane workers.  Worker i listens on base + i.
DEFAULT_DP_BASE_PORT: int = 9100

#: Default run duration in seconds.  0 = run forever.
DEFAULT_DURATION_S: int = 0

#: Maximum number of data-plane workers this launcher will spawn.
MAX_DATA_PLANES: int = 32

#: Directory for subprocess log files (relative to project root).
LOG_DIR_RELATIVE: str = "build/split_logs"

#: Prefix for log file names.
LOG_FILE_PREFIX_CP: str = "control_plane"
LOG_FILE_PREFIX_DP: str = "data_plane"

#: Log file extension.
LOG_FILE_EXT: str = ".log"

#: Sleep interval in seconds when polling for process health.
POLL_INTERVAL_S: float = 0.5

#: Grace period in seconds given to each process to terminate cleanly.
TERMINATE_GRACE_S: float = 2.0


# ---------------------------------------------------------------------------
# Process target functions
# ---------------------------------------------------------------------------


def _run_control_plane(
    port: int,
    dp_addresses: list[str],
    log_path: str,
) -> None:
    """Entry point for the control-plane subprocess."""
    _redirect_output(log_path)
    # Project root must be on sys.path for sub-process imports to work.
    _ensure_project_root_on_path()
    import asyncio
    from control_plane.auth import BearerAuth
    from control_plane.discovery import ShardCoordinator
    from control_plane.ota_coordinator import OtaCoordinator
    from control_plane.main import ControlPlaneServer

    # This used to `os.environ.setdefault("AZOS_BRAIN_ALLOW_INSECURE", "1")`,
    # which made the documented way to launch the stack silently disable
    # control-plane auth: the launcher pre-answered the exact question
    # BearerAuth exists to force the operator to answer. Set
    # AZOS_BRAIN_CP_API_KEY (or, deliberately, AZOS_BRAIN_ALLOW_INSECURE=1)
    # in the environment before launching — BearerAuth raises otherwise.
    auth = BearerAuth()
    coord = ShardCoordinator(nodes=dp_addresses)
    ota = OtaCoordinator()
    server = ControlPlaneServer(
        coordinator=coord,
        ota_coordinator=ota,
        auth=auth,
        port=port,
    )
    asyncio.run(server.run())


def _run_data_plane(
    self_node: str,
    cp_address: str,
    port: int,
    log_path: str,
) -> None:
    """Entry point for a data-plane subprocess."""
    _redirect_output(log_path)
    _ensure_project_root_on_path()
    import asyncio
    from control_plane.discovery import ShardCoordinator
    from data_plane.main import DataPlaneServer

    server = DataPlaneServer(
        self_node=self_node,
        cp_address=cp_address,
        port=port,
    )
    asyncio.run(server.run())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redirect_output(log_path: str) -> None:
    """Redirect this process's stdout and stderr to *log_path*."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fd = open(log_path, "w", buffering=1)
    sys.stdout = fd
    sys.stderr = fd


def _ensure_project_root_on_path() -> None:
    """Add the project root (parent of this script's directory) to sys.path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _log_path(role: str, idx: int, project_root: str) -> str:
    """Build the absolute path for a subprocess log file."""
    return os.path.join(project_root, LOG_DIR_RELATIVE, f"{role}_{idx}{LOG_FILE_EXT}")


def _terminate_all(procs: list[multiprocessing.Process]) -> None:
    """Send SIGTERM to all processes and wait for them to exit."""
    for p in procs:
        if p.is_alive():
            p.terminate()
    deadline = time.monotonic() + TERMINATE_GRACE_S
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        p.join(timeout=remaining)
        if p.is_alive():
            logger.warning("Process %s did not stop cleanly; killing.", p.name)
            p.kill()
            p.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start control plane + N data-plane workers.")
    p.add_argument(
        "--data-planes",
        type=int,
        default=0,
        help="Number of data-plane workers to spawn (0 = single-process mode).",
    )
    p.add_argument(
        "--cp-port",
        type=int,
        default=DEFAULT_CP_PORT,
        help=f"Control-plane HTTP port (default: {DEFAULT_CP_PORT}).",
    )
    p.add_argument(
        "--dp-base-port",
        type=int,
        default=DEFAULT_DP_BASE_PORT,
        help=f"Base port for data-plane workers (default: {DEFAULT_DP_BASE_PORT}).",
    )
    p.add_argument(
        "--allow-no-shared-state",
        action="store_true",
        default=False,
        help=(
            "Allow multi-process mode with InMemoryBackend (each process has "
            "its own state — suitable only for local testing, NOT production)."
        ),
    )
    p.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_S,
        help="Run for this many seconds then exit (0 = run forever).",
    )
    return p.parse_args(argv)


def run(argv: Optional[list[str]] = None) -> None:
    """Main entry point (importable so tests can call it directly)."""
    args = _parse_args(argv)

    if args.data_planes < 0 or args.data_planes > MAX_DATA_PLANES:
        sys.exit(
            f"[start_split] --data-planes must be in [0, {MAX_DATA_PLANES}], "
            f"got {args.data_planes}"
        )

    # Sanity guard: multi-process + InMemoryBackend is a footgun.
    # TODO: remove (or convert to --redis-url check) once Redis backend lands.
    if args.data_planes > 0 and not args.allow_no_shared_state:
        sys.exit(
            "[start_split] REQUIRES shared backend (Redis) for multi-process "
            "mode.  Each subprocess has its own in-memory state; robot "
            "registrations are NOT visible across processes.\n"
            "Pass --allow-no-shared-state to proceed anyway (local testing only)."
        )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    procs: list[multiprocessing.Process] = []

    if args.data_planes == 0:
        # Single-process mode: run control plane + one in-process data plane
        # in the same asyncio event loop.
        logger.info("[start_split] single-process mode (--data-planes 0)")
        _run_single_process(
            cp_port=args.cp_port,
            dp_port=args.dp_base_port,
            duration=args.duration,
            project_root=project_root,
        )
        return

    # Multi-process mode.
    cp_address = f"localhost:{args.cp_port}"
    dp_addresses: list[str] = [
        f"localhost:{args.dp_base_port + i}" for i in range(args.data_planes)
    ]

    logger.info(
        "[start_split] spawning 1 control plane + %d data planes",
        args.data_planes,
    )

    # Spawn control plane.
    cp_log = _log_path(LOG_FILE_PREFIX_CP, 0, project_root)
    cp_proc = multiprocessing.Process(
        target=_run_control_plane,
        args=(args.cp_port, dp_addresses, cp_log),
        name="control_plane_0",
        daemon=True,
    )
    cp_proc.start()
    logger.info("[start_split] control_plane_0 pid=%d log=%s", cp_proc.pid, cp_log)
    procs.append(cp_proc)

    # Spawn data planes.
    for i, dp_addr in enumerate(dp_addresses):
        dp_port = args.dp_base_port + i
        dp_log = _log_path(LOG_FILE_PREFIX_DP, i, project_root)
        dp_proc = multiprocessing.Process(
            target=_run_data_plane,
            args=(dp_addr, cp_address, dp_port, dp_log),
            name=f"data_plane_{i}",
            daemon=True,
        )
        dp_proc.start()
        logger.info(
            "[start_split] data_plane_%d pid=%d port=%d log=%s",
            i,
            dp_proc.pid,
            dp_port,
            dp_log,
        )
        procs.append(dp_proc)

    try:
        if args.duration > 0:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_S)
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    logger.error(
                        "[start_split] process(es) died: %s",
                        [p.name for p in dead],
                    )
                    break
        else:
            # Run forever; block until any child dies.
            while True:
                time.sleep(POLL_INTERVAL_S)
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    logger.error(
                        "[start_split] process(es) died: %s",
                        [p.name for p in dead],
                    )
                    break
    except KeyboardInterrupt:
        logger.info("[start_split] interrupted")
    finally:
        _terminate_all(procs)
        logger.info("[start_split] all processes stopped")


def _run_single_process(
    cp_port: int,
    dp_port: int,
    duration: int,
    project_root: str,
) -> None:
    """Run control plane + data plane in one asyncio event loop."""
    import asyncio
    from control_plane.auth import BearerAuth
    from control_plane.discovery import ShardCoordinator
    from control_plane.ota_coordinator import OtaCoordinator
    from control_plane.main import ControlPlaneServer
    from data_plane import CP_INPROCESS_SENTINEL
    from data_plane.main import DataPlaneServer

    # No AZOS_BRAIN_ALLOW_INSECURE setdefault here either — see
    # _run_control_plane. The operator decides; the launcher does not.
    auth = BearerAuth()
    coord = ShardCoordinator()  # empty ring → single data plane owns everything
    ota = OtaCoordinator()
    cp_server = ControlPlaneServer(
        coordinator=coord,
        ota_coordinator=ota,
        auth=auth,
        port=cp_port,
    )
    dp_server = DataPlaneServer(
        self_node=CP_INPROCESS_SENTINEL,
        coordinator=coord,
        port=dp_port,
    )

    async def _run_all() -> None:
        tasks = [
            asyncio.create_task(cp_server.run()),
            asyncio.create_task(dp_server.run()),
        ]
        if duration > 0:
            await asyncio.sleep(duration)
            for t in tasks:
                t.cancel()
            # Await cancellations without propagating CancelledError.
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        else:
            await asyncio.gather(*tasks)

    asyncio.run(_run_all())


if __name__ == "__main__":
    _ensure_project_root_on_path()
    run()
