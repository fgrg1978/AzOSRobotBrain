# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Control plane package for AzOSRobotBrain.

The control plane is the thin, stateless tier that:
  - Validates bearer tokens for incoming fleet-management requests.
  - Tracks which data-plane worker owns each robot_id via consistent hashing.
  - Fans out fleet-wide OTA pushes to the responsible data planes.
  - Answers read-only fleet queries from operators / dashboards.

Single-process mode (--data-planes 0):
  The control plane and data plane run in the same asyncio event loop; the
  ``ShardCoordinator`` is instantiated in-process and assignment lookups are
  direct Python calls with no HTTP round-trips.

Multi-process mode (--data-planes N):
  Each data plane worker runs as a separate ``multiprocessing.Process``.
  The control plane exposes a tiny HTTP API so workers can validate ownership;
  workers reverse-query it on each new kernel connection.

  NOTE: multi-process mode requires a shared backend (Redis) so that the fleet
  registry and heartbeat state is visible across process boundaries.  The
  current ``InMemoryBackend`` is only valid for single-process mode.
  See ``tools/start_split.py`` for the ``--allow-no-shared-state`` guard.
"""
