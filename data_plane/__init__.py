# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Data-plane package for AzOSRobotBrain.

A data-plane worker:
  - Accepts TCP connections from robot kernels on a dedicated port.
  - On each new connection, validates ownership: queries the control plane
    (or, in single-process mode, a local ShardCoordinator) to confirm that
    THIS data-plane node is responsible for the incoming robot_id.
  - If ownership is confirmed, gates the connection through the existing
    perception / planner / policy pipeline (imported lazily from the
    project-level modules to avoid heavy imports at module load time).
  - Reports metrics back to the control plane (stub; full wiring deferred
    until the S5 metrics agent lands).

Single-process default: when ``self_node`` equals ``CP_INPROCESS_SENTINEL``,
the data plane uses an injected ``ShardCoordinator`` reference instead of
HTTP and runs entirely within the same asyncio event loop.
"""

#: Sentinel value for ``self_node`` that signals in-process (single-process) mode.
CP_INPROCESS_SENTINEL: str = "inprocess"
