# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Robot-to-data-plane discovery via consistent hashing.

Wraps ``fleet_sharding.ShardCoordinator`` when it is available.  Until the
S3 agent lands that module, we provide a local ``ShardCoordinator`` stub with
the same interface so the rest of the control-plane code compiles and tests
pass today.

Interface contract (both stub and real):
  coord = ShardCoordinator(nodes=["dp:9100", "dp:9101"])
  node  = coord.assign(robot_id)          -> str  (e.g. "dp:9100")
  nodes = coord.nodes()                   -> list[str]
  coord.add_node("dp:9102")
  coord.remove_node("dp:9100")

Hashing: SHA-256 of robot_id, virtual-node replication factor of
VIRTUAL_NODES_PER_NODE.  The ring is stored as a sorted list of
(hash_int, node_address) tuples so lookups are O(log N) via bisect.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("brain.control_plane.discovery")

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

# Number of virtual ring positions per physical node.
# Higher values give better key distribution at the cost of slightly more
# memory.  128 is a standard default for consistent-hash rings.
VIRTUAL_NODES_PER_NODE: int = 128

# Encoding used when hashing robot_id strings.
_HASH_ENCODING: str = "utf-8"

# Width of the SHA-256 digest converted to an integer (256-bit ring).
_RING_BITS: int = 256
_RING_MODULUS: int = 2**_RING_BITS


# TODO: replace this stub with fleet_sharding.ShardCoordinator once the S3
#       agent lands that module and it is importable from the project root.
#       The interface (assign / nodes / add_node / remove_node) must remain
#       identical.


class ShardCoordinator:
    """Consistent-hash ring for mapping robot_id → data-plane address.

    Thread / coroutine safety: this class is NOT thread-safe.  Call it only
    from a single asyncio event loop thread (the control-plane main loop).
    If you need cross-thread access, wrap with an asyncio.Lock.
    """

    def __init__(self, nodes: Optional[list[str]] = None) -> None:
        # ring: sorted list of (ring_position, node_address)
        self._ring: list[tuple[int, str]] = []
        self._node_set: set[str] = set()
        for node in nodes or []:
            self.add_node(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Insert *node* into the ring with VIRTUAL_NODES_PER_NODE replicas."""
        if node in self._node_set:
            return
        self._node_set.add(node)
        for vn in range(VIRTUAL_NODES_PER_NODE):
            key = f"{node}#{vn}".encode(_HASH_ENCODING)
            pos = int(hashlib.sha256(key).hexdigest(), 16) % _RING_MODULUS
            bisect.insort(self._ring, (pos, node))
        logger.debug("ShardCoordinator: added node %s (%d vn)", node, VIRTUAL_NODES_PER_NODE)

    def remove_node(self, node: str) -> None:
        """Remove *node* and all its virtual positions from the ring."""
        if node not in self._node_set:
            return
        self._node_set.discard(node)
        self._ring = [(pos, n) for (pos, n) in self._ring if n != node]
        logger.debug("ShardCoordinator: removed node %s", node)

    def assign(self, robot_id: str) -> Optional[str]:
        """Return the data-plane address responsible for *robot_id*.

        Returns ``None`` if the ring is empty (no data planes registered).
        The algorithm finds the first virtual node whose ring position is ≥ the
        hash of *robot_id*, wrapping around to the first position when the hash
        exceeds all positions (clockwise wrap).
        """
        if not self._ring:
            return None
        key = robot_id.encode(_HASH_ENCODING)
        pos = int(hashlib.sha256(key).hexdigest(), 16) % _RING_MODULUS
        idx = bisect.bisect_left(self._ring, (pos,))
        # Wrap around the ring.
        if idx >= len(self._ring):
            idx = 0
        _, node = self._ring[idx]
        return node

    def nodes(self) -> list[str]:
        """Return the de-duplicated list of registered node addresses."""
        return sorted(self._node_set)

    def node_count(self) -> int:
        """Number of physical nodes currently in the ring."""
        return len(self._node_set)
