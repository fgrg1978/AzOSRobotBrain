# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Fleet sharding — consistent-hash ring + shard coordinator + backend abstraction.

S3 control plane: allows a 1000-robot fleet to be partitioned across N brain
data-plane instances by consistent-hashing on ``robot_id``.

Architecture overview
---------------------
- ``ConsistentHashRing`` — pure hash-ring, no I/O.  Maps robot_id → node name.
- ``ShardCoordinator`` — wraps a ring and a "self" node identity; answers
  "is this robot mine?" and "where should I forward it?".
- ``BackendProtocol`` — typing.Protocol (structural interface) for the shared
  robot-state store so the concrete implementation can be swapped without
  touching callers.
- ``InMemoryBackend`` — default implementation using a plain dict + asyncio.Lock.
  One module swap away from a ``RedisBackend`` that implements the same Protocol.

Adding a Redis backend later::

    from fleet_sharding import BackendProtocol
    class RedisBackend(BackendProtocol):
        ...   # implements register_robot / update_heartbeat / query_robot / list_robots

No external dependencies — stdlib only (``hashlib``, ``bisect``, ``asyncio``).
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

# Default number of virtual nodes (ring slots) per physical node.
# Higher values → more uniform distribution at the cost of slightly more memory.
VIRTUAL_NODES_DEFAULT: int = 128

# Number of bytes from the SHA-256 digest used as the ring position integer.
# 8 bytes = 64-bit hash space — enough uniformity for any realistic fleet size.
HASH_BYTES: int = 8

# Sentinel value returned when the ring is empty.
_EMPTY_RING_NODE: str = ""

# Separator between node name and replica index in the virtual-node key.
_VNODE_SEP: str = "#"


# ---------------------------------------------------------------------------
# ConsistentHashRing
# ---------------------------------------------------------------------------


class ConsistentHashRing:
    """Consistent-hash ring that maps arbitrary string keys to node names.

    Uses SHA-256 for key hashing so distribution is uniform and deterministic
    across Python processes and platforms.

    Virtual nodes (replicas per physical node) smooth out load distribution:
    with ``virtual_nodes=128`` and 3 physical nodes → 384 ring slots, giving
    each node roughly N/3 ± 5 % of keys even for small keysets.

    Thread safety: **not** thread-safe on its own.  Wrap with a lock if
    mutated from multiple asyncio tasks or threads.
    """

    def __init__(
        self,
        nodes: List[str],
        virtual_nodes: int = VIRTUAL_NODES_DEFAULT,
    ) -> None:
        """Build a ring from *nodes* with *virtual_nodes* replicas each.

        Args:
            nodes:         Initial list of physical-node identifiers (e.g.
                           ``["brain-0:5000", "brain-1:5000"]``).
            virtual_nodes: Number of ring positions per physical node.
        """
        if virtual_nodes < 1:
            raise ValueError(f"virtual_nodes must be >= 1, got {virtual_nodes}")
        self._virtual_nodes = virtual_nodes
        # Sorted list of (hash_int, node_name) — the ring.
        self._ring: List[Tuple[int, str]] = []
        # Sorted hash positions only — kept in sync for bisect.
        self._positions: List[int] = []
        # Set of known physical nodes — used for existence checks.
        self._nodes: set[str] = set()

        for node in nodes:
            self._add_node_unchecked(node)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _hash(key: str) -> int:
        """Return a 64-bit integer hash for *key* using SHA-256."""
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest[:HASH_BYTES], "big")

    def _vnode_key(self, node: str, replica: int) -> str:
        """Return the virtual-node string that gets hashed onto the ring."""
        return f"{node}{_VNODE_SEP}{replica}"

    def _add_node_unchecked(self, node: str) -> None:
        """Insert *virtual_nodes* slots for *node* (no duplicate guard)."""
        for replica in range(self._virtual_nodes):
            pos = self._hash(self._vnode_key(node, replica))
            idx = bisect.bisect_left(self._positions, pos)
            self._positions.insert(idx, pos)
            self._ring.insert(idx, (pos, node))
        self._nodes.add(node)

    # ── Public API ────────────────────────────────────────────────────────

    def add_node(self, node: str) -> None:
        """Add *node* to the ring (idempotent — ignored if already present).

        Only ~1/N of existing keys shift to the new node; all other
        assignments remain stable.
        """
        if node in self._nodes:
            return
        self._add_node_unchecked(node)

    def remove_node(self, node: str) -> None:
        """Remove *node* from the ring (no-op if not present).

        Only ~1/N keys (those previously owned by *node*) move to their new
        successor; all other assignments remain stable.
        """
        if node not in self._nodes:
            return
        # Rebuild ring without entries for this node (O(V) where V = vnodes).
        new_ring: List[Tuple[int, str]] = [(pos, n) for pos, n in self._ring if n != node]
        self._ring = new_ring
        self._positions = [pos for pos, _ in new_ring]
        self._nodes.discard(node)

    def route(self, robot_id: str) -> str:
        """Return the node responsible for *robot_id*.

        Uses clockwise-successor lookup on the hash ring.  Returns the empty
        string if the ring is empty.
        """
        if not self._ring:
            return _EMPTY_RING_NODE
        pos = self._hash(robot_id)
        idx = bisect.bisect_left(self._positions, pos)
        # Wrap around — if pos > all positions, go to ring[0].
        idx = idx % len(self._ring)
        return self._ring[idx][1]

    @staticmethod
    def redistribution_cost(
        old_ring: "ConsistentHashRing",
        new_ring: "ConsistentHashRing",
        keys: Iterable[str],
    ) -> Dict[str, int]:
        """Count how many keys in *keys* would move between the two ring states.

        Returns a dict mapping each key that *changes owner* to 1 (for easy
        ``sum(cost.values())`` aggregation).  Keys that stay on the same node
        are absent from the result.

        This is a capacity-planning helper: given a key universe (e.g. sampled
        robot_ids) it quantifies the churn when scaling brain-node count.

        Args:
            old_ring: Ring state before the change.
            new_ring: Ring state after the change.
            keys:     Iterable of robot_ids to evaluate.

        Returns:
            Dict[robot_id, 1] for every key whose owner differs.
        """
        moved: Dict[str, int] = {}
        for key in keys:
            old_owner = old_ring.route(key)
            new_owner = new_ring.route(key)
            if old_owner != new_owner:
                moved[key] = 1
        return moved

    @property
    def nodes(self) -> frozenset[str]:
        """Return the current set of physical nodes."""
        return frozenset(self._nodes)

    @property
    def slot_count(self) -> int:
        """Total number of ring slots (physical nodes × virtual_nodes)."""
        return len(self._ring)

    def __repr__(self) -> str:
        return (
            f"ConsistentHashRing(nodes={len(self._nodes)}, "
            f"slots={self.slot_count}, vnodes={self._virtual_nodes})"
        )


# ---------------------------------------------------------------------------
# ShardCoordinator
# ---------------------------------------------------------------------------


class ShardCoordinator:
    """Wraps a ``ConsistentHashRing`` with a local-node identity.

    A brain instance creates one ``ShardCoordinator`` pointing at *itself* and
    uses it to answer "do I own this robot?" and "where should I forward it?".

    Example::

        ring = ConsistentHashRing(["brain-0:5000", "brain-1:5000"])
        coord = ShardCoordinator(ring, self_node="brain-0:5000")

        if coord.is_local("robot_42"):
            handle_locally()
        else:
            forward_to(coord.forward_target("robot_42"))
    """

    def __init__(
        self,
        ring: ConsistentHashRing,
        self_node: str,
    ) -> None:
        """
        Args:
            ring:      Shared ``ConsistentHashRing`` instance (may be mutated
                       externally as nodes join/leave).
            self_node: The network address / identifier of *this* brain instance
                       (must match a node name in *ring*).
        """
        self._ring = ring
        self._self_node = self_node

    # ── Routing ───────────────────────────────────────────────────────────

    def is_local(self, robot_id: str) -> bool:
        """Return True if this brain instance owns *robot_id* on the ring."""
        return self._ring.route(robot_id) == self._self_node

    def forward_target(self, robot_id: str) -> Optional[str]:
        """Return the owning node address for a non-local *robot_id*.

        Returns:
            The node address string if *robot_id* belongs to another node.
            ``None`` if *robot_id* belongs to this node (i.e. ``is_local``).
        """
        owner = self._ring.route(robot_id)
        if owner == self._self_node:
            return None
        return owner

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def self_node(self) -> str:
        """The identifier of this brain instance."""
        return self._self_node

    @property
    def ring(self) -> ConsistentHashRing:
        """The underlying ``ConsistentHashRing``."""
        return self._ring

    def __repr__(self) -> str:
        return f"ShardCoordinator(self={self._self_node!r}, " f"ring={self._ring!r})"


# ---------------------------------------------------------------------------
# BackendProtocol
# ---------------------------------------------------------------------------


class BackendProtocol(Protocol):
    """Structural interface for the shared robot-state store.

    Any class that implements these four ``async`` methods satisfies the
    protocol — no inheritance required (PEP 544).  This keeps the door open
    for swapping ``InMemoryBackend`` for ``RedisBackend`` without changing
    the ``FleetManager`` or ``ShardCoordinator`` code.
    """

    async def register_robot(self, robot_id: str, meta: Dict[str, Any]) -> None:
        """Persist initial registration metadata for *robot_id*."""
        ...

    async def update_heartbeat(self, robot_id: str, ts: float) -> None:
        """Record a heartbeat timestamp for *robot_id*."""
        ...

    async def query_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Return stored metadata for *robot_id*, or ``None`` if unknown."""
        ...

    async def list_robots(
        self, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Return all stored robot records, optionally filtered by *filter_fn*."""
        ...


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class InMemoryBackend:
    """Default in-process backend — asyncio.Lock-protected dict store.

    All state is lost on process restart.  Suitable for single-instance brain
    deployments and for unit tests.

    Satisfies ``BackendProtocol`` structurally.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def register_robot(self, robot_id: str, meta: Dict[str, Any]) -> None:
        """Create or overwrite the record for *robot_id*."""
        async with self._lock:
            self._store[robot_id] = {
                "robot_id": robot_id,
                "registered_at": time.time(),
                "last_heartbeat": 0.0,
                **meta,
            }

    async def update_heartbeat(self, robot_id: str, ts: float) -> None:
        """Update the ``last_heartbeat`` field for *robot_id*.

        If the robot is not registered yet, the update is silently dropped
        (caller should call ``register_robot`` first).
        """
        async with self._lock:
            if robot_id in self._store:
                self._store[robot_id]["last_heartbeat"] = ts

    async def query_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Return a *copy* of the record for *robot_id*, or ``None``."""
        async with self._lock:
            record = self._store.get(robot_id)
            return dict(record) if record is not None else None

    async def list_robots(
        self, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Return copies of all records, optionally filtered by *filter_fn*."""
        async with self._lock:
            records = [dict(r) for r in self._store.values()]
        if filter_fn is None:
            return records
        return [r for r in records if filter_fn(r)]

    @property
    def robot_count(self) -> int:
        """Number of registered robots (snapshot, no lock)."""
        return len(self._store)

    def __repr__(self) -> str:
        return f"InMemoryBackend(robots={self.robot_count})"
