# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Transport Multi-Link Manager (E02).

Manages multiple communication links to the robot (WiFi, LoRa, RF, 4G)
with automatic failover, priority-based selection, and heartbeat monitoring.

Architecture:
  Brain ──WiFi──→ Robot (primary: high bandwidth, low latency)
        ──LoRa──→ Robot (backup: low bandwidth, long range)
        ──4G────→ Robot (backup: medium bandwidth, cellular)

When the primary link fails, traffic automatically switches to the next
available link. Heartbeat probes monitor link health independently.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("brain.multilink")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

## Heartbeat probe interval per link (seconds).
HEARTBEAT_INTERVAL_S = 5.0

## Link considered dead after this many missed heartbeats.
HEARTBEAT_MISS_THRESHOLD = 3

## Time to wait before switching back to a recovered higher-priority link.
FAILBACK_DELAY_S = 10.0

## Maximum number of transport links.
MAX_LINKS = 4

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LinkType(Enum):
    WIFI = "wifi"
    LORA = "lora"
    RF = "rf"
    CELLULAR = "4g"


class LinkState(Enum):
    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class TransportLink:
    """A single communication link to the robot."""

    name: str
    link_type: LinkType
    priority: int  # lower = higher priority (0 = primary)
    host: str = ""
    port: int = 0
    state: LinkState = LinkState.UNKNOWN
    last_heartbeat: float = 0.0
    missed_heartbeats: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    latency_ms: float = 0.0
    recovery_time: float = 0.0  # when link came back up

    @property
    def is_alive(self) -> bool:
        return self.state in (LinkState.UP, LinkState.DEGRADED)


# ---------------------------------------------------------------------------
# MultiLinkManager
# ---------------------------------------------------------------------------


class MultiLinkManager:
    """Manages multiple transport links with automatic failover."""

    def __init__(self, config: dict | None = None):
        self._links: list[TransportLink] = []
        self._active_idx: int = -1
        self._failback_delay = FAILBACK_DELAY_S

        if config:
            self._load_config(config)

    @property
    def active_link(self) -> TransportLink | None:
        if 0 <= self._active_idx < len(self._links):
            return self._links[self._active_idx]
        return None

    @property
    def links(self) -> list[TransportLink]:
        return self._links

    def add_link(self, link: TransportLink):
        """Register a transport link."""
        self._links.append(link)
        # Sort by priority (lower = higher priority)
        self._links.sort(key=lambda l: l.priority)
        logger.info(
            "Added link: %s (%s) priority=%d", link.name, link.link_type.value, link.priority
        )

    def on_heartbeat_received(self, link_name: str, latency_ms: float = 0.0):
        """Called when a heartbeat response arrives from a link."""
        for i, link in enumerate(self._links):
            if link.name == link_name:
                was_down = link.state == LinkState.DOWN
                link.state = LinkState.UP
                link.last_heartbeat = time.time()
                link.missed_heartbeats = 0
                link.latency_ms = latency_ms
                if was_down:
                    link.recovery_time = time.time()
                    logger.info("Link %s recovered", link_name)
                break

    def on_data_received(self, link_name: str, byte_count: int):
        """Track received bytes per link."""
        for link in self._links:
            if link.name == link_name:
                link.bytes_recv += byte_count
                link.last_heartbeat = time.time()
                link.state = LinkState.UP
                break

    def on_data_sent(self, link_name: str, byte_count: int):
        """Track sent bytes per link."""
        for link in self._links:
            if link.name == link_name:
                link.bytes_sent += byte_count
                break

    def tick(self) -> str | None:
        """Periodic health check. Returns action taken or None.

        Call every HEARTBEAT_INTERVAL_S seconds.
        """
        now = time.time()
        action = None

        # Check each link's health
        for link in self._links:
            if link.state == LinkState.UNKNOWN:
                continue
            elapsed = now - link.last_heartbeat if link.last_heartbeat > 0 else 0
            if elapsed > HEARTBEAT_INTERVAL_S * 1.5:
                link.missed_heartbeats += 1
                if link.missed_heartbeats >= HEARTBEAT_MISS_THRESHOLD:
                    if link.state != LinkState.DOWN:
                        link.state = LinkState.DOWN
                        logger.warning(
                            "Link %s DOWN (%d missed heartbeats)", link.name, link.missed_heartbeats
                        )
                elif link.state == LinkState.UP:
                    link.state = LinkState.DEGRADED

        # Select best active link
        prev_idx = self._active_idx
        best_idx = self._select_best_link(now)

        if best_idx != prev_idx and best_idx >= 0:
            if prev_idx >= 0:
                old_name = self._links[prev_idx].name
                new_name = self._links[best_idx].name
                logger.warning("Failover: %s → %s", old_name, new_name)
                action = f"failover:{old_name}→{new_name}"
            self._active_idx = best_idx

        if self._active_idx < 0 and self._links:
            # No active link at all
            action = "all_links_down"

        return action

    def get_send_target(self) -> tuple[str, int] | None:
        """Get (host, port) of the currently active link for sending."""
        link = self.active_link
        if link and link.is_alive:
            return (link.host, link.port)
        return None

    def status_text(self) -> str:
        """Human-readable status for Telegram /links command."""
        lines = []
        for i, link in enumerate(self._links):
            active = " [ACTIVE]" if i == self._active_idx else ""
            lines.append(
                f"{link.name} ({link.link_type.value}): {link.state.value} "
                f"prio={link.priority} lat={link.latency_ms:.0f}ms "
                f"tx={link.bytes_sent} rx={link.bytes_recv}{active}"
            )
        return "\n".join(lines) if lines else "No links configured"

    # ── Internal ─────────────────────────────────────────────────────────

    def _select_best_link(self, now: float) -> int:
        """Select the highest-priority alive link, with failback delay."""
        for i, link in enumerate(self._links):
            if not link.is_alive:
                continue
            # If this link just recovered, wait for failback delay
            if (
                link.recovery_time > 0
                and now - link.recovery_time < self._failback_delay
                and i != self._active_idx
            ):
                continue
            return i
        return -1

    def _load_config(self, config: dict):
        """Load links from config dict."""
        links_cfg = config.get("links", [])
        for i, lc in enumerate(links_cfg[:MAX_LINKS]):
            link_type = LinkType(lc.get("type", "wifi"))
            self.add_link(
                TransportLink(
                    name=lc.get("name", f"link_{i}"),
                    link_type=link_type,
                    priority=lc.get("priority", i),
                    host=lc.get("host", ""),
                    port=lc.get("port", 9000),
                )
            )
