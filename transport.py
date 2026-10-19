# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""E02 — Multi-link transport client (brain side).

Mirrors the kernel-side `MultiLinkTransport` (see
`AzOS/crates/net/src/multilink.rs`).  A [`MultiLinkClient`] owns several
[`LinkAdapter`] instances (WiFi TCP, LoRa UART, RF UART) and transparently
forwards bytes over the healthiest one.

Design:
  * Each adapter exposes the same coroutine interface
    (`connect`, `send`, `recv`, `is_up`, `link_quality`).
  * The client tries adapters in priority order (lower number == primary).
  * A link is marked "down" after
    ``TRANSPORT_MAX_CONSEC_FAILURES`` failed sends OR no RX for
    ``TRANSPORT_FAILOVER_TIMEOUT_S`` seconds.
  * Every ``LINK_PROBE_INTERVAL_S`` seconds the client probes higher-
    priority links and fails back when the primary recovers.

There is intentionally no dependency on the existing
``planner/transport.py`` — that module targets the generic
``TransportManager``; this one specializes the contract for the brain's
binary protocol loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("brain.multilink")

# ---------------------------------------------------------------------------
# Tunables — every constant is named here, no magic numbers
# ---------------------------------------------------------------------------

#: Seconds before a silent link is considered dead.
TRANSPORT_FAILOVER_TIMEOUT_S: float = 5.0

#: How often (seconds) to re-probe a downed higher-priority link.
LINK_PROBE_INTERVAL_S: float = 2.0

#: Consecutive send failures before an adapter is flagged "down".
TRANSPORT_MAX_CONSEC_FAILURES: int = 3

#: Seconds to wait for an initial TCP connect.
WIFI_CONNECT_TIMEOUT_S: float = 5.0

#: Default recv timeout for each poll (seconds, non-blocking-ish).
DEFAULT_RECV_TIMEOUT_S: float = 0.2

#: Link-quality sentinel: 0..255 scale.
LINK_QUALITY_DOWN: int = 0
LINK_QUALITY_GOOD: int = 200
LINK_QUALITY_UNKNOWN: int = 128

#: Default priority values (lower = preferred).  Match kernel defaults.
WIFI_DEFAULT_PRIORITY: int = 0
LORA_DEFAULT_PRIORITY: int = 10
RF_DEFAULT_PRIORITY: int = 20

#: Max bytes requested per `recv()` call.
DEFAULT_RECV_BUFSIZE: int = 4096


# ---------------------------------------------------------------------------
# Link adapter base
# ---------------------------------------------------------------------------


@dataclass
class LinkStats:
    """Per-adapter rolling counters."""

    sent_bytes: int = 0
    recv_bytes: int = 0
    send_failures: int = 0
    consec_failures: int = 0
    last_recv_ts: float = 0.0
    last_probe_ts: float = 0.0


class LinkAdapter(ABC):
    """Abstract transport adapter — WiFi / LoRa / RF all implement this."""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.stats = LinkStats()
        self._up = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """Bring the link up.  Returns True on success."""

    @abstractmethod
    async def disconnect(self) -> None: ...

    # ── data path ──────────────────────────────────────────────────────────

    @abstractmethod
    async def send(self, data: bytes) -> bool:
        """Return True on success."""

    @abstractmethod
    async def recv(self, max_bytes: int = DEFAULT_RECV_BUFSIZE) -> bytes:
        """Return received bytes, or b'' on timeout."""

    # ── health ─────────────────────────────────────────────────────────────

    def is_up(self) -> bool:
        return self._up

    def link_quality(self) -> int:
        """Driver-supplied quality, 0..255.  Default: GOOD if up, DOWN if not."""
        return LINK_QUALITY_GOOD if self._up else LINK_QUALITY_DOWN

    def mark_rx(self, nbytes: int) -> None:
        self.stats.recv_bytes += nbytes
        self.stats.last_recv_ts = time.time()
        self.stats.consec_failures = 0

    def mark_send_ok(self, nbytes: int) -> None:
        self.stats.sent_bytes += nbytes
        self.stats.consec_failures = 0

    def mark_send_fail(self) -> None:
        self.stats.send_failures += 1
        self.stats.consec_failures += 1


# ---------------------------------------------------------------------------
# WiFi / TCP adapter
# ---------------------------------------------------------------------------


class WiFiAdapter(LinkAdapter):
    """Primary link: TCP over Ethernet / WiFi."""

    def __init__(self, host: str, port: int, priority: int = WIFI_DEFAULT_PRIORITY):
        super().__init__(name=f"wifi:{host}:{port}", priority=priority)
        self._host = host
        self._port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> bool:
        # Close out any previous connection before replacing it — otherwise
        # a reconnect (e.g. after failover-then-failback) leaks the old
        # socket's fd since nothing else still holds a reference to it.
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=WIFI_CONNECT_TIMEOUT_S,
            )
            self._up = True
            logger.info("[multilink] WiFi up: %s:%d", self._host, self._port)
            return True
        except Exception as e:
            logger.debug("[multilink] WiFi connect failed: %s", e)
            self._up = False
            return False

    async def disconnect(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._up = False

    async def send(self, data: bytes) -> bool:
        if self._writer is None:
            self.mark_send_fail()
            return False
        try:
            self._writer.write(data)
            await self._writer.drain()
            self.mark_send_ok(len(data))
            return True
        except Exception as e:
            logger.debug("[multilink] WiFi send failed: %s", e)
            self.mark_send_fail()
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:
                    pass
            self._up = False
            return False

    async def recv(self, max_bytes: int = DEFAULT_RECV_BUFSIZE) -> bytes:
        if self._reader is None:
            return b""
        try:
            data = await asyncio.wait_for(
                self._reader.read(max_bytes),
                timeout=DEFAULT_RECV_TIMEOUT_S,
            )
            if data:
                self.mark_rx(len(data))
            return data
        except asyncio.TimeoutError:
            return b""
        except Exception:
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:
                    pass
            self._up = False
            return b""


# ---------------------------------------------------------------------------
# LoRa adapter — stub over UART (protocol framing only, no actual LoRa HW)
# ---------------------------------------------------------------------------


class LoRaAdapter(LinkAdapter):
    """Secondary link stub — UART-transported LoRa frames.

    Real HW will slot in by swapping the ``_serial`` object with a real
    LoRa modem; the framing / interface is stable.
    """

    def __init__(self, port: str, baud: int = 115_200, priority: int = LORA_DEFAULT_PRIORITY):
        super().__init__(name=f"lora:{port}", priority=priority)
        self._port = port
        self._baud = baud
        self._serial = None

    async def connect(self) -> bool:
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(self._port, self._baud, timeout=0)
            self._up = True
            return True
        except ImportError:
            logger.debug("[multilink] pyserial unavailable — LoRa stub")
            self._up = False
            return False
        except Exception as e:
            logger.debug("[multilink] LoRa connect failed: %s", e)
            self._up = False
            return False

    async def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._up = False

    async def send(self, data: bytes) -> bool:
        if self._serial is None:
            self.mark_send_fail()
            return False
        try:
            await asyncio.to_thread(self._serial.write, data)
            self.mark_send_ok(len(data))
            return True
        except Exception:
            self.mark_send_fail()
            self._up = False
            return False

    async def recv(self, max_bytes: int = DEFAULT_RECV_BUFSIZE) -> bytes:
        if self._serial is None:
            return b""
        try:
            waiting = getattr(self._serial, "in_waiting", 0) or 0
            n = min(max_bytes, max(1, waiting))
            data = await asyncio.to_thread(self._serial.read, n)
            if data:
                self.mark_rx(len(data))
            return data
        except Exception:
            return b""

    def link_quality(self) -> int:
        # LoRa modems expose RSSI; stub returns UNKNOWN.
        return LINK_QUALITY_UNKNOWN if self._up else LINK_QUALITY_DOWN


# ---------------------------------------------------------------------------
# RF adapter — stub for future 900 MHz modules
# ---------------------------------------------------------------------------


class RFAdapter(LinkAdapter):
    """Emergency link stub — 900 MHz RF modem.

    Same interface as LoRa but with lower throughput; treated as
    last-resort fallback.  Real implementation TBD once hardware lands.
    """

    def __init__(self, port: str = "", baud: int = 9_600, priority: int = RF_DEFAULT_PRIORITY):
        super().__init__(name=f"rf:{port or 'stub'}", priority=priority)
        self._port = port
        self._baud = baud
        self._serial = None

    async def connect(self) -> bool:
        if not self._port:
            # No real hardware wired — stay "down" but do not raise.
            self._up = False
            return False
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(self._port, self._baud, timeout=0)
            self._up = True
            return True
        except Exception:
            self._up = False
            return False

    async def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._up = False

    async def send(self, data: bytes) -> bool:
        if self._serial is None:
            self.mark_send_fail()
            return False
        try:
            await asyncio.to_thread(self._serial.write, data)
            self.mark_send_ok(len(data))
            return True
        except Exception:
            self.mark_send_fail()
            self._up = False
            return False

    async def recv(self, max_bytes: int = DEFAULT_RECV_BUFSIZE) -> bytes:
        if self._serial is None:
            return b""
        try:
            waiting = getattr(self._serial, "in_waiting", 0) or 0
            n = min(max_bytes, max(1, waiting))
            data = await asyncio.to_thread(self._serial.read, n)
            if data:
                self.mark_rx(len(data))
            return data
        except Exception:
            return b""

    def link_quality(self) -> int:
        return LINK_QUALITY_UNKNOWN if self._up else LINK_QUALITY_DOWN


# ---------------------------------------------------------------------------
# MultiLinkClient — the orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _MuxState:
    active_idx: int = 0


class MultiLinkClient:
    """Multiplexes N ``LinkAdapter`` instances with priority failover.

    Usage:

        client = MultiLinkClient()
        client.add_link(WiFiAdapter("10.0.0.42", 9000))
        client.add_link(LoRaAdapter("/dev/ttyUSB0"))
        client.add_link(RFAdapter())
        await client.connect_all()
        await client.send(packet_bytes)
        data = await client.recv()
    """

    def __init__(self) -> None:
        self._links: list[LinkAdapter] = []
        self._state = _MuxState()

    # ── setup ──────────────────────────────────────────────────────────────

    def add_link(self, link: LinkAdapter) -> None:
        self._links.append(link)
        self._links.sort(key=lambda l: l.priority)

    @property
    def link_count(self) -> int:
        return len(self._links)

    @property
    def active_link(self) -> Optional[LinkAdapter]:
        if not self._links:
            return None
        idx = min(self._state.active_idx, len(self._links) - 1)
        return self._links[idx]

    @property
    def active_index(self) -> int:
        return self._state.active_idx

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect_all(self) -> bool:
        """Try to bring every link up.  Returns True if at least one came up."""
        any_up = False
        for link in self._links:
            if await link.connect():
                any_up = True
        # Pick the highest-priority link that actually came up.
        self._state.active_idx = 0
        for idx, link in enumerate(self._links):
            if link.is_up():
                self._state.active_idx = idx
                break
        return any_up

    async def disconnect_all(self) -> None:
        for link in self._links:
            await link.disconnect()

    # ── health / failover ──────────────────────────────────────────────────

    def _link_down(self, link: LinkAdapter) -> bool:
        """True if a link should be considered dead right now."""
        if not link.is_up():
            return True
        if link.stats.consec_failures >= TRANSPORT_MAX_CONSEC_FAILURES:
            return True
        last_rx = link.stats.last_recv_ts
        if last_rx == 0.0:
            return False  # never received anything yet — grace period
        return (time.time() - last_rx) >= TRANSPORT_FAILOVER_TIMEOUT_S

    def _pick_healthy(self, skip: int = -1) -> Optional[int]:
        for idx, link in enumerate(self._links):
            if idx == skip:
                continue
            if link.is_up() and not self._link_down(link):
                return idx
        return None

    async def _maybe_failback(self) -> None:
        """Periodically probe higher-priority links; fail back on recovery."""
        now = time.time()
        for idx in range(self._state.active_idx):
            link = self._links[idx]
            if now - link.stats.last_probe_ts < LINK_PROBE_INTERVAL_S:
                continue
            link.stats.last_probe_ts = now
            if not link.is_up():
                await link.connect()
            if link.is_up() and not self._link_down(link):
                logger.info("[multilink] failback to %s", link.name)
                self._state.active_idx = idx
                # Reset counters so it gets a fresh chance.
                link.stats.consec_failures = 0
                link.stats.last_recv_ts = now
                return

    async def _failover(self) -> bool:
        """Switch away from the active link.  Returns True if a new one was found."""
        nxt = self._pick_healthy(skip=self._state.active_idx)
        if nxt is None:
            return False
        old = self._links[self._state.active_idx].name
        new = self._links[nxt].name
        logger.warning("[multilink] failover %s -> %s", old, new)
        self._state.active_idx = nxt
        return True

    # ── data path ──────────────────────────────────────────────────────────

    async def send(self, data: bytes) -> bool:
        if not self._links:
            return False
        await self._maybe_failback()

        # Try active link first.
        active = self._links[self._state.active_idx]
        if active.is_up() and await active.send(data):
            return True

        # Active failed — attempt to fail over, then retry on new active.
        if await self._failover():
            retry = self._links[self._state.active_idx]
            if await retry.send(data):
                return True

        # Last-resort: walk every other link.
        for idx, link in enumerate(self._links):
            if idx == self._state.active_idx:
                continue
            if not link.is_up():
                await link.connect()
            if link.is_up() and await link.send(data):
                self._state.active_idx = idx
                return True
        return False

    async def recv(self, max_bytes: int = DEFAULT_RECV_BUFSIZE) -> bytes:
        if not self._links:
            return b""
        await self._maybe_failback()
        active = self._links[self._state.active_idx]
        if not active.is_up():
            await self._failover()
            active = self._links[self._state.active_idx]
            if not active.is_up():
                return b""
        return await active.recv(max_bytes)

    # ── diagnostics ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "active": self.active_link.name if self.active_link else None,
            "active_index": self._state.active_idx,
            "links": [
                {
                    "name": l.name,
                    "priority": l.priority,
                    "up": l.is_up(),
                    "quality": l.link_quality(),
                    "sent_bytes": l.stats.sent_bytes,
                    "recv_bytes": l.stats.recv_bytes,
                    "send_failures": l.stats.send_failures,
                    "consec_failures": l.stats.consec_failures,
                }
                for l in self._links
            ],
        }
