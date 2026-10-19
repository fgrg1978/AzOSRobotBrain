# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Transport abstraction — multi-link communication layer (Phase Z).

Abstracts the communication link between brain and robot, supporting
multiple transports: TCP (Ethernet/WiFi), Serial (UART), UDP, and
future links (LoRa, RF, 4G).

The brain can use whichever transport is available, with automatic
failover: TCP → Serial → UDP.

Usage:
    transport = TransportManager()
    transport.add_link(TcpLink("192.168.1.10", 9000))
    transport.add_link(SerialLink("/dev/ttyUSB0", 115200))
    await transport.connect()
    await transport.send(packet_bytes)
    data = await transport.recv()
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("brain.transport")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONNECT_TIMEOUT_S = 5.0
RECONNECT_DELAY_S = 3.0
LINK_HEALTH_TIMEOUT_S = 10.0  # link considered dead after no data
MAX_SEND_RETRIES = 2


# ---------------------------------------------------------------------------
# Transport link interface
# ---------------------------------------------------------------------------


class TransportLink(ABC):
    """Abstract base for a communication link."""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # lower = preferred
        self.connected = False
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.last_recv_time: float = 0.0
        self.errors = 0

    @abstractmethod
    async def connect(self) -> bool: ...

    @abstractmethod
    async def disconnect(self): ...

    @abstractmethod
    async def send(self, data: bytes) -> bool: ...

    @abstractmethod
    async def recv(self, max_bytes: int = 4096) -> bytes: ...

    @property
    def healthy(self) -> bool:
        if not self.connected:
            return False
        if self.last_recv_time == 0:
            return True  # no data yet, assume OK
        return (time.time() - self.last_recv_time) < LINK_HEALTH_TIMEOUT_S


# ---------------------------------------------------------------------------
# TCP link
# ---------------------------------------------------------------------------


class TcpLink(TransportLink):
    """TCP transport (Ethernet or WiFi)."""

    def __init__(self, host: str, port: int, priority: int = 0):
        super().__init__(name=f"tcp:{host}:{port}", priority=priority)
        self._host = host
        self._port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=CONNECT_TIMEOUT_S,
            )
            self.connected = True
            logger.info("[Transport] TCP connected to %s:%d", self._host, self._port)
            return True
        except Exception as e:
            logger.debug("[Transport] TCP connect failed: %s", e)
            self.errors += 1
            return False

    async def disconnect(self):
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self.connected = False
        self._reader = None
        self._writer = None

    async def send(self, data: bytes) -> bool:
        if not self._writer:
            return False
        try:
            self._writer.write(data)
            await self._writer.drain()
            self.bytes_sent += len(data)
            return True
        except Exception as e:
            logger.debug("[Transport] TCP send failed: %s", e)
            self.connected = False
            self.errors += 1
            return False

    async def recv(self, max_bytes: int = 4096) -> bytes:
        if not self._reader:
            return b""
        try:
            data = await asyncio.wait_for(
                self._reader.read(max_bytes),
                timeout=1.0,
            )
            if data:
                self.bytes_recv += len(data)
                self.last_recv_time = time.time()
            return data
        except asyncio.TimeoutError:
            return b""
        except Exception:
            self.connected = False
            return b""


# ---------------------------------------------------------------------------
# Serial link
# ---------------------------------------------------------------------------


class SerialLink(TransportLink):
    """Serial/UART transport (direct or via USB-serial adapter)."""

    def __init__(self, port: str, baud: int = 115200, priority: int = 10):
        super().__init__(name=f"serial:{port}", priority=priority)
        self._port = port
        self._baud = baud
        self._serial = None

    async def connect(self) -> bool:
        try:
            import serial

            self._serial = serial.Serial(
                self._port,
                self._baud,
                timeout=0.1,
            )
            self.connected = True
            logger.info("[Transport] Serial connected to %s @ %d", self._port, self._baud)
            return True
        except ImportError:
            logger.error("[Transport] pyserial not installed")
            return False
        except Exception as e:
            logger.debug("[Transport] Serial connect failed: %s", e)
            self.errors += 1
            return False

    async def disconnect(self):
        if self._serial:
            self._serial.close()
        self.connected = False
        self._serial = None

    async def send(self, data: bytes) -> bool:
        if not self._serial:
            return False
        try:
            written = await asyncio.to_thread(self._serial.write, data)
            self.bytes_sent += written
            return True
        except Exception as e:
            logger.debug("[Transport] Serial send failed: %s", e)
            self.connected = False
            self.errors += 1
            return False

    async def recv(self, max_bytes: int = 4096) -> bytes:
        if not self._serial:
            return b""
        try:
            data = await asyncio.to_thread(
                self._serial.read,
                min(max_bytes, self._serial.in_waiting or 1),
            )
            if data:
                self.bytes_recv += len(data)
                self.last_recv_time = time.time()
            return data
        except Exception:
            return b""


# ---------------------------------------------------------------------------
# UDP link
# ---------------------------------------------------------------------------


class UdpLink(TransportLink):
    """UDP transport (connectionless, for telemetry or LoRa bridges)."""

    def __init__(self, host: str, port: int, priority: int = 20):
        super().__init__(name=f"udp:{host}:{port}", priority=priority)
        self._host = host
        self._port = port
        self._transport = None
        self._protocol = None

    async def connect(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            self._transport, self._protocol = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=(self._host, self._port),
            )
            self.connected = True
            return True
        except Exception as e:
            logger.debug("[Transport] UDP connect failed: %s", e)
            return False

    async def disconnect(self):
        if self._transport:
            self._transport.close()
        self.connected = False

    async def send(self, data: bytes) -> bool:
        if not self._transport:
            return False
        try:
            self._transport.sendto(data)
            self.bytes_sent += len(data)
            return True
        except Exception:
            return False

    async def recv(self, max_bytes: int = 4096) -> bytes:
        return b""  # UDP recv requires protocol callback, simplified here


# ---------------------------------------------------------------------------
# TransportManager — multi-link with failover
# ---------------------------------------------------------------------------


class TransportManager:
    """Manages multiple transport links with automatic failover."""

    def __init__(self):
        self._links: list[TransportLink] = []
        self._active: Optional[TransportLink] = None

    def add_link(self, link: TransportLink):
        """Add a transport link (sorted by priority)."""
        self._links.append(link)
        self._links.sort(key=lambda l: l.priority)

    @property
    def active_link(self) -> Optional[TransportLink]:
        return self._active

    @property
    def link_count(self) -> int:
        return len(self._links)

    async def connect(self) -> bool:
        """Try to connect using the highest-priority available link."""
        for link in self._links:
            if await link.connect():
                self._active = link
                return True
        return False

    async def disconnect(self):
        """Disconnect all links."""
        for link in self._links:
            if link.connected:
                await link.disconnect()
        self._active = None

    async def send(self, data: bytes) -> bool:
        """Send data on active link, failover if needed."""
        if self._active and self._active.healthy:
            if await self._active.send(data):
                return True

        # Failover: try other links
        for link in self._links:
            if link == self._active:
                continue
            if not link.connected:
                await link.connect()
            if link.connected and await link.send(data):
                self._active = link
                logger.info("[Transport] Failover to %s", link.name)
                return True

        return False

    async def recv(self, max_bytes: int = 4096) -> bytes:
        """Receive data from active link."""
        if self._active and self._active.connected:
            return await self._active.recv(max_bytes)
        return b""

    def status(self) -> dict:
        """Status of all links."""
        return {
            "active": self._active.name if self._active else None,
            "links": [
                {
                    "name": l.name,
                    "connected": l.connected,
                    "healthy": l.healthy,
                    "sent": l.bytes_sent,
                    "recv": l.bytes_recv,
                    "errors": l.errors,
                }
                for l in self._links
            ],
        }
