# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/transport.py — multi-link transport abstraction."""

import asyncio
from planner.transport import (
    TransportManager,
    TransportLink,
    TcpLink,
    SerialLink,
    UdpLink,
    CONNECT_TIMEOUT_S,
    RECONNECT_DELAY_S,
    LINK_HEALTH_TIMEOUT_S,
)


class TestConstants:
    def test_connect_timeout(self):
        assert CONNECT_TIMEOUT_S > 0

    def test_reconnect_delay(self):
        assert RECONNECT_DELAY_S > 0

    def test_health_timeout(self):
        assert LINK_HEALTH_TIMEOUT_S > 0


class TestTcpLink:
    def test_init(self):
        l = TcpLink("192.168.1.10", 9000)
        assert l.name == "tcp:192.168.1.10:9000"
        assert not l.connected
        assert l.priority == 0

    def test_priority(self):
        l = TcpLink("x", 1, priority=5)
        assert l.priority == 5


class TestSerialLink:
    def test_init(self):
        l = SerialLink("/dev/ttyUSB0", 115200)
        assert l.name == "serial:/dev/ttyUSB0"
        assert l.priority == 10

    def test_not_connected(self):
        l = SerialLink("/dev/ttyXXX")
        assert not l.connected


class TestUdpLink:
    def test_init(self):
        l = UdpLink("192.168.1.10", 14550)
        assert l.name == "udp:192.168.1.10:14550"
        assert l.priority == 20


class TestTransportManager:
    def test_empty(self):
        m = TransportManager()
        assert m.link_count == 0
        assert m.active_link is None

    def test_add_links_sorted(self):
        m = TransportManager()
        m.add_link(SerialLink("/dev/x", priority=10))
        m.add_link(TcpLink("x", 1, priority=0))
        m.add_link(UdpLink("x", 2, priority=20))
        assert m.link_count == 3
        # TCP should be first (priority 0)
        assert m._links[0].name.startswith("tcp:")

    def test_status_no_links(self):
        m = TransportManager()
        s = m.status()
        assert s["active"] is None
        assert s["links"] == []

    def test_status_with_links(self):
        m = TransportManager()
        m.add_link(TcpLink("1.2.3.4", 9000))
        s = m.status()
        assert len(s["links"]) == 1
        assert s["links"][0]["name"] == "tcp:1.2.3.4:9000"
        assert not s["links"][0]["connected"]

    def test_send_without_connection(self):
        async def _run():
            m = TransportManager()
            result = await m.send(b"test")
            assert not result

        asyncio.run(_run())

    def test_recv_without_connection(self):
        async def _run():
            m = TransportManager()
            data = await m.recv()
            assert data == b""

        asyncio.run(_run())

    def test_connect_no_links_fails(self):
        async def _run():
            m = TransportManager()
            assert not await m.connect()

        asyncio.run(_run())

    def test_disconnect_noop(self):
        async def _run():
            m = TransportManager()
            await m.disconnect()  # should not error

        asyncio.run(_run())


class TestLinkHealth:
    def test_healthy_when_no_data(self):
        l = TcpLink("x", 1)
        l.connected = True
        l.last_recv_time = 0  # no data yet
        assert l.healthy

    def test_unhealthy_stale(self):
        import time

        l = TcpLink("x", 1)
        l.connected = True
        l.last_recv_time = time.time() - LINK_HEALTH_TIMEOUT_S - 1
        assert not l.healthy

    def test_unhealthy_disconnected(self):
        l = TcpLink("x", 1)
        l.connected = False
        assert not l.healthy
