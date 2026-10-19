# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for E02 — multi-link transport client (`transport.py`).

Covers:
  * Adapter priority sort
  * connect_all returns active link (primary)
  * Failover when primary send fails
  * Failback after LINK_PROBE_INTERVAL_S
  * Health flags (consec_failures → down, RX timeout → down)
"""

from __future__ import annotations

import asyncio
import time

from transport import (
    LinkAdapter,
    LinkStats,
    MultiLinkClient,
    WiFiAdapter,
    LoRaAdapter,
    RFAdapter,
    TRANSPORT_FAILOVER_TIMEOUT_S,
    LINK_PROBE_INTERVAL_S,
    TRANSPORT_MAX_CONSEC_FAILURES,
    LINK_QUALITY_DOWN,
    LINK_QUALITY_GOOD,
    LINK_QUALITY_UNKNOWN,
    WIFI_DEFAULT_PRIORITY,
    LORA_DEFAULT_PRIORITY,
    RF_DEFAULT_PRIORITY,
)

# ---------------------------------------------------------------------------
# Mock adapter: deterministic, used throughout
# ---------------------------------------------------------------------------


class MockAdapter(LinkAdapter):
    def __init__(self, name: str, priority: int, up: bool = True, send_ok: bool = True):
        super().__init__(name=name, priority=priority)
        self._up = up
        self._send_ok = send_ok
        self._rx_queue: list[bytes] = []

    async def connect(self) -> bool:
        return self._up

    async def disconnect(self) -> None:
        self._up = False

    async def send(self, data: bytes) -> bool:
        if not self._up or not self._send_ok:
            self.mark_send_fail()
            return False
        self.mark_send_ok(len(data))
        return True

    async def recv(self, max_bytes: int = 4096) -> bytes:
        if not self._rx_queue:
            return b""
        data = self._rx_queue.pop(0)
        self.mark_rx(len(data))
        return data

    # Test helpers
    def set_up(self, up: bool) -> None:
        self._up = up

    def set_send_ok(self, ok: bool) -> None:
        self._send_ok = ok

    def push_rx(self, data: bytes) -> None:
        self._rx_queue.append(data)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_positive_timeouts(self):
        assert TRANSPORT_FAILOVER_TIMEOUT_S > 0
        assert LINK_PROBE_INTERVAL_S > 0
        assert TRANSPORT_MAX_CONSEC_FAILURES >= 1

    def test_priority_order_matches_kernel(self):
        assert WIFI_DEFAULT_PRIORITY < LORA_DEFAULT_PRIORITY
        assert LORA_DEFAULT_PRIORITY < RF_DEFAULT_PRIORITY

    def test_quality_scale(self):
        assert 0 <= LINK_QUALITY_DOWN <= 255
        assert 0 <= LINK_QUALITY_GOOD <= 255
        assert 0 <= LINK_QUALITY_UNKNOWN <= 255


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_wifi_init(self):
        a = WiFiAdapter("10.0.0.42", 9000)
        assert "wifi" in a.name
        assert a.priority == WIFI_DEFAULT_PRIORITY
        assert not a.is_up()

    def test_lora_init(self):
        a = LoRaAdapter("/dev/ttyUSB0")
        assert "lora" in a.name
        assert a.priority == LORA_DEFAULT_PRIORITY
        assert not a.is_up()

    def test_rf_init(self):
        a = RFAdapter()
        assert "rf" in a.name
        assert a.priority == RF_DEFAULT_PRIORITY
        assert not a.is_up()

    def test_link_quality_down_when_not_up(self):
        a = WiFiAdapter("x", 1)
        assert a.link_quality() == LINK_QUALITY_DOWN

    def test_stats_defaults(self):
        a = WiFiAdapter("x", 1)
        assert a.stats.sent_bytes == 0
        assert a.stats.recv_bytes == 0
        assert a.stats.consec_failures == 0


# ---------------------------------------------------------------------------
# MultiLinkClient — topology
# ---------------------------------------------------------------------------


class TestTopology:
    def test_empty_client(self):
        c = MultiLinkClient()
        assert c.link_count == 0
        assert c.active_link is None

    def test_add_sorts_by_priority(self):
        c = MultiLinkClient()
        c.add_link(MockAdapter("rf", priority=20))
        c.add_link(MockAdapter("wifi", priority=0))
        c.add_link(MockAdapter("lora", priority=10))
        assert c.link_count == 3
        assert c._links[0].name == "wifi"
        assert c._links[1].name == "lora"
        assert c._links[2].name == "rf"


# ---------------------------------------------------------------------------
# connect_all / send / recv happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_connect_all_picks_primary(self):
        async def _run():
            c = MultiLinkClient()
            c.add_link(MockAdapter("wifi", priority=0, up=True))
            c.add_link(MockAdapter("lora", priority=10, up=True))
            assert await c.connect_all()
            assert c.active_link is not None
            assert c.active_link.name == "wifi"

        asyncio.run(_run())

    def test_connect_all_skips_down_primary(self):
        async def _run():
            c = MultiLinkClient()
            c.add_link(MockAdapter("wifi", priority=0, up=False))
            c.add_link(MockAdapter("lora", priority=10, up=True))
            assert await c.connect_all()
            assert c.active_link.name == "lora"

        asyncio.run(_run())

    def test_connect_all_no_links(self):
        async def _run():
            c = MultiLinkClient()
            assert not await c.connect_all()

        asyncio.run(_run())

    def test_send_via_primary(self):
        async def _run():
            c = MultiLinkClient()
            c.add_link(MockAdapter("wifi", priority=0))
            c.add_link(MockAdapter("lora", priority=10))
            await c.connect_all()
            assert await c.send(b"hello")
            assert c.active_link.name == "wifi"
            assert c._links[0].stats.sent_bytes == 5

        asyncio.run(_run())

    def test_recv_from_primary(self):
        async def _run():
            c = MultiLinkClient()
            wifi = MockAdapter("wifi", priority=0)
            wifi.push_rx(b"packet")
            c.add_link(wifi)
            await c.connect_all()
            data = await c.recv()
            assert data == b"packet"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------


class TestFailover:
    def test_failover_when_primary_send_fails(self):
        async def _run():
            c = MultiLinkClient()
            wifi = MockAdapter("wifi", priority=0, send_ok=False)
            lora = MockAdapter("lora", priority=10, send_ok=True)
            c.add_link(wifi)
            c.add_link(lora)
            await c.connect_all()
            assert c.active_link.name == "wifi"
            # Send should fall back to lora.
            assert await c.send(b"x")
            assert c.active_link.name == "lora"

        asyncio.run(_run())

    def test_failover_when_primary_goes_down_mid_flight(self):
        async def _run():
            c = MultiLinkClient()
            wifi = MockAdapter("wifi", priority=0)
            lora = MockAdapter("lora", priority=10)
            c.add_link(wifi)
            c.add_link(lora)
            await c.connect_all()
            assert c.active_link.name == "wifi"
            wifi.set_up(False)
            assert await c.send(b"x")
            assert c.active_link.name == "lora"

        asyncio.run(_run())

    def test_failover_fails_when_all_down(self):
        async def _run():
            c = MultiLinkClient()
            c.add_link(MockAdapter("wifi", priority=0, up=False))
            c.add_link(MockAdapter("lora", priority=10, up=False))
            await c.connect_all()
            assert not await c.send(b"x")

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Health flags
# ---------------------------------------------------------------------------


class TestHealth:
    def test_link_down_after_consec_failures(self):
        a = MockAdapter("wifi", priority=0)
        a._up = True
        c = MultiLinkClient()
        c.add_link(a)
        # No failures yet.
        assert not c._link_down(a)
        # Spam failures up to the threshold.
        for _ in range(TRANSPORT_MAX_CONSEC_FAILURES):
            a.mark_send_fail()
        assert c._link_down(a)

    def test_link_down_after_rx_timeout(self):
        a = MockAdapter("wifi", priority=0)
        a._up = True
        a.stats.last_recv_ts = time.time() - (TRANSPORT_FAILOVER_TIMEOUT_S + 1)
        c = MultiLinkClient()
        c.add_link(a)
        assert c._link_down(a)

    def test_link_healthy_before_first_rx(self):
        a = MockAdapter("wifi", priority=0)
        a._up = True
        a.stats.last_recv_ts = 0.0  # never received
        c = MultiLinkClient()
        c.add_link(a)
        assert not c._link_down(a)


# ---------------------------------------------------------------------------
# Failback (primary recovers → return to it)
# ---------------------------------------------------------------------------


class TestFailback:
    def test_failback_after_probe_interval(self):
        async def _run():
            c = MultiLinkClient()
            wifi = MockAdapter("wifi", priority=0, up=False)
            lora = MockAdapter("lora", priority=10, up=True)
            c.add_link(wifi)
            c.add_link(lora)
            await c.connect_all()
            assert c.active_link.name == "lora"
            # Primary comes back and we age past the probe interval.
            wifi.set_up(True)
            wifi.stats.last_probe_ts = time.time() - (LINK_PROBE_INTERVAL_S + 1)
            await c._maybe_failback()
            assert c.active_link.name == "wifi"

        asyncio.run(_run())

    def test_no_failback_before_probe_interval(self):
        async def _run():
            c = MultiLinkClient()
            wifi = MockAdapter("wifi", priority=0, up=False)
            lora = MockAdapter("lora", priority=10, up=True)
            c.add_link(wifi)
            c.add_link(lora)
            await c.connect_all()
            wifi.set_up(True)
            wifi.stats.last_probe_ts = time.time()  # just probed
            await c._maybe_failback()
            # Still on lora because we haven't waited long enough.
            assert c.active_link.name == "lora"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_includes_all_links(self):
        async def _run():
            c = MultiLinkClient()
            c.add_link(MockAdapter("wifi", priority=0))
            c.add_link(MockAdapter("lora", priority=10))
            await c.connect_all()
            s = c.status()
            assert s["active"] == "wifi"
            assert len(s["links"]) == 2
            assert {l["name"] for l in s["links"]} == {"wifi", "lora"}
            for l in s["links"]:
                assert "quality" in l
                assert "up" in l
                assert "consec_failures" in l

        asyncio.run(_run())

    def test_status_empty_client(self):
        c = MultiLinkClient()
        s = c.status()
        assert s["active"] is None
        assert s["links"] == []
