# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for metrics.py — Prometheus-style observability for phanes-brain.

Tests cover:
  - Counter increment, label cardinality, value reads
  - Gauge set/inc/dec
  - Histogram bucket correctness and cumulative semantics
  - render_text() output structure (OpenMetrics format)
  - /metrics endpoint integration (api.py wiring)
  - Fleet metric wiring (fleet_size, conn_active)
  - No dependency on prometheus_client
"""

from __future__ import annotations

import re

import pytest

from metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    M,
    OPENMETRICS_CONTENT_TYPE,
    DEFAULT_LATENCY_BUCKETS_MS,
    HISTOGRAM_INF_BUCKET,
)

# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------


class TestCounter:
    def test_counter_starts_at_zero(self) -> None:
        c = Counter("test_zero", "")
        assert c.labels().value == 0.0

    def test_counter_increment_default(self) -> None:
        c = Counter("test_inc_default", "")
        c.inc()
        c.inc()
        assert c.labels().value == 2.0

    def test_counter_increment_amount(self) -> None:
        c = Counter("test_inc_amount", "")
        c.labels().inc(5.0)
        c.labels().inc(3.5)
        assert c.labels().value == 8.5

    def test_counter_negative_raises(self) -> None:
        c = Counter("test_neg", "")
        with pytest.raises(ValueError):
            c.labels().inc(-1.0)

    def test_counter_label_cardinality(self) -> None:
        c = Counter("test_labels", "", label_names=("robot_id",))
        c.labels(robot_id="bot_1").inc(3.0)
        c.labels(robot_id="bot_2").inc(7.0)
        # Each label set is independent.
        assert c.labels(robot_id="bot_1").value == 3.0
        assert c.labels(robot_id="bot_2").value == 7.0

    def test_counter_multi_label_order_invariant(self) -> None:
        c = Counter("test_multi", "", label_names=("a", "b"))
        # Same labels, different order in kwargs — should map to same child.
        c.labels(a="x", b="y").inc(1.0)
        c.labels(b="y", a="x").inc(1.0)
        assert c.labels(a="x", b="y").value == 2.0


# ---------------------------------------------------------------------------
# Gauge tests
# ---------------------------------------------------------------------------


class TestGauge:
    def test_gauge_set(self) -> None:
        g = Gauge("test_gauge_set", "")
        g.set(42.0)
        assert g.value == 42.0

    def test_gauge_inc_dec(self) -> None:
        g = Gauge("test_gauge_incdec", "")
        g.inc(3.0)
        g.dec(1.0)
        assert g.value == 2.0

    def test_gauge_can_go_negative(self) -> None:
        g = Gauge("test_gauge_neg", "")
        g.dec(5.0)
        assert g.value == -5.0

    def test_gauge_labels_independent(self) -> None:
        g = Gauge("test_gauge_labels", "", label_names=("type",))
        g.labels(type="a").set(10.0)
        g.labels(type="b").set(20.0)
        assert g.labels(type="a").value == 10.0
        assert g.labels(type="b").value == 20.0


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_histogram_bucket_assignment(self) -> None:
        h = Histogram("test_hist_buckets", "", buckets=(10.0, 50.0, 100.0))
        h.observe(5.0)  # → bucket 10
        h.observe(20.0)  # → bucket 50
        h.observe(80.0)  # → bucket 100
        h.observe(200.0)  # → +Inf
        buckets, total_sum, total_count = h.labels().snapshot()
        # Cumulative: bucket[10] = 1, bucket[50] = 2, bucket[100] = 3, +Inf = 4
        cum_dict = {le: cnt for le, cnt in buckets}
        assert cum_dict[10.0] == 1.0
        assert cum_dict[50.0] == 2.0
        assert cum_dict[100.0] == 3.0
        assert cum_dict[float("inf")] == 4.0
        assert total_count == 4.0
        assert total_sum == pytest.approx(5.0 + 20.0 + 80.0 + 200.0)

    def test_histogram_plus_inf_bucket_present(self) -> None:
        h = Histogram("test_hist_inf", "", buckets=(1.0, 5.0))
        h.observe(0.5)
        buckets, _, _ = h.labels().snapshot()
        upper_bounds = [le for le, _ in buckets]
        assert float("inf") in upper_bounds

    def test_histogram_cumulative_monotone(self) -> None:
        h = Histogram("test_hist_mono", "", buckets=(1.0, 5.0, 10.0))
        for v in (0.1, 3.0, 7.0, 15.0):
            h.observe(v)
        buckets, _, _ = h.labels().snapshot()
        counts = [cnt for _, cnt in buckets]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], "Cumulative buckets must be monotone"

    def test_histogram_sum_and_count(self) -> None:
        h = Histogram("test_hist_sum", "", buckets=(100.0,))
        h.observe(10.0)
        h.observe(20.0)
        _, total_sum, total_count = h.labels().snapshot()
        assert total_count == 2.0
        assert total_sum == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# OpenMetrics render_text tests
# ---------------------------------------------------------------------------


class TestRenderText:
    def _make_registry(self) -> MetricsRegistry:
        reg = MetricsRegistry()
        c = Counter("my_counter", "A test counter", label_names=("env",))
        g = Gauge("my_gauge", "A test gauge")
        h = Histogram("my_hist", "A test histogram", buckets=(5.0, 10.0))
        c.labels(env="prod").inc(3.0)
        g.set(7.5)
        h.observe(4.0)
        h.observe(8.0)
        reg.register(c)
        reg.register(g)
        reg.register(h)
        return reg

    def test_render_contains_help_and_type(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        assert "# HELP my_counter" in text
        assert "# TYPE my_counter counter" in text
        assert "# HELP my_gauge" in text
        assert "# TYPE my_gauge gauge" in text
        assert "# HELP my_hist" in text
        assert "# TYPE my_hist histogram" in text

    def test_render_counter_value(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        # Should contain the labelled counter line.
        assert 'my_counter{env="prod"} 3.0' in text

    def test_render_gauge_value(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        assert "my_gauge 7.5" in text

    def test_render_histogram_buckets(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        # Cumulative: le=5 → 1 sample (4.0 ≤ 5), le=10 → 2 samples, +Inf → 2
        assert f'my_hist_bucket{{le="5.0"}} 1.0' in text
        assert f'my_hist_bucket{{le="10.0"}} 2.0' in text
        assert f'my_hist_bucket{{le="+Inf"}} 2.0' in text

    def test_render_histogram_sum_count(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        assert "my_hist_sum" in text
        assert "my_hist_count" in text

    def test_render_parseable_by_regex(self) -> None:
        """Verify the output is structurally parseable as OpenMetrics text."""
        reg = self._make_registry()
        text = reg.render_text()
        # Every non-comment, non-blank line must match: metric_name{labels} value
        line_pat = re.compile(
            r"^[a-z_][a-z0-9_]*(\{[^}]*\})?\s+[\d.e+\-inf]+$",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert line_pat.match(stripped), f"Unparseable metrics line: {stripped!r}"

    def test_global_M_render_text(self) -> None:
        """M.render_text() runs without error and contains expected metric names."""
        text = M.render_text()
        assert "phanes_brain_pkt_rx_total" in text
        assert "phanes_brain_conn_active" in text
        assert "phanes_brain_fleet_size" in text
        assert "phanes_brain_e2e_latency_ms" in text
        assert "phanes_brain_ota_pushes_total" in text

    def test_render_ends_with_newline(self) -> None:
        reg = self._make_registry()
        text = reg.render_text()
        assert text.endswith("\n")


# ---------------------------------------------------------------------------
# /metrics endpoint integration (api.py wiring)
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Verify the /metrics route wires up correctly.

    We test _metrics_response directly (writing to a mock StreamWriter) so we
    don't need to spin up a real TCP server or patch asyncio internals.
    """

    def _make_writer(self) -> "MockWriter":
        return MockWriter()

    def test_metrics_route_returns_openmetrics_content_type(self) -> None:
        from api import _metrics_response

        w = MockWriter()
        _metrics_response(w)
        output = b"".join(w.chunks).decode()
        assert OPENMETRICS_CONTENT_TYPE in output

    def test_metrics_route_status_200(self) -> None:
        from api import _metrics_response

        w = MockWriter()
        _metrics_response(w)
        first_line = b"".join(w.chunks).split(b"\r\n")[0].decode()
        assert "200" in first_line

    def test_metrics_route_body_contains_metric_names(self) -> None:
        from api import _metrics_response

        w = MockWriter()
        _metrics_response(w)
        raw = b"".join(w.chunks).decode()
        # Split on the blank line separating headers from body.
        header_part, _, body = raw.partition("\r\n\r\n")
        assert "phanes_brain_conn_active" in body
        assert "phanes_brain_fleet_size" in body

    def test_http_bytes_counter_increments_on_response(self) -> None:
        from api import _response
        import metrics

        # Record current value before the call.
        before = metrics.M.http_bytes_total.labels(direction="out").value
        w = MockWriter()
        _response(w, 200, {"ok": True})
        after = metrics.M.http_bytes_total.labels(direction="out").value
        assert after > before, "http_bytes_total should have been incremented"

    def test_http_bytes_counter_not_incremented_when_suppressed(self) -> None:
        from api import _response
        import metrics

        before = metrics.M.http_bytes_total.labels(direction="out").value
        w = MockWriter()
        _response(w, 200, {"ok": True}, _count_bytes=False)
        after = metrics.M.http_bytes_total.labels(direction="out").value
        assert after == before, "bytes counter must NOT increment when _count_bytes=False"


# ---------------------------------------------------------------------------
# Fleet metric wiring tests
# ---------------------------------------------------------------------------


class TestFleetMetrics:
    def test_fleet_size_updates_on_register(self) -> None:
        """fleet_size gauge equals len(fm._robots) after register."""
        import metrics
        from fleet import FleetManager

        fm = FleetManager()
        fm.register(robot_id="_test_reg_001_fs", writer=None)
        # After one registration the gauge must match the fleet's own count.
        assert metrics.M.fleet_size.value == float(fm.count)

    def test_fleet_size_updates_on_unregister(self) -> None:
        """fleet_size gauge decreases and equals fleet count after unregister."""
        import metrics
        from fleet import FleetManager

        fm = FleetManager()
        fm.register(robot_id="_test_unreg_001_fs", writer=None)
        fm.unregister("_test_unreg_001_fs")
        assert metrics.M.fleet_size.value == float(fm.count)

    def test_conn_active_increments_on_online_register(self) -> None:
        """conn_active gauge increases by 1 when an online robot registers."""
        import metrics
        from fleet import FleetManager

        class _W:
            pass  # mock writer (non-None → robot is online)

        fm = FleetManager()
        before = metrics.M.conn_active.value
        fm.register(robot_id="_test_conn_001_ca", writer=_W())
        after = metrics.M.conn_active.value
        assert after == before + 1.0

    def test_conn_active_decrements_on_mark_disconnected(self) -> None:
        """conn_active gauge decreases by 1 when mark_disconnected is called."""
        import metrics
        from fleet import FleetManager

        class _W:
            pass

        fm = FleetManager()
        fm.register(robot_id="_test_conn_002_ca", writer=_W())
        mid = metrics.M.conn_active.value
        fm.mark_disconnected("_test_conn_002_ca")
        after = metrics.M.conn_active.value
        assert after == mid - 1.0

    def test_ota_error_counter_increments_on_failed_push(self) -> None:
        """ota_pushes_total{status=error} increments when push_firmware_to_robot fails."""
        import asyncio
        import metrics
        from fleet import FleetManager
        from fleet_ota import push_firmware_to_robot, DISPATCH_STATUS_ERROR

        async def _run() -> None:
            fm = FleetManager()

            class _W:
                def get_extra_info(self, key: str) -> object:
                    return ("127.0.0.1", 9999) if key == "peername" else None

            fm.register(robot_id="_test_ota_err_001", writer=_W())
            robot = fm.get("_test_ota_err_001")
            assert robot is not None

            before = metrics.M.ota_pushes_total.labels(status=DISPATCH_STATUS_ERROR).value

            async def _refuse(host: str, port: int):  # type: ignore[misc]
                raise ConnectionRefusedError("simulated connect failure")

            await push_firmware_to_robot(
                robot=robot,
                image=b"\x00" * 16,
                sig=b"",
                platform="qemu",
                fw_version=1,
                open_conn_fn=_refuse,
            )

            after = metrics.M.ota_pushes_total.labels(status=DISPATCH_STATUS_ERROR).value
            assert (
                after == before + 1.0
            ), "ota_pushes_total{status=error} should increment on failed push"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Predefined metric names / labels sanity
# ---------------------------------------------------------------------------


class TestPreDefinedMetrics:
    def test_all_predefined_metrics_registered(self) -> None:
        text = M.render_text()
        expected = [
            "phanes_brain_pkt_rx_total",
            "phanes_brain_pkt_tx_total",
            "phanes_brain_pkt_bytes_total",
            "phanes_brain_http_bytes_total",
            "phanes_brain_conn_active",
            "phanes_brain_conn_accepted_total",
            "phanes_brain_conn_dropped_total",
            "phanes_brain_e2e_latency_ms",
            "phanes_brain_fleet_size",
            "phanes_brain_ota_pushes_total",
        ]
        for name in expected:
            assert name in text, f"Expected metric '{name}' not found in render_text()"

    def test_default_latency_buckets_sorted(self) -> None:
        buckets = DEFAULT_LATENCY_BUCKETS_MS
        assert list(buckets) == sorted(buckets)

    def test_histogram_inf_bucket_constant(self) -> None:
        assert HISTOGRAM_INF_BUCKET == "+Inf"

    def test_e2e_latency_observe(self) -> None:
        M.e2e_latency_ms.observe(42.0)
        _, s, c = M.e2e_latency_ms.labels().snapshot()
        assert c >= 1.0
        assert s >= 42.0


# ---------------------------------------------------------------------------
# Helper: mock asyncio.StreamWriter
# ---------------------------------------------------------------------------


class MockWriter:
    """Minimal stand-in for asyncio.StreamWriter that captures written bytes."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)
