# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Prometheus-compatible metrics for AzOSRobotBrain.

Implements Counter, Gauge, and Histogram classes with label support and
OpenMetrics text-format rendering.  No dependency on prometheus_client —
the implementation is self-contained (~200 lines).

All increment/observe/set operations are synchronous.  Internal label-dict
creation is guarded by a threading.Lock; CPython's GIL provides enough
atomicity for the hot float/int updates themselves.

Usage:
    from metrics import M

    # Count every received packet, labelled by type and robot_id.
    M.pkt_rx_total.labels(type="sensor", robot_id="bot_1").inc()

    # Expose all metrics as an OpenMetrics text response.
    text = M.render_text()
"""

from __future__ import annotations

import threading
import time
from typing import Optional

__all__ = ["Counter", "Gauge", "Histogram", "MetricsRegistry", "M"]

# ── Named constants ────────────────────────────────────────────────────────────

# Default histogram buckets (milliseconds) for E2E latency tracking.
# Chosen to give useful resolution from sub-millisecond to one-second spans.
DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
)

# Prometheus / OpenMetrics content-type header value.
OPENMETRICS_CONTENT_TYPE: str = "text/plain; version=0.0.4"

# Sentinel bucket label used for the cumulative +Inf bucket.
HISTOGRAM_INF_BUCKET: str = "+Inf"

# Separator used between label names in the internal label key.
_LABEL_KEY_SEPARATOR: str = "\x00"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _labels_to_key(labels: dict[str, str]) -> str:
    """Convert a label dict to a stable, hashable string key.

    Sort by name so label order doesn't create phantom distinct series.
    """
    return _LABEL_KEY_SEPARATOR.join(f"{k}={v}" for k, v in sorted(labels.items()))


def _render_labels(labels: dict[str, str]) -> str:
    """Render a label dict to the ``{k="v",...}`` Prometheus text syntax."""
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


# ── Child (label-instantiated) metric objects ──────────────────────────────────


class _CounterChild:
    """A single Counter series (one label set)."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError(f"Counter.inc amount must be >= 0, got {amount}")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _GaugeChild:
    """A single Gauge series (one label set)."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _HistogramChild:
    """A single Histogram series (one label set)."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        # bucket_counts[i] = raw count in the i-th bucket (upper bound = buckets[i]).
        self._buckets: tuple[float, ...] = buckets
        self._bucket_counts: list[float] = [0.0] * len(buckets)
        self._inf_count: float = 0.0  # samples > last bucket
        self._sum: float = 0.0
        self._count: float = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1.0
            placed = False
            for i, upper in enumerate(self._buckets):
                if value <= upper:
                    self._bucket_counts[i] += 1.0
                    placed = True
                    break
            if not placed:
                self._inf_count += 1.0

    def snapshot(self) -> tuple[list[tuple[float, float]], float, float]:
        """Return (cumulative_buckets, sum, count).

        Cumulative: each bucket includes all observations <= its upper bound.
        """
        with self._lock:
            cumulative: list[tuple[float, float]] = []
            running: float = 0.0
            for i, upper in enumerate(self._buckets):
                running += self._bucket_counts[i]
                cumulative.append((upper, running))
            # +Inf bucket = total count
            total = running + self._inf_count
            cumulative.append((float("inf"), total))
            return cumulative, self._sum, self._count


# ── Parent metric types (hold the label registry) ─────────────────────────────


class Counter:
    """A monotonically increasing counter, optionally with labels."""

    KIND: str = "counter"

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self.label_names: tuple[str, ...] = label_names
        self._children: dict[str, _CounterChild] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs: str) -> _CounterChild:
        key = _labels_to_key(kwargs)
        with self._lock:
            if key not in self._children:
                self._children[key] = _CounterChild()
            return self._children[key]

    def inc(self, amount: float = 1.0) -> None:
        """Increment the no-label series."""
        self.labels().inc(amount)

    def render(self) -> str:
        lines: list[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.KIND}",
        ]
        with self._lock:
            snapshot = dict(self._children)
        for key, child in snapshot.items():
            labels = _key_to_labels(key)
            lines.append(f"{self.name}{_render_labels(labels)} {child.value}")
        return "\n".join(lines)


class Gauge:
    """A value that can go up and down, optionally with labels."""

    KIND: str = "gauge"

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self.label_names: tuple[str, ...] = label_names
        self._children: dict[str, _GaugeChild] = {}
        self._lock = threading.Lock()
        # Create a default no-label child eagerly so Gauges always emit.
        self._children[""] = _GaugeChild()

    def labels(self, **kwargs: str) -> _GaugeChild:
        key = _labels_to_key(kwargs)
        with self._lock:
            if key not in self._children:
                self._children[key] = _GaugeChild()
            return self._children[key]

    def set(self, value: float) -> None:
        """Set the no-label series."""
        self._children[""].set(value)

    def inc(self, amount: float = 1.0) -> None:
        self._children[""].inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        self._children[""].dec(amount)

    @property
    def value(self) -> float:
        return self._children[""].value

    def render(self) -> str:
        lines: list[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.KIND}",
        ]
        with self._lock:
            snapshot = dict(self._children)
        for key, child in snapshot.items():
            labels = _key_to_labels(key)
            lines.append(f"{self.name}{_render_labels(labels)} {child.value}")
        return "\n".join(lines)


class Histogram:
    """A sampled distribution with configurable buckets, optionally labelled."""

    KIND: str = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS,
        label_names: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.help_text = help_text
        self._buckets: tuple[float, ...] = buckets
        self.label_names: tuple[str, ...] = label_names
        self._children: dict[str, _HistogramChild] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs: str) -> _HistogramChild:
        key = _labels_to_key(kwargs)
        with self._lock:
            if key not in self._children:
                self._children[key] = _HistogramChild(self._buckets)
            return self._children[key]

    def observe(self, value: float) -> None:
        """Record an observation in the no-label series."""
        self.labels().observe(value)

    def render(self) -> str:
        lines: list[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.KIND}",
        ]
        with self._lock:
            snapshot = dict(self._children)
        for key, child in snapshot.items():
            labels = _key_to_labels(key)
            buckets, total_sum, total_count = child.snapshot()
            for upper, cum_count in buckets:
                le_label = HISTOGRAM_INF_BUCKET if upper == float("inf") else str(upper)
                bucket_labels = dict(_key_to_labels(key))
                bucket_labels["le"] = le_label
                lines.append(f"{self.name}_bucket{_render_labels(bucket_labels)} {cum_count}")
            lines.append(f"{self.name}_sum{_render_labels(labels)} {total_sum}")
            lines.append(f"{self.name}_count{_render_labels(labels)} {total_count}")
        return "\n".join(lines)


def _key_to_labels(key: str) -> dict[str, str]:
    """Inverse of _labels_to_key — reconstructs the label dict."""
    if not key:
        return {}
    result: dict[str, str] = {}
    for pair in key.split(_LABEL_KEY_SEPARATOR):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


# ── Registry ───────────────────────────────────────────────────────────────────


class MetricsRegistry:
    """Holds all registered metric families and renders them."""

    def __init__(self) -> None:
        self._metrics: list[Counter | Gauge | Histogram] = []

    def register(self, metric: Counter | Gauge | Histogram) -> Counter | Gauge | Histogram:
        self._metrics.append(metric)
        return metric

    def render_text(self) -> str:
        """Render all metrics in OpenMetrics text format."""
        parts: list[str] = []
        ts = time.time()
        parts.append(f"# phanes-brain metrics rendered at {ts:.3f}\n")
        for metric in self._metrics:
            rendered = metric.render()
            if rendered:
                parts.append(rendered)
        # OpenMetrics text format ends with a trailing newline.
        return "\n".join(parts) + "\n"


# ── Pre-defined metrics (module-level singletons) ─────────────────────────────


class _Metrics:
    """Container for all pre-defined brain metrics.  Import as `from metrics import M`."""

    _registry: MetricsRegistry

    def __init__(self) -> None:
        self._registry = MetricsRegistry()
        r = self._registry.register

        # Packet counters — hot-path, labelled by packet type and robot_id.
        self.pkt_rx_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_pkt_rx_total",
                "Total packets received from robots, by type and robot_id",
                label_names=("type", "robot_id"),
            )
        )
        self.pkt_tx_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_pkt_tx_total",
                "Total packets sent to robots, by type and robot_id",
                label_names=("type", "robot_id"),
            )
        )

        # RFC-0039 step 1: server.py's `_dispatch` used to be an if/elif chain
        # with no final `else` — an inbound type it didn't act on (OTA_ACK,
        # SENSOR_COMPACT, or anything genuinely unknown) was counted as
        # received and silently dropped, indistinguishable from a type it
        # handled. These two counters make that distinction visible from
        # outside the process; label is the packet type as "0xNN" so a reader
        # of /metrics doesn't have to decode a bitmap.
        self.pkt_dispatch_handled_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_pkt_dispatch_handled_total",
                "Inbound packets the dispatcher recognised and acted on, by type",
                label_names=("type",),
            )
        )
        self.pkt_dispatch_ignored_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_pkt_dispatch_ignored_total",
                "Inbound packets with no dispatch handler (RFC-0039 step 1), by type",
                label_names=("type",),
            )
        )

        # Byte counters — labelled by direction and robot_id.
        self.pkt_bytes_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_pkt_bytes_total",
                "Total bytes transferred on the binary protocol path, by direction and robot_id",
                label_names=("direction", "robot_id"),
            )
        )

        # HTTP response bytes (api.py _response path — no robot_id label here).
        self.http_bytes_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_http_bytes_total",
                "Total bytes written to HTTP clients via _response",
                label_names=("direction",),
            )
        )

        # Connection gauges / counters.
        self.conn_active: Gauge = r(
            Gauge(  # type: ignore[assignment]
                "phanes_brain_conn_active",
                "Current number of open kernel TCP connections",
            )
        )
        self.conn_accepted_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_conn_accepted_total",
                "Total TCP connections accepted from robots",
            )
        )
        self.conn_dropped_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_conn_dropped_total",
                "Total connections dropped, labelled by reason",
                label_names=("reason",),
            )
        )

        # End-to-end latency histogram (milliseconds).
        self.e2e_latency_ms: Histogram = r(
            Histogram(  # type: ignore[assignment]
                "phanes_brain_e2e_latency_ms",
                "End-to-end brain processing latency in milliseconds",
                buckets=DEFAULT_LATENCY_BUCKETS_MS,
            )
        )

        # Fleet size gauge.
        self.fleet_size: Gauge = r(
            Gauge(  # type: ignore[assignment]
                "phanes_brain_fleet_size",
                "Current number of robots registered in the fleet",
            )
        )

        # OTA push counter.
        self.ota_pushes_total: Counter = r(
            Counter(  # type: ignore[assignment]
                "phanes_brain_ota_pushes_total",
                "Total OTA push attempts, labelled by status (ok/error)",
                label_names=("status",),
            )
        )

    def render_text(self) -> str:
        """Delegate to the internal registry."""
        return self._registry.render_text()


#: Module-level singleton.  All brain modules import this.
M: _Metrics = _Metrics()
