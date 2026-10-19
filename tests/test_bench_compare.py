# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""tests/test_bench_compare.py — Unit tests for tools/bench_compare.py.

Tests cover:
  - Regression detection per direction (good-when-smaller, good-when-larger)
  - Threshold edge cases (exactly 5%, 4.9%, 5.1%)
  - Waiver parsing (well-formed, malformed, empty)
  - Null handling on both sides
  - Unknown metric (defaults to good-smaller with "unknown" flag)
  - Exit codes via main()
  - _meta / meta key exclusion
  - Empty jitter_ns dict treated as no leaves
  - Dot-path walker on nested wcet structure
  - Markdown report section headers present
  - Waiver path normalization (whitespace tolerance)
  - Improvement detection for good-when-larger metric
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

# Add tools directory to sys.path so we can import bench_compare directly.
import sys

_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from bench_compare import (  # noqa: E402
    _compare_value,
    _direction,
    _parse_waiver_set,
    _walk_leaves,
    compare,
    main,
    render_report,
    NON_METRIC_TOP_LEVEL_KEYS,
    REGRESSION_THRESHOLD,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid result JSON matching the A5 schema."""
    base: dict[str, Any] = {
        "meta": {
            "sha": "aabbccddee11",
            "harness_version": "1.0",
        },
        "rtt_ms": {
            "p50": 102.5,
            "p95": 103.8,
            "p99": 104.0,
            "stddev": 0.9,
            "n_samples": 6,
        },
        "throughput": {
            "steady_msgs_per_s": 0.23,
            "burst_peak_msgs_per_s": 400000.0,
        },
        "boot_ms": 1000.0,
        "wcet_us": {
            "timer_isr": {
                "min": 1000,
                "max": 1000,
                "avg": 1000,
                "p99": 1000,
                "violations": 0,
            }
        },
        "jitter_ns": {},
        "footprint": {
            "text_bytes": 300000,
            "rodata_bytes": 80000,
            "data_bytes": 700000,
            "bss_bytes": 2000000,
            "total_bytes": 3080000,
        },
    }
    base.update(overrides)
    return base


def _make_baseline(**overrides: Any) -> dict[str, Any]:
    """Return a baseline JSON (adds _meta block)."""
    bl = _make_result(**overrides)
    bl["_meta"] = {
        "baseline_sha": "5e61db3c9fa7",
        "baseline_date": "2026-05-28",
        "harness_version": "1.0",
    }
    return bl


# ── Waiver parsing ────────────────────────────────────────────────────────────


class TestParseWaiverSet:
    def test_empty_string_returns_empty_set(self) -> None:
        assert _parse_waiver_set("") == set()

    def test_no_marker_returns_empty_set(self) -> None:
        assert _parse_waiver_set("fix: some change\n\nNo waivers here.") == set()

    def test_single_path(self) -> None:
        assert _parse_waiver_set("BENCH-WAIVER: rtt_ms.p99") == {"rtt_ms.p99"}

    def test_multiple_paths_comma_separated(self) -> None:
        result = _parse_waiver_set(
            "feat: optimise loop\n\nBENCH-WAIVER: rtt_ms.p99,wcet_us.timer_isr.max"
        )
        assert result == {"rtt_ms.p99", "wcet_us.timer_isr.max"}

    def test_whitespace_around_paths_is_stripped(self) -> None:
        result = _parse_waiver_set("BENCH-WAIVER: rtt_ms.p99 , wcet_us.timer_isr.max ")
        assert result == {"rtt_ms.p99", "wcet_us.timer_isr.max"}

    def test_marker_in_middle_of_body(self) -> None:
        body = "Some text\nBENCH-WAIVER: footprint.total_bytes\nMore text"
        assert _parse_waiver_set(body) == {"footprint.total_bytes"}

    def test_malformed_no_paths_after_marker(self) -> None:
        # "BENCH-WAIVER:" with nothing after should give empty set (no non-empty entries).
        result = _parse_waiver_set("BENCH-WAIVER: ")
        assert result == set()


# ── Direction table ───────────────────────────────────────────────────────────


class TestDirection:
    def test_boot_ms_is_smaller(self) -> None:
        d, known = _direction("boot_ms")
        assert d == "smaller"
        assert known is True

    def test_rtt_ms_p99_is_smaller(self) -> None:
        d, known = _direction("rtt_ms.p99")
        assert d == "smaller"
        assert known is True

    def test_rtt_ms_n_samples_is_info(self) -> None:
        d, known = _direction("rtt_ms.n_samples")
        assert d == "info"
        assert known is True

    def test_throughput_steady_is_larger(self) -> None:
        d, known = _direction("throughput.steady_msgs_per_s")
        assert d == "larger"
        assert known is True

    def test_wcet_prefix_match(self) -> None:
        d, known = _direction("wcet_us.timer_isr.max")
        assert d == "smaller"
        assert known is True

    def test_unknown_metric_defaults_to_smaller_not_known(self) -> None:
        d, known = _direction("some.unknown.metric.path")
        assert d == "smaller"
        assert known is False


# ── _compare_value ────────────────────────────────────────────────────────────


class TestCompareValue:
    def test_regression_smaller_above_threshold(self) -> None:
        # baseline=1000, result=1060 → 6% increase on good-when-smaller → regression
        row = _compare_value("boot_ms", 1000.0, 1060.0, set())
        assert row.status == "regression"
        assert row.pct_change is not None
        assert row.pct_change > REGRESSION_THRESHOLD

    def test_ok_smaller_below_threshold(self) -> None:
        # baseline=1000, result=1049 → 4.9% → ok
        row = _compare_value("boot_ms", 1000.0, 1049.0, set())
        assert row.status == "ok"

    def test_regression_exactly_at_threshold(self) -> None:
        # baseline=1000, result=1050 → exactly 5% → regression (>= 5%)
        row = _compare_value("boot_ms", 1000.0, 1050.0, set())
        assert row.status == "regression"

    def test_ok_just_below_threshold(self) -> None:
        # baseline=1000, result=1049 → 4.9% → ok
        row = _compare_value("boot_ms", 1000.0, 1049.0, set())
        assert row.status == "ok"

    def test_regression_just_above_threshold(self) -> None:
        # baseline=1000, result=1051 → 5.1% → regression
        row = _compare_value("boot_ms", 1000.0, 1051.0, set())
        assert row.status == "regression"

    def test_improvement_smaller_direction(self) -> None:
        # baseline=1000, result=900 → improved (got smaller)
        row = _compare_value("boot_ms", 1000.0, 900.0, set())
        assert row.status == "improvement"

    def test_regression_larger_direction(self) -> None:
        # throughput regresses when it drops by >= 5%
        row = _compare_value("throughput.steady_msgs_per_s", 100.0, 94.0, set())
        assert row.status == "regression"

    def test_ok_larger_direction(self) -> None:
        # throughput drops by only 4.9% → ok
        row = _compare_value("throughput.steady_msgs_per_s", 100.0, 95.1, set())
        assert row.status == "ok"

    def test_improvement_larger_direction(self) -> None:
        # throughput improves when it grows by >= 5%
        row = _compare_value("throughput.steady_msgs_per_s", 100.0, 110.0, set())
        assert row.status == "improvement"

    def test_baseline_null_result_populated_is_new(self) -> None:
        row = _compare_value("some.new.metric", None, 42.0, set())
        assert row.status == "new"

    def test_baseline_populated_result_null_is_missing(self) -> None:
        row = _compare_value("boot_ms", 1000.0, None, set())
        assert row.status == "missing"

    def test_both_null_is_skip(self) -> None:
        row = _compare_value("rtt_ms.p50", None, None, set())
        assert row.status == "skip"

    def test_info_metric_never_regresses(self) -> None:
        # n_samples goes from 6 to 0 — should be "info", not "regression"
        row = _compare_value("rtt_ms.n_samples", 6.0, 0.0, set())
        assert row.status == "info"

    def test_waiver_marks_regression_as_waived(self) -> None:
        row = _compare_value("boot_ms", 1000.0, 1100.0, {"boot_ms"})
        assert row.status == "regression"
        assert row.waived is True

    def test_unknown_metric_flagged_not_known(self) -> None:
        row = _compare_value("mysterious.deep.metric", 50.0, 60.0, set())
        assert row.is_known is False


# ── compare() and meta key exclusion ─────────────────────────────────────────


class TestCompare:
    def test_self_vs_self_has_no_regressions(self) -> None:
        data = _make_result()
        rows = compare(data, data, set())
        regressions = [r for r in rows if r.status == "regression"]
        assert regressions == []

    def test_meta_keys_excluded(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        rows = compare(result, baseline, set())
        paths = {r.path for r in rows}
        # No path should start with "meta." or "_meta."
        for path in paths:
            assert not path.startswith("meta."), f"meta key leaked: {path}"
            assert not path.startswith("_meta."), f"_meta key leaked: {path}"

    def test_empty_jitter_dict_produces_no_leaves(self) -> None:
        data = _make_result()
        assert data["jitter_ns"] == {}
        rows = compare(data, data, set())
        jitter_paths = [r.path for r in rows if r.path.startswith("jitter_ns")]
        assert jitter_paths == []

    def test_wcet_nested_paths_compared(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        rows = compare(result, baseline, set())
        paths = {r.path for r in rows}
        assert "wcet_us.timer_isr.max" in paths

    def test_regression_detected_in_footprint(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        # Inflate total_bytes by 10% to trigger regression.
        result["footprint"]["total_bytes"] = int(baseline["footprint"]["total_bytes"] * 1.10)
        rows = compare(result, baseline, set())
        reg = [r for r in rows if r.path == "footprint.total_bytes" and r.status == "regression"]
        assert len(reg) == 1

    def test_waiver_suppresses_regression(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        result["footprint"]["total_bytes"] = int(baseline["footprint"]["total_bytes"] * 1.10)
        rows = compare(result, baseline, {"footprint.total_bytes"})
        unwaived = [r for r in rows if r.status == "regression" and not r.waived]
        waived = [r for r in rows if r.path == "footprint.total_bytes" and r.waived]
        assert unwaived == []
        assert len(waived) == 1


# ── render_report ─────────────────────────────────────────────────────────────


class TestRenderReport:
    def test_report_contains_bench_report_marker(self) -> None:
        rows = compare(_make_result(), _make_baseline(), set())
        report = render_report(rows)
        assert "<!-- BENCH-REPORT -->" in report

    def test_report_no_regressions_heading_present(self) -> None:
        rows = compare(_make_result(), _make_baseline(), set())
        report = render_report(rows)
        assert "No unwaived regressions" in report

    def test_report_regression_heading_present_when_regression_exists(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        result["boot_ms"] = baseline["boot_ms"] * 2.0  # 100% regression
        rows = compare(result, baseline, set())
        report = render_report(rows)
        assert "Regressions (unwaived)" in report

    def test_report_improvement_heading_present(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        result["boot_ms"] = baseline["boot_ms"] * 0.5  # 50% improvement
        rows = compare(result, baseline, set())
        report = render_report(rows)
        assert "Improvements" in report

    def test_stable_section_collapsed(self) -> None:
        rows = compare(_make_result(), _make_baseline(), set())
        report = render_report(rows)
        assert "<details>" in report
        assert "Stable metrics" in report


# ── Exit codes via main() ─────────────────────────────────────────────────────


class TestMainExitCodes:
    def _write_json(self, data: dict[str, Any], suffix: str = ".json") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        return path

    def test_exit_0_self_vs_self(self) -> None:
        data = _make_baseline()
        result_path = self._write_json(_make_result())
        baseline_path = self._write_json(data)
        try:
            rc = main(["--result", result_path, "--baseline", baseline_path])
            assert rc == 0
        finally:
            os.unlink(result_path)
            os.unlink(baseline_path)

    def test_exit_1_on_regression(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        result["boot_ms"] = baseline["boot_ms"] * 2.0
        result_path = self._write_json(result)
        baseline_path = self._write_json(baseline)
        try:
            rc = main(["--result", result_path, "--baseline", baseline_path])
            assert rc == 1
        finally:
            os.unlink(result_path)
            os.unlink(baseline_path)

    def test_exit_0_waived_regression(self) -> None:
        result = _make_result()
        baseline = _make_baseline()
        result["boot_ms"] = baseline["boot_ms"] * 2.0
        result_path = self._write_json(result)
        baseline_path = self._write_json(baseline)
        try:
            rc = main(
                [
                    "--result",
                    result_path,
                    "--baseline",
                    baseline_path,
                    "--waiver-text",
                    "BENCH-WAIVER: boot_ms",
                ]
            )
            assert rc == 0
        finally:
            os.unlink(result_path)
            os.unlink(baseline_path)
