# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for tools/bench_dashboard.py.

All tests use tmp_path so they don't touch bench/results/ or bench/dashboard.html.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Make tools/ importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from bench_dashboard import (
    HEADLINE_METRICS,
    compare_color,
    format_qemu_mode,
    iter_leaves,
    load_baseline,
    load_results,
    main,
    render_dashboard,
    sparkline_svg,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

EXAMPLE_JSON_PATH = Path(__file__).parent.parent / "bench" / "results" / "5e61db3c9fa7.json"


def _make_result(
    sha: str,
    timestamp: str,
    boot_ms: Optional[float] = 1000.0,
    rtt_p99: Optional[float] = 100.0,
    total_bytes: int = 3_000_000,
) -> Dict[str, Any]:
    """Return a minimal but schema-valid bench result dict."""
    return {
        "meta": {
            "sha": sha,
            "defconfig": "qemu",
            "timestamp_utc": timestamp,
            "n_runs": 1,
        },
        "rtt_ms": {
            "p50": rtt_p99,
            "p95": rtt_p99,
            "p99": rtt_p99,
            "stddev": 0.5,
            "n_samples": 3,
        },
        "throughput": {
            "steady_msgs_per_s": 1.0,
            "burst_peak_msgs_per_s": 100_000.0,
        },
        "boot_ms": boot_ms,
        "wcet_us": {
            "timer_isr": {
                "min": 500,
                "max": 500,
                "avg": 500,
                "p99": 500,
                "violations": 0,
            }
        },
        "jitter_ns": {},
        "footprint": {
            "text_bytes": 300_000,
            "rodata_bytes": 80_000,
            "data_bytes": 700_000,
            "bss_bytes": total_bytes - 1_080_000,
            "total_bytes": total_bytes,
        },
    }


# ── Test 1: loading the example JSON renders without errors ───────────────────


def test_example_json_renders(tmp_path: Path) -> None:
    """The A5 example JSON must render without exceptions.

    Rendered from a directory holding only the example. The results table lists
    the ten most recent runs, so pointing at ``bench/results`` made the SHA
    assertion depend on how many runs the repo had accumulated rather than on
    the example rendering.
    """
    if not EXAMPLE_JSON_PATH.exists():
        pytest.skip("example JSON not present")

    results = tmp_path / "results"
    results.mkdir()
    (results / EXAMPLE_JSON_PATH.name).write_bytes(EXAMPLE_JSON_PATH.read_bytes())
    out = tmp_path / "dashboard.html"
    ret = main(
        [
            "--results",
            str(results),
            "--baseline",
            str(tmp_path / "nonexistent_baselines.json"),
            "--out",
            str(out),
        ]
    )
    assert ret == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "PHANES bench dashboard" in content
    assert "5e61db3c9fa7" in content


# ── Test 2: sparkline SVG is valid ────────────────────────────────────────────


def test_sparkline_valid_svg_multi_point() -> None:
    """Multi-point sparkline must be parseable SVG with polyline + circles."""
    series = [("abc1", 10.0), ("abc2", 20.0), ("abc3", 15.0)]
    svg = sparkline_svg(series)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<polyline" in svg
    assert "<circle" in svg


def test_sparkline_single_point() -> None:
    """Single-sample sparkline: no polyline but has circle."""
    svg = sparkline_svg([("abc1", 42.0)])
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<circle" in svg
    # Should NOT have a polyline (only 1 point)
    assert "<polyline" not in svg


def test_sparkline_all_null() -> None:
    """All-null series produces 'no data' SVG, not a crash."""
    svg = sparkline_svg([("abc1", None), ("abc2", None)])
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "no data" in svg


def test_sparkline_flat_series_no_divzero() -> None:
    """All-same values → flat sparkline, no ZeroDivisionError."""
    series = [("a", 50.0), ("b", 50.0), ("c", 50.0)]
    svg = sparkline_svg(series)
    assert "<svg" in svg
    assert "<polyline" in svg


# ── Test 3: direction colors fire correctly ───────────────────────────────────


def test_compare_color_regression_smaller_is_better() -> None:
    """Value 10% above baseline is 'bad' when smaller is better."""
    color = compare_color(value=110.0, baseline=100.0, direction="smaller")
    assert color == "cell-bad"


def test_compare_color_improvement_smaller_is_better() -> None:
    """Value 10% below baseline is 'good' when smaller is better."""
    color = compare_color(value=90.0, baseline=100.0, direction="smaller")
    assert color == "cell-good"


def test_compare_color_regression_larger_is_better() -> None:
    """Value 10% below baseline is 'bad' when larger is better."""
    color = compare_color(value=90.0, baseline=100.0, direction="larger")
    assert color == "cell-bad"


def test_compare_color_improvement_larger_is_better() -> None:
    """Value 10% above baseline is 'good' when larger is better."""
    color = compare_color(value=110.0, baseline=100.0, direction="larger")
    assert color == "cell-good"


def test_compare_color_within_threshold_neutral() -> None:
    """Value within 5% threshold returns neutral."""
    color = compare_color(value=103.0, baseline=100.0, direction="smaller")
    assert color == "cell-neutral"


def test_compare_color_none_values() -> None:
    """None value or baseline always returns neutral."""
    assert compare_color(None, 100.0, "smaller") == "cell-neutral"
    assert compare_color(100.0, None, "smaller") == "cell-neutral"


# ── Test 4: empty results dir produces 'no data' page, not crash ──────────────


def test_empty_results_dir_no_crash(tmp_path: Path) -> None:
    """Empty results dir renders 'no data' page without exceptions."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out = tmp_path / "dashboard.html"
    ret = main(
        [
            "--results",
            str(results_dir),
            "--out",
            str(out),
        ]
    )
    assert ret == 0
    content = out.read_text(encoding="utf-8")
    assert "PHANES bench dashboard" in content
    assert "No bench results" in content or "no data" in content.lower()


def test_nonexistent_results_dir_no_crash(tmp_path: Path) -> None:
    """Missing results dir produces empty dashboard, not FileNotFoundError."""
    out = tmp_path / "dashboard.html"
    ret = main(
        [
            "--results",
            str(tmp_path / "does_not_exist"),
            "--out",
            str(out),
        ]
    )
    assert ret == 0


# ── Test 5: multiple SHAs sort chronologically ───────────────────────────────


def test_multiple_shas_sort_chronologically(tmp_path: Path) -> None:
    """Three JSONs with distinct timestamps must appear in chrono order."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    entries = [
        ("sha_CCCC", "2026-06-01T10:00:00Z", 900.0),
        ("sha_AAAA", "2026-05-01T10:00:00Z", 1100.0),
        ("sha_BBBB", "2026-05-15T10:00:00Z", 1000.0),
    ]
    for sha, ts, boot in entries:
        data = _make_result(sha, ts, boot_ms=boot)
        (results_dir / f"{sha}.json").write_text(json.dumps(data), encoding="utf-8")

    loaded = load_results(results_dir)
    sha_order = [r["meta"]["sha"] for r in loaded]
    assert sha_order == [
        "sha_AAAA",
        "sha_BBBB",
        "sha_CCCC",
    ], f"Expected chrono order, got: {sha_order}"


def test_sparkline_has_three_points_for_three_shas(tmp_path: Path) -> None:
    """Three SHAs with varying boot_ms → polyline with 3 coordinate pairs."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    for i, (sha, ts, boot) in enumerate(
        [
            ("sha111111", "2026-05-01T00:00:00Z", 1000.0),
            ("sha222222", "2026-05-02T00:00:00Z", 1100.0),
            ("sha333333", "2026-05-03T00:00:00Z", 950.0),
        ]
    ):
        (results_dir / f"{sha}.json").write_text(
            json.dumps(_make_result(sha, ts, boot_ms=boot)), encoding="utf-8"
        )

    out = tmp_path / "dashboard.html"
    ret = main(["--results", str(results_dir), "--out", str(out)])
    assert ret == 0

    content = out.read_text(encoding="utf-8")
    # Find a polyline points= attribute and count coordinate pairs.
    import re

    polyline_match = re.search(r'<polyline points="([^"]+)"', content)
    assert polyline_match is not None, "No polyline found in rendered HTML"
    coords = polyline_match.group(1).strip().split()
    assert len(coords) == 3, f"Expected 3 coordinate pairs, got {len(coords)}: {coords}"


# ── Test 6: baseline overlay appears / disappears appropriately ───────────────


def test_baseline_overlay_present_when_baseline_file_exists(tmp_path: Path) -> None:
    """Dashed baseline line appears in sparkline when baselines.json is present."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "sha_BASE.json").write_text(
        json.dumps(_make_result("sha_BASE", "2026-05-01T00:00:00Z", boot_ms=1000.0)),
        encoding="utf-8",
    )

    baseline_file = tmp_path / "baselines.json"
    baseline_file.write_text(
        json.dumps({"boot_ms": 1000.0, "footprint": {"total_bytes": 3_000_000}}),
        encoding="utf-8",
    )

    out = tmp_path / "dashboard.html"
    ret = main(
        [
            "--results",
            str(results_dir),
            "--baseline",
            str(baseline_file),
            "--out",
            str(out),
        ]
    )
    assert ret == 0
    content = out.read_text(encoding="utf-8")
    # Dashed baseline line uses stroke-dasharray
    assert "stroke-dasharray" in content


def test_baseline_overlay_absent_when_no_baseline_file(tmp_path: Path) -> None:
    """No dashed line when baselines.json is absent."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "sha_NOBL.json").write_text(
        json.dumps(_make_result("sha_NOBL", "2026-05-01T00:00:00Z", boot_ms=1000.0)),
        encoding="utf-8",
    )

    out = tmp_path / "dashboard.html"
    ret = main(
        [
            "--results",
            str(results_dir),
            "--baseline",
            str(tmp_path / "nonexistent.json"),
            "--out",
            str(out),
        ]
    )
    assert ret == 0
    content = out.read_text(encoding="utf-8")
    assert "stroke-dasharray" not in content


# ── Test 7: leaf iterator correctness ─────────────────────────────────────────


def test_iter_leaves_nested() -> None:
    """iter_leaves extracts nested scalars with dotted keys."""
    data = {"a": {"b": 1.0, "c": None}, "d": 2}
    leaves = dict(iter_leaves(data))
    assert leaves == {"a.b": 1.0, "a.c": None, "d": 2}


def test_iter_leaves_skips_strings() -> None:
    """iter_leaves skips string values silently."""
    data = {"x": "hello", "y": 3.14}
    leaves = dict(iter_leaves(data))
    assert "x" not in leaves
    assert leaves["y"] == 3.14


# ── Test 8: self-contained HTML (no external assets) ─────────────────────────


def test_html_is_self_contained(tmp_path: Path) -> None:
    """Rendered HTML must not reference external CSS/JS assets (CDN, stylesheets).

    Commit links to github.com in <a href=...> are allowed — they are not
    external assets that block offline viewing.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "sha_SC.json").write_text(
        json.dumps(_make_result("sha_SC", "2026-05-01T00:00:00Z")),
        encoding="utf-8",
    )
    out = tmp_path / "dashboard.html"
    main(["--results", str(results_dir), "--out", str(out)])
    content = out.read_text(encoding="utf-8")

    # No external stylesheet <link rel="stylesheet" href="...">
    import re

    external_css = re.search(r'<link[^>]+href=["\']https?://', content)
    assert external_css is None, f"Found external CSS link: {external_css.group()}"

    # No external JS via <script src="https?://...">
    external_js = re.search(r'<script[^>]+src=["\']https?://', content)
    assert external_js is None, f"Found external JS src: {external_js.group()}"

    # No CDN domains in stylesheet or script positions
    assert "cdn." not in content.split("<a ")[0]  # before first anchor tag

    # Pure HTML + inline CSS, no JavaScript at all
    assert "<script" not in content


# ── format_qemu_mode ──────────────────────────────────────────────────────────


def test_format_qemu_mode_default_smp_only() -> None:
    """Default fast mode: SMP set, icount absent → just 'smp=N'."""
    assert format_qemu_mode({"qemu_smp": 4}) == "smp=4"


def test_format_qemu_mode_deterministic_includes_shift() -> None:
    """Deterministic mode: both fields → 'smp=N det=S'."""
    assert format_qemu_mode({"qemu_smp": 4, "qemu_icount_shift": 5}) == "smp=4 det=5"


def test_format_qemu_mode_pre_flag_results_show_unknown() -> None:
    """Old results predating the QEMU mode tags fall back to '?'."""
    assert format_qemu_mode({}) == "?"
    # Other meta keys present but no qemu_smp:
    assert format_qemu_mode({"sha": "abc", "defconfig": "qemu"}) == "?"


def test_dashboard_table_includes_mode_column(tmp_path: Path) -> None:
    """End-to-end: render_dashboard emits a 'Mode' header and a row cell."""
    # Minimal result with the QEMU mode tags populated.
    result = {
        "meta": {
            "sha": "abc123def456",
            "timestamp_utc": "2026-05-28T12:00:00Z",
            "defconfig": "qemu",
            "qemu_smp": 4,
            "qemu_icount_shift": 5,
        },
    }
    html_out = render_dashboard([result], baseline=None, github_repo="x/y")
    assert "<th>Mode</th>" in html_out
    assert ">smp=4 det=5<" in html_out


# ── detect_mixed_modes / mixed-mode banner ────────────────────────────────────


def _result_with_mode(sha: str, smp: int, icount: Optional[int]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "sha": sha,
        "timestamp_utc": "2026-05-28T12:00:00Z",
        "defconfig": "qemu",
        "qemu_smp": smp,
    }
    if icount is not None:
        meta["qemu_icount_shift"] = icount
    return {"meta": meta}


def test_detect_mixed_modes_uniform_returns_none() -> None:
    """All recent results in same mode → no warning."""
    from bench_dashboard import detect_mixed_modes

    results = [_result_with_mode(f"s{i:02d}", smp=4, icount=None) for i in range(5)]
    assert detect_mixed_modes(results) is None


def test_detect_mixed_modes_mixed_returns_list() -> None:
    """Window mixes smp-only and det runs → list of both modes."""
    from bench_dashboard import detect_mixed_modes

    results = [
        _result_with_mode("s00", smp=4, icount=None),
        _result_with_mode("s01", smp=4, icount=5),
        _result_with_mode("s02", smp=4, icount=None),
    ]
    modes = detect_mixed_modes(results)
    assert modes is not None
    assert "smp=4" in modes
    assert "smp=4 det=5" in modes


def test_detect_mixed_modes_only_window_counts() -> None:
    """A mode-switch older than MIXED_MODE_WINDOW is ignored."""
    from bench_dashboard import detect_mixed_modes, MIXED_MODE_WINDOW

    # 1 old det run + WINDOW recent smp-only runs → no warning.
    results = [_result_with_mode("old", smp=4, icount=5)]
    results.extend(
        _result_with_mode(f"new{i:02d}", smp=4, icount=None) for i in range(MIXED_MODE_WINDOW)
    )
    assert detect_mixed_modes(results) is None


def test_dashboard_emits_mixed_modes_banner() -> None:
    """End-to-end: render_dashboard surfaces the warning when modes mix."""
    results = [
        _result_with_mode("a" * 12, smp=4, icount=None),
        _result_with_mode("b" * 12, smp=4, icount=5),
    ]
    html_out = render_dashboard(results, baseline=None, github_repo="x/y")
    assert "Mixed QEMU modes" in html_out
    assert "smp=4" in html_out
    assert "smp=4 det=5" in html_out
