# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""bench_dashboard.py — Static HTML dashboard for AzOSRobotBrain bench results.

CLI:
    python3 tools/bench_dashboard.py \
        [--results bench/results/] \
        [--baseline bench/baselines.json] \
        [--out bench/dashboard.html]

Pure stdlib only: json, pathlib, datetime, html, os, sys, argparse, warnings.

Metric direction is imported from tools/bench_compare.py (_direction function)
when that module is present on sys.path.  If unavailable (e.g. tests that run
in isolation), falls back to a local copy.  The fallback keeps a TODO so the
duplication is visible.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_RESULTS_DIR = "bench/results"
DEFAULT_BASELINE_FILE = "bench/baselines.json"
DEFAULT_OUT_FILE = "bench/dashboard.html"

GITHUB_REPO_ENV = "PHANES_GITHUB_REPO"
DEFAULT_GITHUB_REPO = "fgrg1978/AzOS"

# Width/height for sparkline SVGs.
SVG_WIDTH = 200
SVG_HEIGHT = 40
SVG_PAD = 4  # px padding each side

# Regression threshold: 5 % in the bad direction triggers red; 5% good → green.
REGRESSION_THRESHOLD = 0.05

# Section display order and labels.
SECTION_ORDER: List[Tuple[str, str]] = [
    ("rtt_ms", "RTT (ms)"),
    ("throughput", "Throughput"),
    ("boot_ms", "Boot time (ms)"),
    ("wcet_us", "WCET (µs)"),
    ("jitter_ns", "Jitter (ns)"),
    ("footprint", "Footprint (bytes)"),
]

# ── Metric direction ──────────────────────────────────────────────────────────
# Prefer bench_compare._direction() — single source of truth.
# Fall back to local table if bench_compare is not importable.

_BENCH_COMPARE_DIRECTION_FN = None  # populated at import time below


def _try_import_bench_compare() -> None:
    """Attempt to import _direction from tools/bench_compare.py."""
    global _BENCH_COMPARE_DIRECTION_FN  # noqa: PLW0603
    tools_dir = Path(__file__).parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from bench_compare import _direction as _bc_dir  # type: ignore[import]

        _BENCH_COMPARE_DIRECTION_FN = _bc_dir
    except ImportError:
        pass  # bench_compare not yet committed; use local fallback


_try_import_bench_compare()

# Local fallback direction table.
# TODO(A6-unify): Remove this once bench_compare._direction covers all metrics
# and import always succeeds.  This table intentionally mirrors bench_compare's
# _DIRECTION_TABLE so behaviour is identical.
_FALLBACK_DIRECTION_TABLE: List[Tuple[str, str]] = [
    ("rtt_ms.p50", "smaller"),
    ("rtt_ms.p95", "smaller"),
    ("rtt_ms.p99", "smaller"),
    ("rtt_ms.stddev", "smaller"),
    ("rtt_ms.n_samples", "larger"),
    ("throughput.steady_msgs_per_s", "larger"),
    ("throughput.burst_peak_msgs_per_s", "larger"),
    ("boot_ms", "smaller"),
    ("wcet_us.", "smaller"),  # prefix match
    ("jitter_ns.", "smaller"),  # prefix match
    ("footprint.text_bytes", "smaller"),
    ("footprint.rodata_bytes", "smaller"),
    ("footprint.data_bytes", "smaller"),
    ("footprint.bss_bytes", "smaller"),
    ("footprint.total_bytes", "smaller"),
]

_WARNED_UNKNOWN_DIRECTIONS: set[str] = set()


def metric_direction(dotted_key: str) -> str:
    """Return 'smaller' or 'larger'; warn once for unknown keys.

    Delegates to bench_compare._direction() when available so there is a
    single source of truth.  Falls back to _FALLBACK_DIRECTION_TABLE.
    """
    if _BENCH_COMPARE_DIRECTION_FN is not None:
        direction, is_known = _BENCH_COMPARE_DIRECTION_FN(dotted_key)
        if not is_known and dotted_key not in _WARNED_UNKNOWN_DIRECTIONS:
            warnings.warn(
                f"bench_dashboard: unknown metric direction for '{dotted_key}'; "
                "defaulting to 'smaller'",
                stacklevel=2,
            )
            _WARNED_UNKNOWN_DIRECTIONS.add(dotted_key)
        # bench_compare returns "info" for sample-count metrics; treat as neutral
        return "smaller" if direction == "info" else direction

    # Fallback: local table with prefix matching.
    for prefix, direction in _FALLBACK_DIRECTION_TABLE:
        if prefix.endswith("."):
            if dotted_key.startswith(prefix) or dotted_key == prefix[:-1]:
                return direction
        elif dotted_key == prefix:
            return direction
    if dotted_key not in _WARNED_UNKNOWN_DIRECTIONS:
        warnings.warn(
            f"bench_dashboard: unknown metric direction for '{dotted_key}'; "
            "defaulting to 'smaller'",
            stacklevel=2,
        )
        _WARNED_UNKNOWN_DIRECTIONS.add(dotted_key)
    return "smaller"


# ── Leaf-metric flattening ─────────────────────────────────────────────────────

Scalar = Optional[float]  # mypy-friendly alias


def iter_leaves(
    obj: Any,
    prefix: Tuple[str, ...] = (),
) -> Iterator[Tuple[str, Scalar]]:
    """Yield (dotted_key, value) for every numeric/null leaf in *obj*.

    Recursively descends into dicts.  Non-numeric scalars (str, bool) are
    silently skipped.  Lists are skipped (no list metrics in current schema).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, prefix + (k,))
    elif obj is None or isinstance(obj, (int, float)):
        yield ".".join(prefix), obj
    # else: str/bool/list — skip silently


def section_leaves(
    result: Dict[str, Any],
    section_key: str,
) -> List[Tuple[str, Scalar]]:
    """Return sorted (dotted_key, value) pairs for one section of a result."""
    raw = result.get(section_key)
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        # Scalar top-level metric (e.g. boot_ms)
        return [(section_key, raw)]
    if isinstance(raw, dict):
        leaves = list(iter_leaves(raw, (section_key,)))
        return sorted(leaves, key=lambda t: t[0])
    return []


# ── JSON loading + chronological indexing ─────────────────────────────────────


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO-8601 UTC timestamp, tolerating trailing 'Z'."""
    ts = ts.rstrip("Z")
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all *.json files from *results_dir*, sorted chronologically.

    Files that fail to parse are skipped with a warning.  Non-JSON files
    (e.g. .gitkeep) are silently ignored.
    """
    entries: List[Tuple[datetime, Dict[str, Any]]] = []
    for p in results_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.warn(f"bench_dashboard: skipping {p.name}: {exc}")
            continue
        ts_raw = data.get("meta", {}).get("timestamp_utc", "")
        try:
            ts = _parse_timestamp(ts_raw) if ts_raw else datetime.min.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.min.replace(tzinfo=timezone.utc)
        entries.append((ts, data))
    entries.sort(key=lambda t: t[0])
    return [d for _, d in entries]


def load_baseline(baseline_file: Path) -> Optional[Dict[str, Any]]:
    """Load baselines.json if it exists; return None otherwise."""
    if not baseline_file.exists():
        return None
    try:
        return json.loads(baseline_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.warn(f"bench_dashboard: failed to parse {baseline_file}: {exc}")
        return None


# ── Sparkline SVG builder ──────────────────────────────────────────────────────


def _fmt(v: float, precision: int = 2) -> str:
    """Format a float compactly, avoiding scientific notation."""
    if v == int(v):
        return str(int(v))
    return f"{v:.{precision}f}"


def sparkline_svg(
    series: List[Tuple[str, Scalar]],
    baseline_value: Optional[float] = None,
    direction: str = "smaller",
    width: int = SVG_WIDTH,
    height: int = SVG_HEIGHT,
    pad: int = SVG_PAD,
) -> str:
    """Render a sparkline as a self-contained SVG string.

    Args:
        series: list of (sha_label, value) in chronological order.
        baseline_value: if given, draw a dashed horizontal reference line.
        direction: 'smaller' or 'larger' — used to color the last point.
        width, height, pad: geometry.

    Returns an ``<svg ...>...</svg>`` string.
    """
    # Filter out None values, but keep index position for X mapping.
    valid = [(i, sha, v) for i, (sha, v) in enumerate(series) if v is not None]

    n_total = len(series)
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    def _color_last(last_val: float) -> str:
        if baseline_value is None:
            return "#4a90d9"
        delta = (last_val - baseline_value) / baseline_value if baseline_value != 0 else 0
        improved = (
            delta < -REGRESSION_THRESHOLD
            if direction == "smaller"
            else delta > REGRESSION_THRESHOLD
        )
        regressed = (
            delta > REGRESSION_THRESHOLD
            if direction == "smaller"
            else delta < -REGRESSION_THRESHOLD
        )
        if improved:
            return "#27ae60"  # green
        if regressed:
            return "#e74c3c"  # red
        return "#4a90d9"  # neutral blue

    # No data at all.
    if not valid:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#f8f8f8;border-radius:3px">'
            f'<text x="{width//2}" y="{height//2+4}" text-anchor="middle" '
            f'font-size="9" fill="#aaa">no data</text></svg>'
        )

    values = [v for _, _, v in valid]
    vmin = min(values)
    vmax = max(values)

    def x_pos(idx: int) -> float:
        if n_total <= 1:
            return pad + usable_w / 2
        return pad + (idx / (n_total - 1)) * usable_w

    def y_pos(val: float) -> float:
        if vmax == vmin:
            return pad + usable_h / 2
        return pad + (1.0 - (val - vmin) / (vmax - vmin)) * usable_h

    parts: List[str] = []
    parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#f8f8f8;border-radius:3px;overflow:visible">'
    )

    # Baseline horizontal dashed line.
    if baseline_value is not None:
        # Clamp baseline within the viewport Y range (with a small guard).
        by = max(pad, min(height - pad, y_pos(baseline_value)))
        last_val = valid[-1][2]
        bl_color = _color_last(last_val)
        parts.append(
            f'<line x1="{pad}" y1="{by:.1f}" x2="{width-pad}" y2="{by:.1f}" '
            f'stroke="{bl_color}" stroke-width="1" stroke-dasharray="3,2" '
            f'opacity="0.7"/>'
        )

    # Trend polyline.
    if len(valid) >= 2:
        points_str = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, _, v in valid)
        parts.append(
            f'<polyline points="{points_str}" fill="none" stroke="#4a90d9" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
        )

    # Data-point circles with hover tooltips.
    last_idx_valid = valid[-1][0] if valid else -1
    for idx, sha, val in valid:
        cx = x_pos(idx)
        cy = y_pos(val)
        is_last = idx == last_idx_valid
        dot_color = _color_last(val) if is_last else "#4a90d9"
        r = 3.5 if is_last else 2.5
        title = html.escape(f"{sha}: {_fmt(val, 3)}")
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{dot_color}">'
            f"<title>{title}</title></circle>"
        )

    parts.append("</svg>")
    return "".join(parts)


# ── Comparison helpers ─────────────────────────────────────────────────────────


def compare_color(
    value: Optional[float],
    baseline: Optional[float],
    direction: str,
) -> str:
    """Return CSS color class name for a table cell."""
    if value is None or baseline is None or baseline == 0:
        return "cell-neutral"
    delta = (value - baseline) / baseline
    improved = (
        delta < -REGRESSION_THRESHOLD if direction == "smaller" else delta > REGRESSION_THRESHOLD
    )
    regressed = (
        delta > REGRESSION_THRESHOLD if direction == "smaller" else delta < -REGRESSION_THRESHOLD
    )
    if improved:
        return "cell-good"
    if regressed:
        return "cell-bad"
    return "cell-neutral"


# ── HTML rendering ─────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #333; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.6rem; margin-bottom: 4px; color: #1a1a2e; }
.sub { font-size: 0.85rem; color: #666; margin-bottom: 24px; }
h2 { font-size: 1.1rem; margin: 24px 0 10px; color: #1a1a2e;
     border-bottom: 2px solid #ddd; padding-bottom: 4px; }
h3 { font-size: 0.9rem; color: #555; margin: 12px 0 6px; }
.metrics-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.metric-card { background: #fff; border-radius: 6px; padding: 10px 14px;
               box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 260px; }
.metric-card .title { font-size: 0.78rem; color: #888; margin-bottom: 4px; }
.metric-card .stats { font-size: 0.8rem; color: #555; margin-top: 4px; }
.metric-card .stats span { margin-right: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem;
        background: #fff; border-radius: 6px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }
th { background: #1a1a2e; color: #fff; padding: 6px 8px;
     text-align: left; font-weight: 600; white-space: nowrap; }
td { padding: 5px 8px; border-bottom: 1px solid #eee; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f9f9f9; }
.sha-link { font-family: monospace; font-size: 0.78rem; }
.cell-good { background: #d4edda; color: #155724; }
.cell-bad  { background: #f8d7da; color: #721c24; }
.cell-neutral { }
.no-data { color: #aaa; font-style: italic; font-size: 0.85rem; padding: 8px 0; }
"""

# Headline metrics shown in the recent-runs table (dotted leaf keys).
HEADLINE_METRICS: List[str] = [
    "boot_ms",
    "rtt_ms.p99",
    "throughput.burst_peak_msgs_per_s",
    "wcet_us.timer_isr.max",
    "footprint.total_bytes",
]


def _github_commit_url(sha: str, repo: str) -> str:
    return f"https://github.com/{repo}/commit/{html.escape(sha)}"


def _render_stat(label: str, value: Optional[float]) -> str:
    if value is None:
        return f"<span>{label}: <em>n/a</em></span>"
    return f"<span>{label}: {_fmt(value)}</span>"


def format_qemu_mode(meta: Dict[str, Any]) -> str:
    """Render the harness-mode signature for a result row.

    The same kernel SHA bench'd under different QEMU modes (default SMP vs
    `-icount` deterministic timing) produces different `rdcycle`-based
    numbers — a diff between two rows that look like apples-to-apples but
    aren't is a common source of false-positive regressions.  This column
    surfaces the mode so you can spot it at a glance.

    Format:
      "smp=N"             — default fast mode (no icount)
      "smp=N det=S"       — deterministic mode (`-icount shift=S`)
      "?"                 — pre-flag results that didn't tag the mode
    """
    smp = meta.get("qemu_smp")
    icount = meta.get("qemu_icount_shift")
    if smp is None:
        return "?"
    if icount is not None:
        return f"smp={smp} det={icount}"
    return f"smp={smp}"


# Window over which to check mode mixing (in number of most-recent results,
# not days — calendar gaps don't matter for the comparability question).
MIXED_MODE_WINDOW: int = 10


def detect_mixed_modes(results: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Return the distinct QEMU modes in the most-recent window, or None
    if all results in the window share the same mode (no warning needed).

    Sparklines plot a single series per metric; if half the recent points
    were taken with `-icount` (deterministic rdcycle) and half without, the
    trend mixes apples and oranges and can manufacture phantom regressions.
    This check surfaces the situation as a banner at the top of the
    dashboard so the reader knows to interpret the curve with caution.
    """
    window = results[-MIXED_MODE_WINDOW:]
    if not window:
        return None
    modes = []
    for r in window:
        m = format_qemu_mode(r.get("meta", {}))
        if m not in modes:
            modes.append(m)
    return modes if len(modes) > 1 else None


def render_dashboard(
    results: List[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
    github_repo: str,
) -> str:
    """Build and return the full HTML string for the dashboard."""

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_shas = len(results)

    # Build per-SHA short labels for sparkline tooltips.
    sha_labels = [r.get("meta", {}).get("sha", f"#{i}")[:12] for i, r in enumerate(results)]

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>PHANES bench dashboard</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head>")
    parts.append('<body><div class="wrap">')
    parts.append("<h1>PHANES bench dashboard</h1>")
    parts.append(
        f'<p class="sub">Generated: {html.escape(generated_at)} &nbsp;|&nbsp; '
        f"{n_shas} SHA{'s' if n_shas != 1 else ''} indexed</p>"
    )

    mixed = detect_mixed_modes(results)
    if mixed is not None:
        mode_list = ", ".join(html.escape(m) for m in mixed)
        parts.append(
            '<p class="sub" style="color:#a00;border:1px solid #a00;'
            'padding:0.5em;border-radius:4px;">'
            f"<strong>Mixed QEMU modes in last {MIXED_MODE_WINDOW} runs: "
            f"{mode_list}.</strong> "
            "Sparklines below interleave points from different modes; "
            "absolute deltas across mode boundaries are not directly comparable "
            "(deterministic <code>-icount</code> mode yields different rdcycle "
            "values than the default fast SMP mode)."
            "</p>"
        )

    if not results:
        parts.append('<p class="no-data">No bench results found in the results directory.</p>')
        parts.append("</div></body></html>")
        return "\n".join(parts)

    # ── Sections: one per top-level metric category ────────────────────────────
    for section_key, section_label in SECTION_ORDER:
        parts.append(f"<h2>{html.escape(section_label)}</h2>")
        parts.append('<div class="metrics-grid">')

        # Collect all leaf keys in this section across ALL results.
        leaf_keys_seen: List[str] = []
        for r in results:
            for lk, _ in section_leaves(r, section_key):
                if lk not in leaf_keys_seen:
                    leaf_keys_seen.append(lk)

        if not leaf_keys_seen:
            parts.append('<p class="no-data">No data collected yet.</p>')
            parts.append("</div>")
            continue

        for dotted_key in leaf_keys_seen:
            # Build series for this leaf across all SHAs.
            series: List[Tuple[str, Scalar]] = []
            for r in results:
                sha_lbl = r.get("meta", {}).get("sha", "?")[:8]
                # Walk the dotted path.
                parts_path = dotted_key.split(".")
                node: Any = r
                for seg in parts_path:
                    if isinstance(node, dict):
                        node = node.get(seg)
                    else:
                        node = None
                        break
                val: Scalar = node if isinstance(node, (int, float)) else None
                series.append((sha_lbl, val))

            # Baseline value for this leaf.
            bl_val: Optional[float] = None
            if baseline is not None:
                parts_path = dotted_key.split(".")
                bl_node: Any = baseline
                for seg in parts_path:
                    if isinstance(bl_node, dict):
                        bl_node = bl_node.get(seg)
                    else:
                        bl_node = None
                        break
                if isinstance(bl_node, (int, float)):
                    bl_val = float(bl_node)

            direction = metric_direction(dotted_key)
            svg = sparkline_svg(series, baseline_value=bl_val, direction=direction)

            # Stats: min, max, last, baseline.
            valid_vals = [v for _, v in series if v is not None]
            last_val = series[-1][1] if series else None

            parts.append('<div class="metric-card">')
            parts.append(f'<div class="title">{html.escape(dotted_key)}</div>')
            parts.append(svg)
            parts.append('<div class="stats">')
            parts.append(_render_stat("last", last_val))
            if valid_vals:
                parts.append(_render_stat("min", min(valid_vals)))
                parts.append(_render_stat("max", max(valid_vals)))
            if bl_val is not None:
                parts.append(_render_stat("baseline", bl_val))
            parts.append("</div>")
            parts.append("</div>")

        parts.append("</div>")  # .metrics-grid

    # ── Recent-runs table ──────────────────────────────────────────────────────
    parts.append("<h2>Recent runs (last 10)</h2>")
    recent = results[-10:]

    # Build baseline map for headline metrics.
    baseline_headline: Dict[str, Optional[float]] = {}
    for hk in HEADLINE_METRICS:
        bl_node2: Any = baseline
        if bl_node2 is not None:
            for seg in hk.split("."):
                if isinstance(bl_node2, dict):
                    bl_node2 = bl_node2.get(seg)
                else:
                    bl_node2 = None
                    break
        baseline_headline[hk] = float(bl_node2) if isinstance(bl_node2, (int, float)) else None

    parts.append("<table>")
    parts.append("<thead><tr>")
    parts.append("<th>SHA</th><th>Timestamp</th><th>Config</th><th>Mode</th>")
    for hk in HEADLINE_METRICS:
        parts.append(f"<th>{html.escape(hk)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")

    for r in reversed(recent):  # newest first
        meta = r.get("meta", {})
        sha = meta.get("sha", "?")[:12]
        ts_raw = meta.get("timestamp_utc", "")
        defconfig = meta.get("defconfig", "?")
        qemu_mode = format_qemu_mode(meta)
        commit_url = _github_commit_url(sha, github_repo)

        parts.append("<tr>")
        parts.append(
            f'<td><a class="sha-link" href="{commit_url}" target="_blank">'
            f"{html.escape(sha)}</a></td>"
        )
        parts.append(f"<td>{html.escape(ts_raw)}</td>")
        parts.append(f"<td>{html.escape(str(defconfig))}</td>")
        parts.append(f"<td>{html.escape(qemu_mode)}</td>")

        for hk in HEADLINE_METRICS:
            # Walk path to extract value.
            node3: Any = r
            for seg in hk.split("."):
                if isinstance(node3, dict):
                    node3 = node3.get(seg)
                else:
                    node3 = None
                    break
            val3: Optional[float] = float(node3) if isinstance(node3, (int, float)) else None
            bl3 = baseline_headline.get(hk)
            direction3 = metric_direction(hk)
            css_class = compare_color(val3, bl3, direction3)
            cell_text = _fmt(val3) if val3 is not None else "n/a"
            parts.append(f'<td class="{css_class}">{html.escape(cell_text)}</td>')

        parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ── CLI entry point ────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a static HTML dashboard from bench/results/*.json"
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_DIR,
        metavar="DIR",
        help=f"Directory containing *.json bench results (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_FILE,
        metavar="FILE",
        help=f"Path to baselines.json (default: {DEFAULT_BASELINE_FILE}); "
        "silently skipped if absent.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_FILE,
        metavar="FILE",
        help=f"Output HTML file (default: {DEFAULT_OUT_FILE})",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    baseline_file = Path(args.baseline)
    out_file = Path(args.out)

    if not results_dir.exists():
        print(
            f"bench_dashboard: results directory '{results_dir}' does not exist; "
            "rendering empty dashboard.",
            file=sys.stderr,
        )
        results: List[Dict[str, Any]] = []
    else:
        results = load_results(results_dir)

    baseline = load_baseline(baseline_file)
    github_repo = os.environ.get(GITHUB_REPO_ENV, DEFAULT_GITHUB_REPO)

    html_content = render_dashboard(results, baseline, github_repo)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")

    n = len(results)
    print(
        f"bench_dashboard: wrote {out_file} "
        f"({n} SHA{'s' if n != 1 else ''}, "
        f"baseline={'yes' if baseline else 'no'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
