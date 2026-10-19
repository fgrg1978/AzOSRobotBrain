#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""bench_compare.py — Compare a bench result JSON against a committed baseline.

CLI usage
---------
    python3 tools/bench_compare.py \\
        --result   bench/results/<sha>.json \\
        --baseline bench/baselines.json \\
        [--waiver-text "BENCH-WAIVER: rtt_ms.p99,wcet_us.timer_isr.max"]

Exit codes
----------
    0  no unwaived regressions
    1  one or more unwaived regressions detected

Output
------
Markdown report to stdout (suitable for pasting into a GitHub PR comment).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

# ── Regression threshold ──────────────────────────────────────────────────────
# A metric has regressed if it moved in the "bad" direction by at least this
# fraction of the baseline value.
REGRESSION_THRESHOLD = 0.05  # 5 %
IMPROVEMENT_THRESHOLD = 0.05  # 5 % (only used for reporting, not gating)

# `bench_synth.*` (synthetic kernel microbenches) live on a noisier substrate:
# QEMU TCG `rdcycle` shows ~8% mean cross-run variance even via the cleanest
# path (the early-boot quiescent capture, task #73), and the smallest benches
# (sub-1000-cycle) swing further from host↔TCG cycle-mapping. A 5% gate there
# is all false positives; 15% sits above the measured noise floor while still
# catching the gross regressions TCG can actually resolve. Fine-grained (<10%)
# perf gating on these waits for real hardware (rdcycle = true counter).
BENCH_SYNTH_THRESHOLD = 0.15  # 15 %


def _threshold(path: str) -> float:
    """Regression threshold for a metric path. Wider for the TCG-noisy
    synthetic microbenches; default 5% for wall-clock / footprint metrics."""
    if path.startswith("bench_synth."):
        return BENCH_SYNTH_THRESHOLD
    return REGRESSION_THRESHOLD


# ── Top-level keys that are NOT metrics and must be skipped entirely ──────────
# Both result JSON and baseline JSON carry metadata blocks that should never
# be compared as numeric metrics.
NON_METRIC_TOP_LEVEL_KEYS = frozenset({"meta", "_meta"})

# ── Metric directions ─────────────────────────────────────────────────────────
# "smaller" → a smaller value is better (e.g. latency, code size).
#             Regression if result > baseline by >= threshold.
# "larger"  → a larger value is better (e.g. throughput).
#             Regression if result < baseline by >= threshold.
# "info"    → informational only; never gate on this metric.
#
# Keys are dot-separated paths into the result JSON structure.
# A prefix match is used: "wcet_us.<name>.max" matches any WCET point name.
# Prefix patterns are checked in order; the FIRST match wins.
_DIRECTION_TABLE: list[tuple[str, str]] = [
    # --- rtt_ms ---
    ("rtt_ms.p50", "smaller"),
    ("rtt_ms.p95", "smaller"),
    ("rtt_ms.p99", "smaller"),
    ("rtt_ms.stddev", "smaller"),
    ("rtt_ms.n_samples", "info"),  # sample count: not a regression metric
    # --- throughput ---
    ("throughput.steady_msgs_per_s", "larger"),
    ("throughput.burst_peak_msgs_per_s", "larger"),
    # --- boot ---
    ("boot_ms", "smaller"),
    # --- wcet_us (any point, any sub-field) ---
    ("wcet_us.", "smaller"),  # prefix for ALL wcet_us.*.*
    # --- wcet_per_fn (RFC-0027): telemetry only, NEVER gate. ---
    # The per-function WCET CI gate was rejected (RFC-0027 §Results, KILL2):
    # under QEMU TCG SMP, rdcycle includes cross-hart time, so per-function
    # samples vary 347%-298,809% on a fixed binary — noise, not timing.
    # Kept as informational telemetry for real-hardware (VF2/K1) bring-up.
    ("wcet_per_fn.", "info"),  # prefix for ALL wcet_per_fn.*.* — never gates
    # --- jitter_ns (any series, any sub-field) ---
    ("jitter_ns.", "smaller"),  # prefix for ALL jitter_ns.*.*
    # --- footprint ---
    ("footprint.text_bytes", "smaller"),
    ("footprint.rodata_bytes", "smaller"),
    ("footprint.data_bytes", "smaller"),
    ("footprint.bss_bytes", "smaller"),
    ("footprint.total_bytes", "smaller"),
]


def _direction(path: str) -> tuple[str, bool]:
    """Return (direction, is_known).

    direction is one of "smaller", "larger", "info".
    is_known is False if the path was not found in the table (unknown metric).
    """
    for prefix, direction in _DIRECTION_TABLE:
        if prefix.endswith("."):
            # Prefix match.
            if path.startswith(prefix) or path == prefix[:-1]:
                return direction, True
        else:
            if path == prefix:
                return direction, True
    # Unknown metric: default to good-when-smaller.
    return "smaller", False


def _walk_leaves(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Recursively yield (dot_path, value) for all leaf values.

    Dicts are traversed; lists produce indexed paths (e.g. "key.0").
    Non-None scalars (int, float, bool, str) are leaves.
    None values are also yielded so callers can handle them explicitly.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{prefix}.{k}" if prefix else k
            yield from _walk_leaves(v, child_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_path = f"{prefix}.{i}" if prefix else str(i)
            yield from _walk_leaves(v, child_path)
    else:
        yield prefix, obj


def _parse_waiver_set(waiver_text: str) -> set[str]:
    """Extract the set of waived metric paths from a raw waiver text.

    The waiver text may come from a commit body.  We look for the substring
    ``BENCH-WAIVER:`` and take everything after it on the same logical chunk
    (up to the next newline).  Multiple paths are comma-separated.

    Known limitation: in GitHub Actions, `git log -1 --format=%B` on a PR
    event returns the *merge commit* message (auto-generated), not the PR
    head commit message.  If the waiver is in the PR commit body rather than
    the merge commit body, it will not be found.  To reliably pass a waiver
    from a PR, put the BENCH-WAIVER line in the PR description and extract it
    from `${{ github.event.pull_request.body }}` instead.

    Examples
    --------
    >>> _parse_waiver_set("fix: tighten loop\\n\\nBENCH-WAIVER: rtt_ms.p99, wcet_us.timer_isr.max")
    {'rtt_ms.p99', 'wcet_us.timer_isr.max'}
    >>> _parse_waiver_set("")
    set()
    >>> _parse_waiver_set("no waiver here")
    set()
    """
    marker = "BENCH-WAIVER:"
    idx = waiver_text.find(marker)
    if idx == -1:
        return set()
    after = waiver_text[idx + len(marker) :]
    # Take until next newline (if any).
    line = after.split("\n")[0]
    paths = {p.strip() for p in line.split(",") if p.strip()}
    return paths


# ── Comparison result dataclasses (plain dicts for simplicity) ────────────────


class _Row:
    """Holds information about a single metric comparison."""

    __slots__ = (
        "path",
        "baseline_val",
        "result_val",
        "pct_change",
        "direction",
        "is_known",
        "status",  # "regression" | "improvement" | "ok" | "new" | "missing" | "info" | "skip"
        "waived",
    )

    def __init__(
        self,
        path: str,
        baseline_val: Optional[float],
        result_val: Optional[float],
        pct_change: Optional[float],
        direction: str,
        is_known: bool,
        status: str,
        waived: bool,
    ) -> None:
        self.path = path
        self.baseline_val = baseline_val
        self.result_val = result_val
        self.pct_change = pct_change
        self.direction = direction
        self.is_known = is_known
        self.status = status
        self.waived = waived

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"_Row(path={self.path!r}, status={self.status!r}, "
            f"pct={self.pct_change}, waived={self.waived})"
        )


def _compare_value(
    path: str,
    baseline_val: Optional[float],
    result_val: Optional[float],
    waiver_set: set[str],
) -> _Row:
    direction, is_known = _direction(path)
    waived = path in waiver_set

    # Both null → skip.
    if baseline_val is None and result_val is None:
        return _Row(path, None, None, None, direction, is_known, "skip", waived)

    # Baseline null, result populated → new metric.
    if baseline_val is None:
        return _Row(path, None, result_val, None, direction, is_known, "new", waived)

    # Baseline populated, result null → missing (may indicate harness regression).
    if result_val is None:
        return _Row(path, baseline_val, None, None, direction, is_known, "missing", waived)

    # direction "info" → informational only.
    if direction == "info":
        return _Row(path, baseline_val, result_val, None, direction, is_known, "info", waived)

    # Baseline is zero → avoid division by zero.
    if baseline_val == 0:
        # Can't compute percent change; treat as OK unless result also zero.
        status = "ok" if result_val == 0 else "skip"
        return _Row(path, baseline_val, result_val, None, direction, is_known, status, waived)

    raw_change = (result_val - baseline_val) / abs(baseline_val)

    # For good-when-smaller: regression if result > baseline (positive change).
    # For good-when-larger:  regression if result < baseline (negative change).
    thresh = _threshold(path)
    if direction == "smaller":
        pct_change = raw_change  # positive = got bigger = bad
        regression = pct_change >= thresh
        improvement = pct_change <= -IMPROVEMENT_THRESHOLD
    else:  # "larger"
        pct_change = -raw_change  # positive = got smaller = bad
        regression = pct_change >= thresh
        improvement = pct_change <= -IMPROVEMENT_THRESHOLD

    if regression:
        status = "regression"
    elif improvement:
        status = "improvement"
    else:
        status = "ok"

    return _Row(path, baseline_val, result_val, pct_change, direction, is_known, status, waived)


def compare(
    result: dict[str, Any],
    baseline: dict[str, Any],
    waiver_set: set[str],
) -> list[_Row]:
    """Compare all leaf metrics in result against baseline.

    Returns a list of _Row objects, one per metric path found in either
    the result or the baseline (excluding non-metric top-level keys).
    """

    # Filter out non-metric top-level keys before walking.
    def _filtered(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k not in NON_METRIC_TOP_LEVEL_KEYS}

    result_leaves = dict(_walk_leaves(_filtered(result)))
    baseline_leaves = dict(_walk_leaves(_filtered(baseline)))

    all_paths = set(result_leaves) | set(baseline_leaves)
    rows: list[_Row] = []

    for path in sorted(all_paths):
        result_val = result_leaves.get(path)
        baseline_val = baseline_leaves.get(path)

        # Coerce to float if numeric, else skip (string/bool metadata).
        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return None  # non-numeric leaf → skip

        r_float = _to_float(result_val)
        b_float = _to_float(baseline_val)

        # If both sides are non-numeric non-None, skip silently.
        if r_float is None and result_val is not None:
            continue
        if b_float is None and baseline_val is not None:
            continue

        rows.append(_compare_value(path, b_float, r_float, waiver_set))

    return rows


# ── Markdown report rendering ─────────────────────────────────────────────────


def _fmt_val(v: Optional[float]) -> str:
    if v is None:
        return "null"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.4g}"


def _fmt_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "—"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct * 100:.1f}%"


def render_report(rows: list[_Row]) -> str:
    """Render a markdown report from the comparison rows."""
    regressions = [r for r in rows if r.status == "regression" and not r.waived]
    waived_regs = [r for r in rows if r.status == "regression" and r.waived]
    improvements = [r for r in rows if r.status == "improvement"]
    new_metrics = [r for r in rows if r.status == "new"]
    missing = [r for r in rows if r.status == "missing"]
    unknown_warns = [
        r for r in rows if not r.is_known and r.status not in ("skip", "new", "missing", "info")
    ]
    stable = [r for r in rows if r.status in ("ok", "info")]

    lines: list[str] = ["<!-- BENCH-REPORT -->", "## Benchmark Regression Report", ""]

    # ── Regressions ───────────────────────────────────────────────────────────
    if regressions:
        lines.append("### ❌ Regressions (unwaived)")
        lines.append("")
        lines.append("| Metric | Baseline | Result | Change | Direction |")
        lines.append("|--------|----------|--------|--------|-----------|")
        for r in regressions:
            lines.append(
                f"| `{r.path}` | {_fmt_val(r.baseline_val)} | "
                f"{_fmt_val(r.result_val)} | **{_fmt_pct(r.pct_change)}** | "
                f"{r.direction} |"
            )
        lines.append("")
    else:
        lines.append("### ✅ No unwaived regressions")
        lines.append("")

    # ── Waived regressions ────────────────────────────────────────────────────
    if waived_regs:
        lines.append("### ⚠️ Regressions (waived by commit body)")
        lines.append("")
        lines.append("| Metric | Baseline | Result | Change |")
        lines.append("|--------|----------|--------|--------|")
        for r in waived_regs:
            lines.append(
                f"| `{r.path}` | {_fmt_val(r.baseline_val)} | "
                f"{_fmt_val(r.result_val)} | {_fmt_pct(r.pct_change)} |"
            )
        lines.append("")

    # ── Improvements ──────────────────────────────────────────────────────────
    if improvements:
        lines.append("### ✅ Improvements")
        lines.append("")
        lines.append("| Metric | Baseline | Result | Change |")
        lines.append("|--------|----------|--------|--------|")
        for r in improvements:
            lines.append(
                f"| `{r.path}` | {_fmt_val(r.baseline_val)} | "
                f"{_fmt_val(r.result_val)} | {_fmt_pct(r.pct_change)} |"
            )
        lines.append("")

    # ── New metrics ───────────────────────────────────────────────────────────
    if new_metrics:
        lines.append("### ℹ️ New metrics (not in baseline)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for r in new_metrics:
            lines.append(f"| `{r.path}` | {_fmt_val(r.result_val)} |")
        lines.append("")

    # ── Missing metrics ───────────────────────────────────────────────────────
    if missing:
        lines.append("### ⚠️ Missing metrics (in baseline, not in result)")
        lines.append("")
        lines.append(
            "> These metrics were captured when the baseline was recorded but "
            "are absent from this run.  This may indicate a harness issue."
        )
        lines.append("")
        lines.append("| Metric | Baseline value |")
        lines.append("|--------|----------------|")
        for r in missing:
            lines.append(f"| `{r.path}` | {_fmt_val(r.baseline_val)} |")
        lines.append("")

    # ── Unknown metric warnings ───────────────────────────────────────────────
    if unknown_warns:
        lines.append("### ⚠️ Unknown metrics (defaulted to good-when-smaller)")
        lines.append("")
        lines.append("| Metric | Baseline | Result | Change |")
        lines.append("|--------|----------|--------|--------|")
        for r in unknown_warns:
            lines.append(
                f"| `{r.path}` | {_fmt_val(r.baseline_val)} | "
                f"{_fmt_val(r.result_val)} | {_fmt_pct(r.pct_change)} |"
            )
        lines.append("")

    # ── Stable (collapsed) ────────────────────────────────────────────────────
    if stable:
        lines.append("<details>")
        lines.append("<summary>Stable metrics (no significant change)</summary>")
        lines.append("")
        lines.append("| Metric | Baseline | Result | Change |")
        lines.append("|--------|----------|--------|--------|")
        for r in stable:
            lines.append(
                f"| `{r.path}` | {_fmt_val(r.baseline_val)} | "
                f"{_fmt_val(r.result_val)} | {_fmt_pct(r.pct_change)} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── Summary line ──────────────────────────────────────────────────────────
    n_reg = len(regressions)
    n_waived = len(waived_regs)
    n_imp = len(improvements)
    n_new = len(new_metrics)
    n_miss = len(missing)
    summary_parts = []
    if n_reg:
        summary_parts.append(f"**{n_reg} regression(s)**")
    if n_waived:
        summary_parts.append(f"{n_waived} waived regression(s)")
    if n_imp:
        summary_parts.append(f"{n_imp} improvement(s)")
    if n_new:
        summary_parts.append(f"{n_new} new metric(s)")
    if n_miss:
        summary_parts.append(f"{n_miss} missing metric(s)")

    if not summary_parts:
        summary_parts.append("all metrics stable")

    lines.append(f"**Summary**: {', '.join(summary_parts)}.")
    lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot load {path!r}: {exc}", file=sys.stderr)
        sys.exit(10)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare a bench result JSON against a committed baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--result",
        required=True,
        metavar="PATH",
        help="bench/results/<sha>.json produced by bench_e2e_collect.py",
    )
    ap.add_argument(
        "--baseline",
        required=True,
        metavar="PATH",
        help="bench/baselines.json committed in this repo",
    )
    ap.add_argument(
        "--waiver-text",
        default="",
        metavar="TEXT",
        help=(
            "Text that may contain 'BENCH-WAIVER: path1,path2'.  "
            "Typically the commit message body extracted by CI."
        ),
    )
    args = ap.parse_args(argv)

    result_data = _load_json(args.result)
    baseline_data = _load_json(args.baseline)
    waiver_set = _parse_waiver_set(args.waiver_text)

    if waiver_set:
        print(
            f"[bench_compare] waiver active for: {', '.join(sorted(waiver_set))}", file=sys.stderr
        )

    rows = compare(result_data, baseline_data, waiver_set)
    report = render_report(rows)
    print(report)

    # Exit 1 if any unwaived regression.
    unwaived_regressions = [r for r in rows if r.status == "regression" and not r.waived]
    return 1 if unwaived_regressions else 0


if __name__ == "__main__":
    sys.exit(main())
