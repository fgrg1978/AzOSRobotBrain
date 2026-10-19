# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""The WCET split in tools/bench_e2e_collect.py is keyed by point NAME.

The kernel renumbers the ids in wcet_points.json whenever a generated point is
added or removed; the split must not depend on them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile

_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "bench_e2e_collect.py")
_spec = importlib.util.spec_from_file_location("bench_e2e_collect", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _sample() -> dict:
    return {"min": 1, "max": 2, "avg": 1, "count": 3, "viol": 0}


def _split_with_index(index: dict) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "wcet_points.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
        old = os.environ.get("AZOS_WCET_POINTS")
        os.environ["AZOS_WCET_POINTS"] = path
        try:
            return mod._split_wcet(
                {"pid_loop": _sample(), "tcp_send_segment": _sample(), "lifecycle_schedule": _sample()}
            )
        finally:
            if old is None:
                del os.environ["AZOS_WCET_POINTS"]
            else:
                os.environ["AZOS_WCET_POINTS"] = old


def test_split_ignores_generated_point_ids():
    base = {
        "fixed_points": ["pid_loop"],
        "generated_points": [
            {"id": 15, "name": "lifecycle_schedule", "budget_us": 30},
            {"id": 16, "name": "tcp_send_segment", "budget_us": 200},
        ],
    }
    renumbered = {
        "fixed_points": ["pid_loop"],
        "generated_points": [
            {"id": 19, "name": "tcp_send_segment", "budget_us": 200},
            {"id": 15, "name": "lifecycle_schedule", "budget_us": 30},
        ],
    }
    a = _split_with_index(base)
    b = _split_with_index(renumbered)
    assert a == b
    assert set(a["wcet_us"]) == {"pid_loop"}
    assert set(a["wcet_per_fn"]) == {"tcp_send_segment", "lifecycle_schedule"}


def test_unreadable_index_falls_back_to_historical_fixed_set():
    old = os.environ.get("AZOS_WCET_POINTS")
    os.environ["AZOS_WCET_POINTS"] = "/nonexistent/wcet_points.json"
    try:
        names = mod._load_fixed_wcet_names()
    finally:
        if old is None:
            del os.environ["AZOS_WCET_POINTS"]
        else:
            os.environ["AZOS_WCET_POINTS"] = old
    assert "pid_loop" in names
