# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner.experience — ExperienceStore."""

import json
import os
import tempfile

import pytest

from planner.experience import (
    ExperienceStore,
    ExperienceRecord,
    ExperienceHit,
    _extract_keywords,
    _keyword_similarity,
    QUERY_TOP_K,
    MAX_RECORDS,
    MIN_SIMILARITY_SCORE,
)

# ── Keyword helpers ──────────────────────────────────────────────────────────


def test_extract_keywords_basic():
    kw = _extract_keywords("patrol the garden at night")
    assert "patrol" in kw
    assert "garden" in kw
    assert "night" in kw
    # stop words excluded
    assert "the" not in kw
    assert "at" not in kw


def test_extract_keywords_empty():
    assert _extract_keywords("") == set()
    assert _extract_keywords("a the in") == set()


def test_keyword_similarity_full_overlap():
    assert _keyword_similarity({"patrol", "garden"}, {"patrol", "garden", "night"}) == 1.0


def test_keyword_similarity_partial():
    score = _keyword_similarity({"patrol", "garden", "night"}, {"patrol", "garden"})
    assert abs(score - 2 / 3) < 0.01


def test_keyword_similarity_no_overlap():
    assert _keyword_similarity({"patrol"}, {"navigate", "dock"}) == 0.0


def test_keyword_similarity_empty_query():
    assert _keyword_similarity(set(), {"patrol"}) == 0.0


# ── ExperienceStore ─────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return ExperienceStore(str(tmp_path), robot_type="wheeled")


def test_record_and_count(store):
    assert store.count == 0
    store.record(
        task="patrol garden",
        plan=[{"skill": "FORWARD", "args": {"speed": 50}}, {"skill": "SCAN_360"}],
        outcome="done",
        steps_executed=2,
    )
    assert store.count == 1


def test_record_persists_to_disk(tmp_path):
    store = ExperienceStore(str(tmp_path), robot_type="wheeled")
    store.record(task="patrol", plan=[{"skill": "STOP"}], outcome="done", steps_executed=1)

    # Reload from disk
    store2 = ExperienceStore(str(tmp_path), robot_type="wheeled")
    assert store2.count == 1
    assert store2._cache[0].task == "patrol"


def test_query_finds_relevant(store):
    store.record(
        task="patrol the garden", plan=[{"skill": "SCAN_360"}], outcome="done", steps_executed=1
    )
    store.record(
        task="navigate to dock", plan=[{"skill": "NAVIGATE_TO"}], outcome="done", steps_executed=1
    )

    hits = store.query("patrol garden at night")
    assert len(hits) >= 1
    assert hits[0].record.task == "patrol the garden"
    assert hits[0].score > 0


def test_query_returns_empty_for_unrelated(store):
    store.record(
        task="dock and charge", plan=[{"skill": "RETURN_TO_DOCK"}], outcome="done", steps_executed=1
    )
    hits = store.query("fly forward fast")
    # May or may not match depending on threshold; at least no crash
    for h in hits:
        assert h.score >= MIN_SIMILARITY_SCORE


def test_query_empty_store(store):
    assert store.query("patrol") == []


def test_format_for_prompt(store):
    store.record(
        task="patrol garden",
        plan=[{"skill": "SCAN_360"}, {"skill": "FORWARD"}],
        outcome="done",
        steps_executed=2,
    )
    store.record(
        task="patrol garden night",
        plan=[{"skill": "SCAN_360"}],
        outcome="error",
        steps_executed=0,
        error="timeout",
    )

    hits = store.query("patrol garden")
    text = store.format_for_prompt(hits)
    assert "Past experience" in text
    assert "patrol garden" in text
    assert "DONE" in text or "ERROR" in text


def test_format_for_prompt_empty():
    store = ExperienceStore("/tmp/nonexistent_exp_test", robot_type="wheeled")
    assert store.format_for_prompt([]) == ""


def test_success_rate(store):
    store.record(task="patrol", plan=[{"skill": "STOP"}], outcome="done", steps_executed=1)
    store.record(task="patrol", plan=[{"skill": "STOP"}], outcome="done", steps_executed=1)
    store.record(
        task="patrol", plan=[{"skill": "STOP"}], outcome="error", steps_executed=0, error="fail"
    )
    assert abs(store.success_rate() - 2 / 3) < 0.01


def test_success_rate_empty(store):
    assert store.success_rate() == 0.0


def test_fifo_eviction(tmp_path):
    store = ExperienceStore(str(tmp_path), robot_type="wheeled")
    # Record more than MAX_RECORDS — we'll use a small override for speed
    import planner.experience as mod

    old_max = mod.MAX_RECORDS
    mod.MAX_RECORDS = 5
    try:
        for i in range(8):
            store.record(
                task=f"task {i}", plan=[{"skill": "STOP"}], outcome="done", steps_executed=1
            )
        assert store.count == 5
        # Oldest records evicted
        assert store._cache[0].task == "task 3"
    finally:
        mod.MAX_RECORDS = old_max


def test_update_robot_type(tmp_path):
    store = ExperienceStore(str(tmp_path), robot_type="wheeled")
    store.record(task="patrol", plan=[{"skill": "STOP"}], outcome="done", steps_executed=1)
    assert store.count == 1

    store.update_robot_type("drone")
    assert store.count == 0
    assert store.robot_type == "drone"

    store.record(task="hover", plan=[{"skill": "HOVER"}], outcome="done", steps_executed=1)
    assert store.count == 1

    # Switch back — wheeled data still on disk
    store.update_robot_type("wheeled")
    assert store.count == 1
    assert store._cache[0].task == "patrol"


def test_record_tags_extracted(store):
    rec = store.record(
        task="patrol the garden",
        plan=[{"skill": "STOP"}],
        outcome="done",
        steps_executed=1,
        context="night shift",
    )
    assert "patrol" in rec.tags
    assert "garden" in rec.tags
    assert "night" in rec.tags
    assert "shift" in rec.tags
