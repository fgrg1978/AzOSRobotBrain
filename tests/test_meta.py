# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner.meta — MetaReviewer."""

import json
import os
import tempfile
import time

import pytest

from planner.experience import ExperienceStore
from planner.meta import (
    MetaReviewer,
    HeuristicRule,
    HeuristicSet,
    MIN_RECORDS_FOR_REVIEW,
    REVIEW_COOLDOWN_S,
    MAX_RULES,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def exp_store(tmp_path):
    return ExperienceStore(str(tmp_path), robot_type="wheeled")


class FakeLLMClient:
    """Fake OpenAI client that returns a canned heuristic response."""

    def __init__(self, response: str = ""):
        self._response = response

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        class Choice:
            class Message:
                content = self._response

            message = Message()

        class Resp:
            choices = [Choice()]

        return Resp()


def _make_reviewer(exp_store, response: str = "", heuristics_dir=None) -> MetaReviewer:
    """Create a MetaReviewer with a fake LLM client.

    `heuristics_dir` defaults to a throwaway directory. It is NOT optional in
    spirit: `_save_rules` writes `heuristics_{robot_type}.json` there, and
    until 2026-09-09 this helper left the attribute unset so the write landed
    in the production path (`data/experience/`). That file is versioned, so
    every run of this suite left the working tree dirty with test output —
    and the copy committed there was itself test output (`"rule": "rule one"`).
    """
    reviewer = MetaReviewer.__new__(MetaReviewer)
    reviewer.model = "test"
    reviewer.experience = exp_store
    reviewer.robot_type = "wheeled"
    reviewer.heuristics_dir = str(heuristics_dir or _throwaway_dir())
    reviewer._heuristics = HeuristicSet(robot_type="wheeled")
    reviewer.client = FakeLLMClient(response)
    return reviewer


def _throwaway_dir() -> str:
    """A temp directory that outlives the test but not the machine."""
    return tempfile.mkdtemp(prefix="meta-heuristics-")


def _populate_experience(store, n=10):
    """Add N records to the experience store."""
    for i in range(n):
        outcome = "done" if i % 3 != 0 else "error"
        store.record(
            task=f"patrol zone {i}",
            plan=[{"skill": "SCAN_360"}, {"skill": "FORWARD", "args": {"speed": 50}}],
            outcome=outcome,
            steps_executed=2 if outcome == "done" else 1,
            error="timeout" if outcome == "error" else "",
        )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_should_review_not_enough_records(exp_store):
    reviewer = _make_reviewer(exp_store)
    assert not reviewer.should_review()


def test_should_review_enough_records(exp_store):
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    reviewer = _make_reviewer(exp_store)
    assert reviewer.should_review()


def test_should_review_cooldown(exp_store):
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    reviewer = _make_reviewer(exp_store)
    reviewer._heuristics.last_review = time.time()  # just reviewed
    assert not reviewer.should_review()


def test_review_parses_rules(exp_store):
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)

    llm_response = json.dumps(
        {
            "rules": [
                {
                    "id": "rule_01",
                    "rule": "Always SCAN_360 before NAVIGATE_TO in unknown areas",
                    "reason": "3 out of 4 failures happened without prior scanning",
                    "confidence": 0.8,
                },
                {
                    "id": "rule_02",
                    "rule": "Use lower speed (30) in patrol zones with obstacles",
                    "reason": "High speed patrols had more interrupts",
                    "confidence": 0.6,
                },
            ]
        }
    )

    reviewer = _make_reviewer(exp_store, response=llm_response)
    rules = reviewer.review()

    assert len(rules) == 2
    assert rules[0].rule == "Always SCAN_360 before NAVIGATE_TO in unknown areas"
    assert rules[0].confidence == 0.8
    assert rules[1].id == "rule_02"
    assert reviewer.review_count == 1


def test_review_handles_bad_json(exp_store):
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    reviewer = _make_reviewer(exp_store, response="not json at all")
    rules = reviewer.review()
    assert rules == []  # no crash, empty rules


def test_review_handles_markdown_fenced_json(exp_store):
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    llm_response = '```json\n{"rules": [{"id": "r1", "rule": "test rule", "reason": "test", "confidence": 0.7}]}\n```'
    reviewer = _make_reviewer(exp_store, response=llm_response)
    rules = reviewer.review()
    assert len(rules) == 1
    assert rules[0].rule == "test rule"


def test_rules_for_prompt_empty(exp_store):
    reviewer = _make_reviewer(exp_store)
    assert reviewer.rules_for_prompt() == ""


def test_rules_for_prompt_formats(exp_store):
    reviewer = _make_reviewer(exp_store)
    reviewer._heuristics.rules = [
        HeuristicRule(id="r1", rule="Always scan first", reason="safety", confidence=0.8),
        HeuristicRule(
            id="r2", rule="Low confidence rule", reason="weak", confidence=0.2
        ),  # below threshold
    ]
    text = reviewer.rules_for_prompt()
    assert "Always scan first" in text
    assert "Low confidence rule" not in text  # filtered by confidence


def test_rules_persist_to_disk(tmp_path, exp_store):
    """Rules reach disk, in the directory the reviewer was given.

    This test used to monkeypatch the module-level `HEURISTICS_DIR` and put it
    back in a `finally`. The instinct was right and the reach was too short:
    the seam was a module global, so the patch protected THIS test and every
    other one in the file still wrote into `data/experience/` — the production
    path, versioned, which is why the suite left the working tree dirty. The
    directory is now an instance attribute, so redirecting one reviewer
    redirects exactly one reviewer.
    """
    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    llm_response = json.dumps(
        {"rules": [{"id": "r1", "rule": "test persist", "reason": "test", "confidence": 0.9}]}
    )
    reviewer = _make_reviewer(exp_store, response=llm_response, heuristics_dir=tmp_path)
    reviewer.review()

    path = os.path.join(str(tmp_path), "heuristics_wheeled.json")
    assert os.path.exists(path)

    with open(path) as f:
        data = json.load(f)
    assert len(data["rules"]) == 1
    assert data["rules"][0]["rule"] == "test persist"


def test_update_robot_type(exp_store):
    reviewer = _make_reviewer(exp_store)
    reviewer._heuristics.rules = [
        HeuristicRule(id="r1", rule="wheeled rule", reason="", confidence=0.8)
    ]
    reviewer.update_robot_type("drone")
    assert reviewer.robot_type == "drone"
    assert reviewer.rule_count == 0  # fresh set for drone


def test_rule_count_and_review_count(exp_store):
    reviewer = _make_reviewer(exp_store)
    assert reviewer.rule_count == 0
    assert reviewer.review_count == 0

    _populate_experience(exp_store, MIN_RECORDS_FOR_REVIEW)
    llm_response = json.dumps(
        {"rules": [{"id": "r1", "rule": "rule one", "reason": "r", "confidence": 0.5}]}
    )
    reviewer = _make_reviewer(exp_store, response=llm_response)
    reviewer.review()
    assert reviewer.rule_count == 1
    assert reviewer.review_count == 1
