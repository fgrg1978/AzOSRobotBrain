# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Experience store — persistent memory of plan executions and outcomes.

Records every task→plan→outcome cycle so the TaskPlanner can learn from
past successes and failures.  Implements the "experience loop" inspired by
Hyperagents (Zhang et al., 2026): the robot remembers what worked.

Storage: one JSONL file per robot type under ``data/experience/``.
Each line is an ExperienceRecord serialised as JSON.

Retrieval: keyword overlap between the new task description and stored
task descriptions.  Top-K most relevant records are returned as context
for the LLM planner.

Usage:
    store = ExperienceStore("data/experience", robot_type="wheeled")
    store.record(task="patrol the garden", plan=[...], outcome="done",
                 context="night, battery 80%", steps_executed=3, error="")
    hits = store.query("patrol garden at night", k=3)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger("brain.experience")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIENCE_DIR_DEFAULT = "data/experience"
QUERY_TOP_K = 3  # default number of past experiences to retrieve
MAX_RECORDS = 5_000  # cap per robot type (FIFO eviction)
MIN_SIMILARITY_SCORE = 0.15  # ignore records below this relevance


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ExperienceRecord:
    """One plan-execution cycle."""

    timestamp: float
    task: str  # original free-text task
    context: str  # sensor/env context at planning time
    plan: list[dict]  # [{"skill": ..., "args": ...}, ...]
    outcome: str  # "done" | "interrupted" | "error"
    steps_total: int  # len(plan)
    steps_executed: int  # how many actually ran
    error: str = ""  # error message if outcome == "error"
    interrupt_reason: str = ""  # reason if outcome == "interrupted"
    duration_s: float = 0.0  # wall-clock time for execution
    tags: list[str] = field(default_factory=list)  # auto-extracted keywords


@dataclass
class ExperienceHit:
    """A retrieved experience with relevance score."""

    record: ExperienceRecord
    score: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Keyword extraction (simple, no deps)
# ---------------------------------------------------------------------------

# Common stop words to ignore during keyword matching
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "it",
        "that",
        "this",
        "be",
        "as",
        "do",
        "all",
        "any",
        "no",
        "not",
        "so",
        "if",
        "up",
        "out",
        "from",
    }
)


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keyword tokens from free text."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _keyword_similarity(query_kw: set[str], record_kw: set[str]) -> float:
    """Jaccard-like overlap normalised by query size."""
    if not query_kw:
        return 0.0
    overlap = query_kw & record_kw
    # Weight by query coverage (how much of the query is matched)
    return len(overlap) / len(query_kw)


# ---------------------------------------------------------------------------
# ExperienceStore
# ---------------------------------------------------------------------------


class ExperienceStore:
    """Persistent store of plan-execution outcomes."""

    def __init__(self, base_dir: str = EXPERIENCE_DIR_DEFAULT, robot_type: str = "wheeled"):
        self._base_dir = base_dir
        self._robot_type = robot_type
        self._file_path = os.path.join(base_dir, f"{robot_type}.jsonl")
        self._cache: list[ExperienceRecord] = []
        # Parallel cache of pre-computed keyword sets per record. Built
        # once at load and updated on every record/cap/clear so query()
        # doesn't have to re-extract keywords for every record on every
        # call (was O(K*N) per query for K candidates × N records).
        self._kw_cache: list[set[str]] = []
        self._dirty = False
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def record(
        self,
        task: str,
        plan: list[dict],
        outcome: str,
        *,
        context: str = "",
        steps_executed: int = 0,
        error: str = "",
        interrupt_reason: str = "",
        duration_s: float = 0.0,
    ) -> ExperienceRecord:
        """Record a completed plan execution."""
        tags = sorted(_extract_keywords(task) | _extract_keywords(context))
        rec = ExperienceRecord(
            timestamp=time.time(),
            task=task,
            context=context,
            plan=plan,
            outcome=outcome,
            steps_total=len(plan),
            steps_executed=steps_executed,
            error=error,
            interrupt_reason=interrupt_reason,
            duration_s=duration_s,
            tags=tags,
        )
        self._cache.append(rec)
        self._kw_cache.append(set(rec.tags) | _extract_keywords(rec.task))
        self._enforce_cap()
        self._append_to_disk(rec)
        logger.info(
            "[Experience] Recorded: %s → %s (%d/%d steps)",
            task[:60],
            outcome,
            steps_executed,
            len(plan),
        )
        return rec

    def query(self, task: str, context: str = "", k: int = QUERY_TOP_K) -> list[ExperienceHit]:
        """Find the K most relevant past experiences for a task."""
        query_kw = _extract_keywords(task) | _extract_keywords(context)
        if not query_kw:
            return []

        scored: list[ExperienceHit] = []
        # Iterate in lock-step with the keyword cache built at record/load
        # time. Falls back to recomputing if the cache is somehow out of
        # sync (defensive).
        for i, rec in enumerate(self._cache):
            rec_kw = (
                self._kw_cache[i]
                if i < len(self._kw_cache)
                else (set(rec.tags) | _extract_keywords(rec.task))
            )
            score = _keyword_similarity(query_kw, rec_kw)
            if score >= MIN_SIMILARITY_SCORE:
                scored.append(ExperienceHit(record=rec, score=score))

        # Sort by score desc, then recency
        scored.sort(key=lambda h: (-h.score, -h.record.timestamp))
        return scored[:k]

    def format_for_prompt(self, hits: list[ExperienceHit]) -> str:
        """Format experience hits as text suitable for LLM prompt injection."""
        if not hits:
            return ""
        lines = ["Past experience (learn from these):"]
        for i, hit in enumerate(hits, 1):
            r = hit.record
            plan_summary = " → ".join(s.get("skill", "?") for s in r.plan)
            outcome_str = r.outcome.upper()
            if r.error:
                outcome_str += f" ({r.error})"
            if r.interrupt_reason:
                outcome_str += f" ({r.interrupt_reason})"
            lines.append(
                f'  {i}. Task: "{r.task}" | Plan: {plan_summary} | '
                f"Result: {outcome_str} ({r.steps_executed}/{r.steps_total} steps)"
            )
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._cache)

    @property
    def robot_type(self) -> str:
        return self._robot_type

    def update_robot_type(self, robot_type: str):
        """Switch to a different robot type's experience file."""
        if robot_type == self._robot_type:
            return
        self._robot_type = robot_type
        self._file_path = os.path.join(self._base_dir, f"{robot_type}.jsonl")
        self._cache = []
        self._kw_cache = []
        self._load()

    def success_rate(self, task_keywords: str = "") -> float:
        """Return success rate (0.0-1.0) for matching records, or overall."""
        records = self._cache
        if task_keywords:
            kw = _extract_keywords(task_keywords)
            records = [
                r
                for r in self._cache
                if _keyword_similarity(kw, set(r.tags)) >= MIN_SIMILARITY_SCORE
            ]
        if not records:
            return 0.0
        done = sum(1 for r in records if r.outcome == "done")
        return done / len(records)

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self):
        """Load records from disk into cache."""
        if not os.path.exists(self._file_path):
            return
        try:
            with open(self._file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    rec = ExperienceRecord(**raw)
                    self._cache.append(rec)
                    # Pre-compute keyword set in lock-step so query()
                    # doesn't have to recompute on every call.
                    self._kw_cache.append(set(rec.tags) | _extract_keywords(rec.task))
            logger.info("[Experience] Loaded %d records for %s", len(self._cache), self._robot_type)
        except Exception as e:
            logger.error("[Experience] Failed to load %s: %s", self._file_path, e)

    def _append_to_disk(self, rec: ExperienceRecord):
        """Append a single record to the JSONL file."""
        os.makedirs(self._base_dir, exist_ok=True)
        try:
            with open(self._file_path, "a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        except Exception as e:
            logger.error("[Experience] Write error: %s", e)

    def _enforce_cap(self):
        """FIFO eviction if over MAX_RECORDS."""
        if len(self._cache) > MAX_RECORDS:
            excess = len(self._cache) - MAX_RECORDS
            self._cache = self._cache[excess:]
            self._kw_cache = self._kw_cache[excess:]
            self._rewrite_disk()

    def _rewrite_disk(self):
        """Rewrite the full JSONL file from cache."""
        os.makedirs(self._base_dir, exist_ok=True)
        try:
            with open(self._file_path, "w") as f:
                for rec in self._cache:
                    f.write(json.dumps(asdict(rec)) + "\n")
        except Exception as e:
            logger.error("[Experience] Rewrite error: %s", e)
