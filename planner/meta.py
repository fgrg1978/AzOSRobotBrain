# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Meta-reviewer — self-improving planner heuristics via LLM reflection.

Inspired by Hyperagents (Zhang et al., 2026): the meta-agent periodically
reviews accumulated experience and distills it into heuristic rules that
improve the task planner's system prompt.

The loop:
  1. Collect recent experience records (successes + failures).
  2. Ask the LLM to analyse patterns and generate concise rules.
  3. Persist rules to ``data/experience/heuristics_{robot_type}.json``.
  4. TaskPlanner injects these rules into its system prompt.

This is the "metacognitive self-modification" layer: the robot improves
*how* it plans, not just *what* it plans.

Usage:
    reviewer = MetaReviewer(host, port, model, experience_store)
    rules = reviewer.review()          # blocking LLM call
    rules = reviewer.load_rules()      # from disk (no LLM)
    text  = reviewer.rules_for_prompt() # formatted for injection
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI
from planner.experience import ExperienceStore, ExperienceRecord

logger = logging.getLogger("brain.meta")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEURISTICS_DIR = "data/experience"
MIN_RECORDS_FOR_REVIEW = 5  # don't review with less than this
REVIEW_COOLDOWN_S = 3600  # min seconds between reviews (1 hour)
MAX_RULES = 20  # cap heuristic rules
REVIEW_WINDOW = 50  # consider last N records

_REVIEW_SYSTEM = """\
You are a meta-learning agent for an autonomous {robot_type} robot.
You are reviewing the robot's recent task execution history to extract
patterns and generate heuristic rules that will improve future planning.

Each record shows: task description, plan (sequence of skills), outcome
(done/interrupted/error), steps executed vs total, and any error message.

Analyse the records and output a JSON object with:
{{
  "rules": [
    {{
      "id": "rule_01",
      "rule": "short imperative rule (e.g. 'Always SCAN_360 before NAVIGATE_TO in unknown areas')",
      "reason": "one-line explanation based on evidence",
      "confidence": 0.0-1.0
    }}
  ]
}}

Guidelines:
- Focus on ACTIONABLE rules that change planning behavior.
- Look for: repeated failures, successful patterns, skill orderings that work.
- Prefer rules backed by multiple records over single incidents.
- Keep rules short and specific to this robot type.
- Maximum {max_rules} rules. Quality over quantity.
- Return ONLY valid JSON. No markdown, no prose.
"""

_REVIEW_USER = """\
Recent execution history ({n} records, {success_rate:.0%} success rate):

{records}

Generate or update heuristic rules based on this evidence.
"""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class HeuristicRule:
    """A single learned planning heuristic."""

    id: str
    rule: str
    reason: str
    confidence: float = 0.5
    created: float = 0.0
    last_reviewed: float = 0.0


@dataclass
class HeuristicSet:
    """Collection of heuristic rules for a robot type."""

    robot_type: str
    rules: list[HeuristicRule] = field(default_factory=list)
    last_review: float = 0.0
    review_count: int = 0


# ---------------------------------------------------------------------------
# MetaReviewer
# ---------------------------------------------------------------------------


class MetaReviewer:
    """LLM-powered meta-learning: reviews experience → generates heuristics."""

    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        experience: ExperienceStore,
        robot_type: str = "wheeled",
    ):
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="not-needed",
        )
        self.model = model
        self.experience = experience
        self.robot_type = robot_type
        # Where learned rules are persisted.
        #
        # An INSTANCE attribute, defaulting to the module constant, so a test
        # can point it at a temp directory. It used to read `HEURISTICS_DIR`
        # directly, which meant `tests/test_meta.py` wrote its fixtures into
        # the production path — `data/experience/heuristics_wheeled.json` was
        # a VERSIONED file that every test run overwrote, so `git status` came
        # back dirty after the suite and stopped meaning "I have unsaved work".
        # The file committed there held `"rule": "rule one", "reason": "r"`:
        # test detritus, not learned heuristics.
        self.heuristics_dir = HEURISTICS_DIR
        self._heuristics: HeuristicSet = HeuristicSet(robot_type=robot_type)
        self._load_rules()

    # ── Public API ─────────────────────────────────────────────────────────

    def review(self) -> list[HeuristicRule]:
        """Run a meta-review: LLM analyses recent experience → new rules.

        Returns the updated rule list. Persists to disk.
        Respects cooldown: returns existing rules if called too soon.
        """
        now = time.time()
        if now - self._heuristics.last_review < REVIEW_COOLDOWN_S:
            logger.info("[Meta] Cooldown active, skipping review")
            return self._heuristics.rules

        records = self.experience._cache[-REVIEW_WINDOW:]
        if len(records) < MIN_RECORDS_FOR_REVIEW:
            logger.info("[Meta] Not enough records (%d < %d)", len(records), MIN_RECORDS_FOR_REVIEW)
            return self._heuristics.rules

        # Format records for LLM
        records_text = self._format_records(records)
        success_rate = self.experience.success_rate()

        system = _REVIEW_SYSTEM.format(
            robot_type=self.robot_type,
            max_rules=MAX_RULES,
        )
        user = _REVIEW_USER.format(
            n=len(records),
            success_rate=success_rate,
            records=records_text,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            new_rules = self._parse_rules(raw)

            if new_rules:
                self._heuristics.rules = new_rules
                self._heuristics.last_review = now
                self._heuristics.review_count += 1
                self._save_rules()
                logger.info(
                    "[Meta] Review #%d: %d rules generated",
                    self._heuristics.review_count,
                    len(new_rules),
                )
            else:
                logger.warning("[Meta] Review produced no valid rules")

        except Exception as e:
            logger.error("[Meta] Review error: %s", e)

        return self._heuristics.rules

    def load_rules(self) -> list[HeuristicRule]:
        """Return current rules (from memory/disk, no LLM call)."""
        return self._heuristics.rules

    def rules_for_prompt(self) -> str:
        """Format current heuristics as text for LLM prompt injection."""
        rules = [r for r in self._heuristics.rules if r.confidence >= 0.3]
        if not rules:
            return ""
        lines = ["Learned heuristics (follow these):"]
        for r in rules:
            lines.append(f"  - {r.rule}")
        return "\n".join(lines)

    def should_review(self) -> bool:
        """Check if a review is due (enough records + cooldown elapsed)."""
        if self.experience.count < MIN_RECORDS_FOR_REVIEW:
            return False
        return time.time() - self._heuristics.last_review >= REVIEW_COOLDOWN_S

    @property
    def rule_count(self) -> int:
        return len(self._heuristics.rules)

    @property
    def review_count(self) -> int:
        return self._heuristics.review_count

    def update_robot_type(self, robot_type: str):
        """Switch robot type — reloads heuristics from disk."""
        if robot_type == self.robot_type:
            return
        self.robot_type = robot_type
        self._heuristics = HeuristicSet(robot_type=robot_type)
        self._load_rules()

    # ── Internal ──────────────────────────────────────────────────────────

    def _format_records(self, records: list[ExperienceRecord]) -> str:
        """Format records as text for the review prompt."""
        lines = []
        for i, r in enumerate(records, 1):
            plan_str = " → ".join(s.get("skill", "?") for s in r.plan)
            status = r.outcome.upper()
            if r.error:
                status += f": {r.error}"
            if r.interrupt_reason:
                status += f": {r.interrupt_reason}"
            lines.append(
                f'{i}. [{status}] "{r.task}" → {plan_str} '
                f"({r.steps_executed}/{r.steps_total} steps, {r.duration_s:.1f}s)"
            )
        return "\n".join(lines)

    def _parse_rules(self, raw: str) -> list[HeuristicRule]:
        """Parse LLM JSON response into HeuristicRule list."""
        import re

        # Strip markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(data, dict) or "rules" not in data:
            return []

        now = time.time()
        rules = []
        for entry in data["rules"][:MAX_RULES]:
            if not isinstance(entry, dict):
                continue
            rule_text = entry.get("rule", "").strip()
            if not rule_text:
                continue
            rules.append(
                HeuristicRule(
                    id=entry.get("id", f"rule_{len(rules)+1:02d}"),
                    rule=rule_text,
                    reason=entry.get("reason", ""),
                    confidence=min(1.0, max(0.0, float(entry.get("confidence", 0.5)))),
                    created=now,
                    last_reviewed=now,
                )
            )
        return rules

    def _heuristics_path(self) -> str:
        return os.path.join(self.heuristics_dir, f"heuristics_{self.robot_type}.json")

    def _load_rules(self):
        """Load heuristics from disk."""
        path = self._heuristics_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._heuristics.last_review = data.get("last_review", 0.0)
            self._heuristics.review_count = data.get("review_count", 0)
            for entry in data.get("rules", []):
                self._heuristics.rules.append(HeuristicRule(**entry))
            logger.info(
                "[Meta] Loaded %d rules for %s", len(self._heuristics.rules), self.robot_type
            )
        except Exception as e:
            logger.error("[Meta] Failed to load rules: %s", e)

    def _save_rules(self):
        """Persist heuristics to disk."""
        os.makedirs(self.heuristics_dir, exist_ok=True)
        path = self._heuristics_path()
        data = {
            "robot_type": self.robot_type,
            "last_review": self._heuristics.last_review,
            "review_count": self._heuristics.review_count,
            "rules": [asdict(r) for r in self._heuristics.rules],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("[Meta] Failed to save rules: %s", e)
