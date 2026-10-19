# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Task planner — decomposes a free-text task description into a skill plan.

The LLM receives the skill catalog for the robot type and a task description,
then returns a JSON array of steps: [{"skill": "FORWARD", "args": {"speed": 60}}, ...]

Usage:
    planner = TaskPlanner(host, port, model, robot_type="wheeled")
    plan = planner.plan("patrol the room and report any unusual objects")
    # -> [{"skill": "SCAN_360"}, {"skill": "FORWARD", "args": {"speed": 50}}, ...]
"""

import json
import re
from typing import Optional

from openai import OpenAI
from planner.decide import LLM_TIMEOUT_S
from planner.skills import skill_list_prompt, get_skills
from planner.experience import ExperienceStore
from planner.meta import MetaReviewer

# Args-validation bounds for LLM-generated plan steps. The policy translator
# clamps again downstream (defense in depth), but a plan step should never
# carry a value so far out of range that it signals a broken/adversarial
# LLM response upstream of that layer.
_ARG_SPEED_MIN: int = -100
_ARG_SPEED_MAX: int = 100
_ARG_DEGREES_MIN: int = -360
_ARG_DEGREES_MAX: int = 360


_SYSTEM_TEMPLATE = """\
You are the task planner for an autonomous {robot_type} robot.
Your job: decompose a task description into an ordered list of skills.

Available skills:
{skill_list}

Respond with a JSON array ONLY — no prose, no markdown:
[
  {{"skill": "SKILL_NAME", "args": {{"arg1": value}}}},
  ...
]

Rules:
- Use ONLY the skills listed above. Do not invent new ones.
- args is optional — omit it for skills with no arguments.
- Keep the plan short (3-10 steps). Prefer simple, safe steps.
- If the task involves going somewhere, include NAVIGATE_TO with the location name.
- If unsure, use STOP as the last step.
- Return only valid JSON. No explanation.
{heuristics}
{experience}
"""


class TaskPlanner:
    """Decomposes a free-text task into a typed skill plan via LLM."""

    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        robot_type: str = "wheeled",
        experience: ExperienceStore | None = None,
        meta: MetaReviewer | None = None,
    ):
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="not-needed",
            timeout=LLM_TIMEOUT_S,
        )
        self.model = model
        self.robot_type = robot_type
        self._skill_list = skill_list_prompt(robot_type)
        self._known_skills = set(get_skills(robot_type).keys())
        self.experience = experience
        self.meta = meta

    def plan(self, task: str, context: str = "") -> list[dict]:
        """Decompose a task description into a skill plan.

        Args:
            task: Natural language task description.
            context: Optional extra context (current location, state, etc.).

        Returns:
            List of {"skill": str, "args": dict} dicts.
            Falls back to [{"skill": "STOP"}] on any error.
        """
        # Query past experience for similar tasks
        experience_text = ""
        if self.experience:
            hits = self.experience.query(task, context)
            experience_text = self.experience.format_for_prompt(hits)

        # Inject learned heuristics from meta-reviewer
        heuristics_text = ""
        if self.meta:
            heuristics_text = self.meta.rules_for_prompt()

        system = _SYSTEM_TEMPLATE.format(
            robot_type=self.robot_type,
            skill_list=self._skill_list,
            heuristics=heuristics_text,
            experience=experience_text,
        )
        user = task
        if context:
            user = f"Context: {context}\nTask: {task}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse(raw)
        except Exception as e:
            print(f"[TaskPlanner] Error: {e}")
            return [{"skill": "STOP"}]

    # Compile-once regex patterns (cls-level so multiple TaskPlanner
    # instances share them and we avoid the per-call `re.compile` cost,
    # which was visible in profiles since LLM responses arrive at task
    # rate and re.sub had to recompile each invocation).
    _RE_FENCE_START = re.compile(r"^```[a-z]*\n?")
    _RE_FENCE_END = re.compile(r"\n?```$")
    _RE_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)

    def _parse(self, raw: str) -> list[dict]:
        """Parse and validate the LLM JSON response."""
        raw = self._RE_FENCE_START.sub("", raw.strip())
        raw = self._RE_FENCE_END.sub("", raw.strip())

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract the first JSON array found
            m = self._RE_JSON_ARRAY.search(raw)
            if not m:
                return [{"skill": "STOP"}]
            try:
                plan = json.loads(m.group())
            except json.JSONDecodeError:
                return [{"skill": "STOP"}]

        if not isinstance(plan, list):
            return [{"skill": "STOP"}]

        validated = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            skill = step.get("skill", "")
            if skill not in self._known_skills:
                print(f"[TaskPlanner] Unknown skill '{skill}' — skipping")
                continue
            validated.append(
                {
                    "skill": skill,
                    "args": self._validate_args(step.get("args", {})),
                }
            )

        return validated if validated else [{"skill": "STOP"}]

    @staticmethod
    def _validate_args(args: object) -> dict:
        """Validate/clamp a plan step's args before they reach the executor.

        `args` comes straight from the LLM's JSON response — untrusted input.
        Only `speed`/`degrees` get range validation here (the two fields the
        policy translators turn directly into motor/PWM values); anything
        that fails to coerce to an int is dropped in favor of a safe default
        rather than propagated as a string/float/out-of-range surprise.
        """
        if not isinstance(args, dict):
            return {}
        out: dict = {}
        for key, value in args.items():
            if key == "speed":
                out[key] = TaskPlanner._clamp_int(value, _ARG_SPEED_MIN, _ARG_SPEED_MAX, default=0)
            elif key == "degrees":
                out[key] = TaskPlanner._clamp_int(
                    value, _ARG_DEGREES_MIN, _ARG_DEGREES_MAX, default=0
                )
            else:
                out[key] = value
        return out

    @staticmethod
    def _clamp_int(value: object, lo: int, hi: int, default: int) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    def update_robot_type(self, robot_type: str):
        """Hot-swap robot type (e.g. when STATUS packet changes it)."""
        self.robot_type = robot_type
        self._skill_list = skill_list_prompt(robot_type)
        self._known_skills = set(get_skills(robot_type).keys())
