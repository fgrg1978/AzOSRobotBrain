# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Mode system — maps named modes to skill sequences and detection rules.

A mode defines:
  - Which skills to execute (patrol loop, guard, etc.)
  - What to detect and how to react
  - Whether to use LLM planning or a fixed script
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModeConfig:
    name: str
    skills: list[str]  # ordered skill names (for scripted modes)
    loop: bool = True  # repeat skill list on completion
    detect: list[str] = field(default_factory=list)  # classes to watch for
    on_detect: list[str] = field(default_factory=list)  # [notify, alert, stop, ...]
    waypoints: list[str] = field(default_factory=list)  # location names
    scan_interval_s: Optional[float] = None
    planner: str = "scripted"  # scripted | llm
    schedule: str = "always"  # always | <cron expression>
    # Guard-specific settings
    sensors: list[str] = field(default_factory=list)  # [pir, sound, ir]
    perimeter_only: bool = False  # ignore interior detections
    patrol_speed_pct: int = 0  # override patrol speed (0 = use default)
    continuous_buzzer: bool = False  # non-stop buzzer in this mode
    continuous_led: bool = False  # non-stop LED in this mode
    led_on_detect: bool = False  # turn on LED on detection (night mode)
    track_intruder: bool = False  # auto-track detected persons


# ── Factory ───────────────────────────────────────────────────────────────────


def load_modes(config: dict) -> dict[str, ModeConfig]:
    """Parse the 'modes' section from config.yaml into ModeConfig objects."""
    modes: dict[str, ModeConfig] = {}
    for name, raw in config.get("modes", {}).items():
        modes[name] = ModeConfig(
            name=name,
            skills=raw.get("skills", []),
            loop=raw.get("loop", True),
            detect=raw.get("detect", []),
            on_detect=raw.get("on_detect", []),
            waypoints=raw.get("waypoints", []),
            scan_interval_s=raw.get("scan_interval_s"),
            planner=raw.get("planner", "scripted"),
            schedule=raw.get("schedule", "always"),
            sensors=raw.get("sensors", []),
            perimeter_only=raw.get("perimeter_only", False),
            patrol_speed_pct=raw.get("patrol_speed_pct", 0),
            continuous_buzzer=raw.get("continuous_buzzer", False),
            continuous_led=raw.get("continuous_led", False),
            led_on_detect=raw.get("led_on_detect", False),
            track_intruder=raw.get("track_intruder", False),
        )
    return modes


class ModeManager:
    """Tracks the current mode and handles transitions."""

    def __init__(self, config: dict):
        self.modes = load_modes(config)
        default_task = config.get("tasks", {}).get("default", "patrulla")
        self._current = default_task
        self._skill_idx = 0

    @property
    def current(self) -> Optional[ModeConfig]:
        return self.modes.get(self._current)

    @property
    def current_name(self) -> str:
        return self._current

    def set_mode(self, name: str) -> bool:
        """Switch to a named mode. Returns False if unknown."""
        if name not in self.modes:
            return False
        self._current = name
        self._skill_idx = 0
        return True

    def next_skill(self) -> Optional[str]:
        """Return the next skill to execute in scripted mode, advancing the cursor."""
        mode = self.current
        if mode is None or not mode.skills:
            return None
        if self._skill_idx >= len(mode.skills):
            return None

        skill = mode.skills[self._skill_idx]
        self._skill_idx += 1

        if self._skill_idx >= len(mode.skills):
            if mode.loop:
                self._skill_idx = 0
            else:
                self._skill_idx = len(mode.skills)  # stay at end

        return skill

    def should_detect(self, label: str) -> bool:
        """Return True if the current mode monitors the given detection label."""
        mode = self.current
        return mode is not None and label in mode.detect

    def on_detect_actions(self) -> list[str]:
        """Return the list of actions to take when detection fires."""
        mode = self.current
        return mode.on_detect if mode else []

    def uses_llm(self) -> bool:
        """Return True if current mode uses LLM planner instead of scripted."""
        mode = self.current
        return mode is not None and mode.planner == "llm"

    def __repr__(self) -> str:
        return f"ModeManager(current={self._current!r}, idx={self._skill_idx})"
