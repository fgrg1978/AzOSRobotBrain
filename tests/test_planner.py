# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/skills.py and planner/modes.py (no LLM required)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.skills import (
    get_skills,
    skill_list_prompt,
    UNIVERSAL_SKILLS,
    WHEELED_SKILLS,
    DRONE_SKILLS,
    HUMANOID_SKILLS,
    ROBOT_SKILLS,
)
from planner.modes import ModeManager, load_modes

# ── Skills ────────────────────────────────────────────────────────────────────


def test_universal_skills_present():
    for name in ("STOP", "WAIT", "ALERT", "EMERGENCY", "SCAN_360", "NAVIGATE_TO"):
        assert name in UNIVERSAL_SKILLS, f"Missing universal skill: {name}"


def test_wheeled_skills_present():
    for name in ("FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "FOLLOW_WALL"):
        assert name in WHEELED_SKILLS, f"Missing wheeled skill: {name}"


def test_drone_skills_present():
    for name in ("HOVER", "TAKEOFF", "LAND", "ASCEND", "DESCEND", "YAW_LEFT", "YAW_RIGHT"):
        assert name in DRONE_SKILLS, f"Missing drone skill: {name}"


def test_humanoid_skills_present():
    for name in ("STAND", "CROUCH", "SIT", "WALK_FORWARD", "WAVE"):
        assert name in HUMANOID_SKILLS, f"Missing humanoid skill: {name}"


def test_get_skills_wheeled():
    skills = get_skills("wheeled")
    assert "FORWARD" in skills
    assert "HOVER" not in skills
    assert "STOP" in skills


def test_get_skills_drone():
    skills = get_skills("drone")
    assert "HOVER" in skills
    assert "FORWARD" not in skills
    assert "STOP" in skills


def test_get_skills_humanoid():
    skills = get_skills("humanoid")
    assert "STAND" in skills
    assert "FORWARD" not in skills
    assert "STOP" in skills


def test_get_skills_unknown_defaults_wheeled():
    skills = get_skills("robot_type_doesnt_exist")
    assert "FORWARD" in skills


def test_skill_list_prompt_wheeled():
    prompt = skill_list_prompt("wheeled")
    assert "FORWARD" in prompt
    assert "STOP" in prompt
    # Each line starts with "- "
    for line in prompt.strip().split("\n"):
        assert line.startswith("- "), f"Bad prompt line: {line!r}"


def test_skill_list_prompt_drone():
    prompt = skill_list_prompt("drone")
    assert "HOVER" in prompt
    assert "TAKEOFF" in prompt


def test_skill_args_format():
    # Each skill with args must have (type, description) tuples
    for robot_type, skills in ROBOT_SKILLS.items():
        for name, info in skills.items():
            for arg_name, spec in info["args"].items():
                assert (
                    isinstance(spec, tuple) and len(spec) == 2
                ), f"{robot_type}.{name}.{arg_name}: args must be (type, description)"


def test_skill_example_present():
    for robot_type, skills in ROBOT_SKILLS.items():
        for name, info in skills.items():
            assert "example" in info, f"Missing example in {robot_type}.{name}"
            assert info["example"], f"Empty example in {robot_type}.{name}"


# ── ModeManager ────────────────────────────────────────────────────────────────

_CONFIG = {
    "tasks": {"default": "patrulla"},
    "modes": {
        "patrulla": {
            "skills": ["SCAN_360", "NAVIGATE_TO"],
            "loop": True,
            "detect": ["person", "obstacle"],
            "on_detect": ["notify"],
            "waypoints": ["A", "B", "C"],
            "planner": "scripted",
        },
        "explorar": {
            "skills": [],
            "loop": False,
            "detect": [],
            "on_detect": [],
            "planner": "llm",
        },
        "volver_base": {
            "skills": ["NAVIGATE_TO"],
            "loop": False,
            "detect": [],
            "on_detect": [],
        },
    },
}


def test_mode_manager_default():
    m = ModeManager(_CONFIG)
    assert m.current_name == "patrulla"
    assert m.current is not None


def test_mode_manager_next_skill_loops():
    m = ModeManager(_CONFIG)
    assert m.next_skill() == "SCAN_360"
    assert m.next_skill() == "NAVIGATE_TO"
    # Loop: back to start
    assert m.next_skill() == "SCAN_360"


def test_mode_manager_set_mode():
    m = ModeManager(_CONFIG)
    assert m.set_mode("explorar") is True
    assert m.current_name == "explorar"


def test_mode_manager_set_unknown_mode():
    m = ModeManager(_CONFIG)
    assert m.set_mode("nonexistent") is False
    assert m.current_name == "patrulla"  # unchanged


def test_mode_manager_resets_idx_on_switch():
    m = ModeManager(_CONFIG)
    m.next_skill()  # advance index
    m.set_mode("volver_base")
    assert m.next_skill() == "NAVIGATE_TO"


def test_mode_manager_no_loop():
    m = ModeManager(_CONFIG)
    m.set_mode("volver_base")
    assert m.next_skill() == "NAVIGATE_TO"
    # No loop: stays at end, returns None
    assert m.next_skill() is None


def test_mode_manager_detect():
    m = ModeManager(_CONFIG)
    assert m.should_detect("person") is True
    assert m.should_detect("fire") is False


def test_mode_manager_on_detect_actions():
    m = ModeManager(_CONFIG)
    actions = m.on_detect_actions()
    assert "notify" in actions


def test_mode_manager_uses_llm():
    m = ModeManager(_CONFIG)
    assert not m.uses_llm()
    m.set_mode("explorar")
    assert m.uses_llm()


def test_load_modes():
    modes = load_modes(_CONFIG)
    assert "patrulla" in modes
    assert modes["patrulla"].loop is True
    assert modes["patrulla"].waypoints == ["A", "B", "C"]


if __name__ == "__main__":
    # Skills
    test_universal_skills_present()
    test_wheeled_skills_present()
    test_drone_skills_present()
    test_humanoid_skills_present()
    test_get_skills_wheeled()
    test_get_skills_drone()
    test_get_skills_humanoid()
    test_get_skills_unknown_defaults_wheeled()
    test_skill_list_prompt_wheeled()
    test_skill_list_prompt_drone()
    test_skill_args_format()
    test_skill_example_present()
    # Modes
    test_mode_manager_default()
    test_mode_manager_next_skill_loops()
    test_mode_manager_set_mode()
    test_mode_manager_set_unknown_mode()
    test_mode_manager_resets_idx_on_switch()
    test_mode_manager_no_loop()
    test_mode_manager_detect()
    test_mode_manager_on_detect_actions()
    test_mode_manager_uses_llm()
    test_load_modes()
    print("All planner tests passed!")
