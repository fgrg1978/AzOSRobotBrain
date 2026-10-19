# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/skills.py — skill catalog per robot type."""

from planner.skills import (
    get_skills,
    skill_list_prompt,
    UNIVERSAL_SKILLS,
    WHEELED_SKILLS,
    DRONE_SKILLS,
    HUMANOID_SKILLS,
    ROBOT_SKILLS,
)


class TestUniversalSkills:
    def test_stop_exists(self):
        assert "STOP" in UNIVERSAL_SKILLS

    def test_wait_exists(self):
        assert "WAIT" in UNIVERSAL_SKILLS

    def test_emergency_exists(self):
        assert "EMERGENCY" in UNIVERSAL_SKILLS

    def test_scan_360_exists(self):
        assert "SCAN_360" in UNIVERSAL_SKILLS

    def test_navigate_to_exists(self):
        assert "NAVIGATE_TO" in UNIVERSAL_SKILLS

    def test_investigate_exists(self):
        assert "INVESTIGATE" in UNIVERSAL_SKILLS

    def test_alert_exists(self):
        assert "ALERT" in UNIVERSAL_SKILLS

    def test_report_exists(self):
        assert "REPORT" in UNIVERSAL_SKILLS

    def test_all_have_description(self):
        for name, info in UNIVERSAL_SKILLS.items():
            assert "description" in info, f"{name} missing description"

    def test_all_have_args(self):
        for name, info in UNIVERSAL_SKILLS.items():
            assert "args" in info, f"{name} missing args"

    def test_all_have_example(self):
        for name, info in UNIVERSAL_SKILLS.items():
            assert "example" in info, f"{name} missing example"


class TestWheeledSkills:
    def test_forward(self):
        assert "FORWARD" in WHEELED_SKILLS

    def test_backward(self):
        assert "BACKWARD" in WHEELED_SKILLS

    def test_turn_left(self):
        assert "TURN_LEFT" in WHEELED_SKILLS

    def test_turn_right(self):
        assert "TURN_RIGHT" in WHEELED_SKILLS

    def test_follow_wall(self):
        assert "FOLLOW_WALL" in WHEELED_SKILLS

    def test_map_perimeter(self):
        assert "MAP_PERIMETER" in WHEELED_SKILLS

    def test_patrol_perimeter(self):
        assert "PATROL_PERIMETER" in WHEELED_SKILLS

    def test_deterrent(self):
        assert "DETERRENT" in WHEELED_SKILLS

    def test_aim_at(self):
        assert "AIM_AT" in WHEELED_SKILLS

    def test_stand_down(self):
        assert "STAND_DOWN" in WHEELED_SKILLS

    def test_investigate_zone(self):
        assert "INVESTIGATE_ZONE" in WHEELED_SKILLS

    def test_return_to_dock(self):
        assert "RETURN_TO_DOCK" in WHEELED_SKILLS

    def test_undock(self):
        assert "UNDOCK" in WHEELED_SKILLS

    def test_track_intruder(self):
        assert "TRACK_INTRUDER" in WHEELED_SKILLS


class TestDroneSkills:
    def test_hover(self):
        assert "HOVER" in DRONE_SKILLS

    def test_takeoff(self):
        assert "TAKEOFF" in DRONE_SKILLS

    def test_land(self):
        assert "LAND" in DRONE_SKILLS

    def test_return_home(self):
        assert "RETURN_HOME" in DRONE_SKILLS

    def test_fly_forward(self):
        assert "FLY_FORWARD" in DRONE_SKILLS


class TestHumanoidSkills:
    def test_stand(self):
        assert "STAND" in HUMANOID_SKILLS

    def test_walk_forward(self):
        assert "WALK_FORWARD" in HUMANOID_SKILLS

    def test_wave(self):
        assert "WAVE" in HUMANOID_SKILLS


class TestGetSkills:
    def test_wheeled(self):
        skills = get_skills("wheeled")
        assert "FORWARD" in skills
        assert "STOP" in skills  # universal included

    def test_drone(self):
        skills = get_skills("drone")
        assert "HOVER" in skills
        assert "STOP" in skills

    def test_humanoid(self):
        skills = get_skills("humanoid")
        assert "STAND" in skills
        assert "STOP" in skills

    def test_ackermann(self):
        skills = get_skills("ackermann")
        assert "FORWARD" in skills

    def test_unknown_defaults_to_wheeled(self):
        skills = get_skills("unknown_type")
        assert "FORWARD" in skills

    def test_case_insensitive(self):
        skills = get_skills("WHEELED")
        assert "FORWARD" in skills

    def test_no_overlap_universal_wheeled(self):
        for name in WHEELED_SKILLS:
            assert name not in UNIVERSAL_SKILLS


class TestRobotSkills:
    def test_four_types(self):
        assert "wheeled" in ROBOT_SKILLS
        assert "drone" in ROBOT_SKILLS
        assert "humanoid" in ROBOT_SKILLS
        assert "ackermann" in ROBOT_SKILLS

    def test_wheeled_includes_universal(self):
        for name in UNIVERSAL_SKILLS:
            assert name in ROBOT_SKILLS["wheeled"]


class TestSkillListPrompt:
    def test_wheeled_prompt(self):
        prompt = skill_list_prompt("wheeled")
        assert "FORWARD" in prompt
        assert "STOP" in prompt
        assert len(prompt) > 100

    def test_drone_prompt(self):
        prompt = skill_list_prompt("drone")
        assert "HOVER" in prompt
        assert "TAKEOFF" in prompt

    def test_prompt_has_descriptions(self):
        prompt = skill_list_prompt("wheeled")
        assert "#" in prompt  # comments with descriptions
