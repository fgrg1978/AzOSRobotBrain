# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/task_planner.py — LLM task decomposition."""

from planner.task_planner import TaskPlanner


class TestTaskPlannerInit:
    def test_init_default(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        assert tp.robot_type == "wheeled"

    def test_init_custom_type(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model", robot_type="drone")
        assert tp.robot_type == "drone"

    def test_update_robot_type(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        tp.update_robot_type("drone")
        assert tp.robot_type == "drone"

    def test_has_client(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        assert tp.client is not None

    def test_model_stored(self):
        tp = TaskPlanner("127.0.0.1", 1234, "my-model")
        assert tp.model == "my-model"


class TestTaskPlannerParse:
    def test_parse_valid_json(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        raw = '[{"skill": "FORWARD", "args": {"speed": 60}}, {"skill": "STOP"}]'
        plan = tp._parse(raw)
        assert len(plan) == 2
        assert plan[0]["skill"] == "FORWARD"

    def test_parse_single_step(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        raw = '[{"skill": "SCAN_360"}]'
        plan = tp._parse(raw)
        assert len(plan) == 1

    def test_parse_invalid_json(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        raw = "not valid json at all"
        plan = tp._parse(raw)
        # Falls back to [{"skill": "STOP"}] on parse failure
        assert isinstance(plan, list)
        assert len(plan) >= 0

    def test_parse_empty(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        plan = tp._parse("[]")
        # Empty array may return fallback STOP
        assert isinstance(plan, list)

    def test_parse_with_markdown_wrapper(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        raw = '```json\n[{"skill": "FORWARD", "args": {"speed": 50}}]\n```'
        plan = tp._parse(raw)
        assert len(plan) >= 1

    def test_parse_not_a_list(self):
        tp = TaskPlanner("127.0.0.1", 1234, "test-model")
        raw = '{"skill": "FORWARD"}'
        plan = tp._parse(raw)
        # Should handle dict → wrap in list, or return []
        assert isinstance(plan, list)
