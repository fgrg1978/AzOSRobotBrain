# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for executor/skill_runner.py — no network, no LLM."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol import ActuatorCmd, ACT_DIFF_DRIVE, FLAG_EMERGENCY
from policy.wheeled import WheeledPolicy
from policy.drone import DronePolicy
from executor.skill_runner import SkillRunner, RunnerState

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_runner(policy=None):
    """Return a SkillRunner with a collecting send_cmd."""
    policy = policy or WheeledPolicy()
    sent: list[ActuatorCmd] = []

    async def collect(cmd: ActuatorCmd):
        sent.append(cmd)

    runner = SkillRunner(policy, collect)
    return runner, sent


def run(coro):
    return asyncio.run(coro)


# ── Basic execution ────────────────────────────────────────────────────────────


def test_empty_plan_completes():
    runner, sent = make_runner()
    state = run(runner.execute_plan([]))
    assert state == RunnerState.DONE
    assert sent == []


def test_stop_step_sends_stop():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "STOP"}]))
    assert len(sent) == 1
    assert sent[0].channels == [0, 0]


def test_forward_step_sends_movement():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "FORWARD", "args": {"speed": 60}}]))
    assert len(sent) == 1
    assert sent[0].channels[0] == 60
    assert sent[0].channels[1] == 60


def test_multi_step_plan():
    runner, sent = make_runner()
    plan = [
        {"skill": "FORWARD", "args": {"speed": 50}},
        {"skill": "TURN_RIGHT", "args": {"degrees": 90}},
        {"skill": "STOP"},
    ]
    state = run(runner.execute_plan(plan))
    assert state == RunnerState.DONE
    assert len(sent) == 3
    assert runner.steps_executed == 3


def test_wait_step():
    runner, sent = make_runner()
    import time

    t0 = time.monotonic()
    run(runner.execute_plan([{"skill": "WAIT", "args": {"seconds": 0.1}}]))
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.08  # slightly under 0.1 is ok due to OS scheduling


def test_emergency_sets_flag():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "EMERGENCY"}]))
    assert sent[0].flags == FLAG_EMERGENCY


def test_alert_step():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "ALERT", "args": {"message": "test"}}]))
    assert sent[0].flags != 0  # FLAG_ALERT


# ── Interrupt ──────────────────────────────────────────────────────────────────


def test_interrupt_stops_plan():
    policy = WheeledPolicy()
    sent = []
    step_count = [0]

    async def slow_send(cmd):
        sent.append(cmd)

    runner = SkillRunner(policy, slow_send)

    async def run_with_interrupt():
        # Schedule interrupt after first step completes
        async def interrupter():
            await asyncio.sleep(0.05)
            runner.interrupt("test interrupt")

        asyncio.create_task(interrupter())
        plan = [
            {"skill": "WAIT", "args": {"seconds": 0.2}},
            {"skill": "FORWARD", "args": {"speed": 60}},
            {"skill": "FORWARD", "args": {"speed": 60}},
        ]
        return await runner.execute_plan(plan)

    state = asyncio.run(run_with_interrupt())
    assert state == RunnerState.INTERRUPTED
    assert runner.steps_executed < 3


def test_interrupt_after_plan_has_no_effect():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "STOP"}]))
    runner.interrupt("late interrupt")  # should not crash
    assert runner.state == RunnerState.DONE


# ── execute_one ────────────────────────────────────────────────────────────────


def test_execute_one_returns_cmd():
    runner, sent = make_runner()
    cmd = run(runner.execute_one("FORWARD", {"speed": 80}))
    assert cmd.channels[0] == 80
    assert len(sent) == 1


def test_execute_one_does_not_change_plan():
    runner, sent = make_runner()
    run(runner.execute_one("STOP"))
    assert runner.current_plan == []


# ── Step duration ──────────────────────────────────────────────────────────────


def test_turn_duration_scales_with_degrees():
    runner, _ = make_runner()
    d90 = runner._duration("TURN_RIGHT", {"degrees": 90})
    d180 = runner._duration("TURN_RIGHT", {"degrees": 180})
    assert d180 > d90


def test_wait_duration_from_args():
    runner, _ = make_runner()
    assert runner._duration("WAIT", {"seconds": 3.5}) == 3.5


def test_hover_duration_from_args():
    runner, _ = make_runner(DronePolicy())
    assert runner._duration("HOVER", {"seconds": 7}) == 7.0


def test_hover_duration_default():
    runner, _ = make_runner(DronePolicy())
    # seconds=0 → use default from table
    d = runner._duration("HOVER", {"seconds": 0})
    assert d == 5.0


# ── Drone policy ───────────────────────────────────────────────────────────────


def test_drone_runner_hover():
    runner, sent = make_runner(DronePolicy(hover_throttle=1450))
    run(runner.execute_plan([{"skill": "HOVER", "args": {"seconds": 0.05}}]))
    assert sent[0].channels[0] == 1450  # throttle at hover


def test_drone_runner_emergency_kills():
    runner, sent = make_runner(DronePolicy())
    run(runner.execute_plan([{"skill": "EMERGENCY"}]))
    assert sent[0].channels[0] < 1000  # disarmed
    assert sent[0].flags == FLAG_EMERGENCY


# ── on_step_done callback ─────────────────────────────────────────────────────


def test_on_step_done_called():
    policy = WheeledPolicy()
    done_calls = []

    async def noop(cmd):
        pass

    def on_done(skill, args, cmd):
        done_calls.append(skill)

    runner = SkillRunner(policy, noop, on_step_done=on_done)
    run(
        runner.execute_plan(
            [
                {"skill": "FORWARD", "args": {"speed": 50}},
                {"skill": "STOP"},
            ]
        )
    )
    assert done_calls == ["FORWARD", "STOP"]


# ── Clear / reuse ─────────────────────────────────────────────────────────────


def test_runner_reuse_after_done():
    runner, sent = make_runner()
    run(runner.execute_plan([{"skill": "STOP"}]))
    assert runner.state == RunnerState.DONE
    # Re-run: clear() is called internally
    run(runner.execute_plan([{"skill": "FORWARD", "args": {"speed": 30}}]))
    assert runner.state == RunnerState.DONE
    assert runner.steps_executed == 1  # reset on second run


if __name__ == "__main__":
    test_empty_plan_completes()
    test_stop_step_sends_stop()
    test_forward_step_sends_movement()
    test_multi_step_plan()
    test_wait_step()
    test_emergency_sets_flag()
    test_alert_step()
    test_interrupt_stops_plan()
    test_interrupt_after_plan_has_no_effect()
    test_execute_one_returns_cmd()
    test_execute_one_does_not_change_plan()
    test_turn_duration_scales_with_degrees()
    test_wait_duration_from_args()
    test_hover_duration_from_args()
    test_hover_duration_default()
    test_drone_runner_hover()
    test_drone_runner_emergency_kills()
    test_on_step_done_called()
    test_runner_reuse_after_done()
    print("All executor tests passed!")
