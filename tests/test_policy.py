# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for policy translators (wheeled, drone, humanoid)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol import (
    ActuatorCmd,
    ACT_DIFF_DRIVE,
    ACT_QUAD_ROTOR,
    ACT_HUMANOID,
    FLAG_EMERGENCY,
    FLAG_ALERT,
    ROBOT_WHEELED,
    ROBOT_DRONE,
    ROBOT_HUMANOID,
)
from policy.wheeled import WheeledPolicy
from policy.drone import DronePolicy
from policy.humanoid import HumanoidPolicy
from policy import get_translator

# Keep backward-compat import working
from policy.actions import to_velocity_cmd

# ── WheeledPolicy ─────────────────────────────────────────────────────────────


def test_wheeled_stop():
    p = WheeledPolicy()
    cmd = p.translate("STOP")
    assert cmd.channels == [0, 0]
    assert cmd.flags == 0


def test_wheeled_forward():
    p = WheeledPolicy(max_speed=80)
    cmd = p.translate("FORWARD", {"speed": 60})
    assert cmd.channels[0] == 60
    assert cmd.channels[1] == 60
    assert cmd.actuator_type == ACT_DIFF_DRIVE


def test_wheeled_forward_clamp():
    p = WheeledPolicy(max_speed=80)
    cmd = p.translate("FORWARD", {"speed": 200})
    assert cmd.channels[0] == 80


def test_wheeled_turn_right():
    p = WheeledPolicy()
    cmd = p.translate("TURN_RIGHT", {"degrees": 90})
    assert cmd.channels[0] > 0
    assert cmd.channels[1] < 0


def test_wheeled_turn_left():
    p = WheeledPolicy()
    cmd = p.translate("TURN_LEFT", {"degrees": 45})
    assert cmd.channels[0] < 0
    assert cmd.channels[1] > 0


def test_wheeled_backward():
    p = WheeledPolicy()
    cmd = p.translate("BACKWARD", {"speed": 30})
    assert cmd.channels[0] == -30
    assert cmd.channels[1] == -30


def test_wheeled_alert():
    p = WheeledPolicy()
    cmd = p.translate("ALERT")
    assert cmd.channels == [0, 0]
    assert cmd.flags == FLAG_ALERT


def test_wheeled_emergency():
    p = WheeledPolicy()
    cmd = p.translate("EMERGENCY")
    assert cmd.flags == FLAG_EMERGENCY


def test_wheeled_unknown_defaults_stop():
    p = WheeledPolicy()
    cmd = p.translate("DANCE around the room")
    assert cmd.channels == [0, 0]


def test_wheeled_from_text():
    p = WheeledPolicy(max_speed=80)
    cmd = p.from_text("FORWARD 60")
    assert cmd.channels[0] == 60
    cmd2 = p.from_text("TURN_RIGHT 45")
    assert cmd2.channels[0] > 0 and cmd2.channels[1] < 0


# ── DronePolicy ───────────────────────────────────────────────────────────────


def test_drone_hover():
    p = DronePolicy(hover_throttle=1450)
    cmd = p.translate("HOVER")
    assert cmd.actuator_type == ACT_QUAD_ROTOR
    assert cmd.channels[0] == 1450  # throttle
    assert cmd.channels[1] == 1500  # roll neutral
    assert cmd.channels[2] == 1500  # pitch neutral


def test_drone_stop_is_hover():
    p = DronePolicy()
    cmd = p.translate("STOP")
    assert cmd.actuator_type == ACT_QUAD_ROTOR
    assert cmd.channels[0] > 900  # not killed — hovering


def test_drone_emergency_kills():
    p = DronePolicy()
    cmd = p.translate("EMERGENCY")
    assert cmd.flags == FLAG_EMERGENCY
    assert cmd.channels[0] < 1000  # disarmed


def test_drone_yaw_right():
    p = DronePolicy()
    cmd = p.translate("YAW_RIGHT", {"degrees": 45})
    assert cmd.channels[3] > 1500  # yaw > neutral


def test_drone_yaw_left():
    p = DronePolicy()
    cmd = p.translate("YAW_LEFT", {"degrees": 45})
    assert cmd.channels[3] < 1500  # yaw < neutral


def test_drone_ascend():
    p = DronePolicy(hover_throttle=1450)
    cmd = p.translate("ASCEND", {"meters": 2})
    assert cmd.channels[0] > 1450  # throttle > hover


def test_drone_descend():
    p = DronePolicy(hover_throttle=1450)
    cmd = p.translate("DESCEND", {"meters": 2})
    assert cmd.channels[0] < 1450  # throttle < hover


# ── HumanoidPolicy ────────────────────────────────────────────────────────────


def test_humanoid_stand():
    p = HumanoidPolicy(num_joints=12)
    cmd = p.translate("STAND")
    assert cmd.actuator_type == ACT_HUMANOID
    assert len(cmd.channels) == 12


def test_humanoid_crouch():
    p = HumanoidPolicy()
    cmd = p.translate("CROUCH")
    # Crouch should flex knees (non-zero angles)
    assert any(a != 0 for a in cmd.channels)


def test_humanoid_emergency_is_crouch():
    p = HumanoidPolicy()
    cmd = p.translate("EMERGENCY")
    assert cmd.flags == FLAG_EMERGENCY
    assert cmd.actuator_type == ACT_HUMANOID


def test_humanoid_unknown_is_stand():
    p = HumanoidPolicy()
    cmd = p.translate("BREAKDANCE")
    assert cmd.channels == p._stand_pose


# ── get_translator ─────────────────────────────────────────────────────────────


def test_get_translator_wheeled():
    t = get_translator(ROBOT_WHEELED)
    assert isinstance(t, WheeledPolicy)


def test_get_translator_drone():
    t = get_translator(ROBOT_DRONE)
    assert isinstance(t, DronePolicy)


def test_get_translator_humanoid():
    t = get_translator(ROBOT_HUMANOID)
    assert isinstance(t, HumanoidPolicy)


def test_get_translator_string():
    assert isinstance(get_translator("wheeled"), WheeledPolicy)
    assert isinstance(get_translator("drone"), DronePolicy)
    assert isinstance(get_translator("humanoid"), HumanoidPolicy)


def test_get_translator_config():
    config = {"robot": {"wheeled": {"max_speed": 50}}}
    t = get_translator("wheeled", config)
    assert isinstance(t, WheeledPolicy)
    assert t.max_speed == 50


# ── Backward compat: policy.actions ──────────────────────────────────────────


def test_compat_stop():
    cmd = to_velocity_cmd("STOP")
    assert cmd.speed_l == 0 and cmd.speed_r == 0


def test_compat_forward():
    cmd = to_velocity_cmd("FORWARD 60")
    assert cmd.speed_l == 60 and cmd.speed_r == 60


def test_compat_turn_right():
    cmd = to_velocity_cmd("TURN_RIGHT 90")
    assert cmd.speed_l > 0 and cmd.speed_r < 0


def test_compat_alert():
    from policy.actions import FLAG_ALERT

    cmd = to_velocity_cmd("ALERT intruder")
    assert cmd.flags == FLAG_ALERT


if __name__ == "__main__":
    # Wheeled
    test_wheeled_stop()
    test_wheeled_forward()
    test_wheeled_forward_clamp()
    test_wheeled_turn_right()
    test_wheeled_turn_left()
    test_wheeled_backward()
    test_wheeled_alert()
    test_wheeled_emergency()
    test_wheeled_unknown_defaults_stop()
    test_wheeled_from_text()
    # Drone
    test_drone_hover()
    test_drone_stop_is_hover()
    test_drone_emergency_kills()
    test_drone_yaw_right()
    test_drone_yaw_left()
    test_drone_ascend()
    test_drone_descend()
    # Humanoid
    test_humanoid_stand()
    test_humanoid_crouch()
    test_humanoid_emergency_is_crouch()
    test_humanoid_unknown_is_stand()
    # get_translator
    test_get_translator_wheeled()
    test_get_translator_drone()
    test_get_translator_humanoid()
    test_get_translator_string()
    test_get_translator_config()
    # Backward compat
    test_compat_stop()
    test_compat_forward()
    test_compat_turn_right()
    test_compat_alert()
    print("All policy tests passed!")
