# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Humanoid robot policy translator — joint angles.

Converts skill names into ActuatorCmd for a humanoid robot.
Channels: [joint_0_cdeg, joint_1_cdeg, ..., joint_N_cdeg] (centidegrees)

NOTE: Locomotion and manipulation require balance (ZMP), a gait generator and IK.
      This stub provides safe static poses and simple joint commands.
"""

from protocol import ActuatorCmd, ACT_HUMANOID, FLAG_EMERGENCY, FLAG_ALERT

# Default 12-DOF layout (per leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
N_JOINTS_DEFAULT = 12

# Safe poses (centidegrees) for 12-DOF humanoid
POSE_STAND = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
POSE_CROUCH = [0, 0, -3000, 6000, -3000, 0, 0, 0, -3000, 6000, -3000, 0]
POSE_SIT = [0, 0, -4500, 9000, -4500, 0, 0, 0, -4500, 9000, -4500, 0]


class HumanoidPolicy:
    """Policy translator for humanoid robots (STUB — requires AO+AP+AQ).

    Until balance/gait/IK are implemented, only static poses work.
    """

    def __init__(self, num_joints: int = N_JOINTS_DEFAULT):
        self.num_joints = num_joints
        self._stand_pose = POSE_STAND[:num_joints]
        self._crouch_pose = POSE_CROUCH[:num_joints]
        self._sit_pose = POSE_SIT[:num_joints]

    def translate(
        self, skill: str, args: dict | None = None, sensors: dict | None = None
    ) -> ActuatorCmd:
        """Translate a skill into humanoid ActuatorCmd (joint angles)."""
        args = args or {}
        s = skill.strip().upper()

        if s in ("EMERGENCY", "E_STOP"):
            # Safe: crouch + lock joints
            return self._pose(self._crouch_pose, flags=FLAG_EMERGENCY)

        if s == "STOP":
            # Humanoid stop = stay in current pose (send stand)
            return self._pose(self._stand_pose)

        if s == "STAND":
            return self._pose(self._stand_pose)

        if s == "CROUCH":
            return self._pose(self._crouch_pose)

        if s in ("SIT", "SIT_DOWN"):
            return self._pose(self._sit_pose)

        if s == "WAIT":
            return self._pose(self._stand_pose)

        if s == "ALERT":
            return self._pose(self._stand_pose, flags=FLAG_ALERT)

        # Skills below need balance, gait and IK — hold the stand pose
        if s in ("WALK_TO", "FORWARD", "BACKWARD"):
            # TODO: gait generator
            return self._pose(self._stand_pose)

        if s in ("TURN_LEFT", "TURN_RIGHT"):
            # TODO: yaw rotation gait
            return self._pose(self._stand_pose)

        if s in ("GRAB", "PICK_UP", "RELEASE"):
            # TODO: IK + grasp planner
            return self._pose(self._stand_pose)

        if s in ("LOOK_AT", "WAVE", "POINT"):
            # TODO: neck/arm IK
            return self._pose(self._stand_pose)

        if s == "SCAN_360":
            # TODO: rotate torso/head
            return self._pose(self._stand_pose)

        # Unknown — stand (safe)
        return self._pose(self._stand_pose)

    def _pose(self, angles: list[int], flags: int = 0) -> ActuatorCmd:
        return ActuatorCmd(
            actuator_type=ACT_HUMANOID,
            channels=angles[: self.num_joints],
            flags=flags,
        )
