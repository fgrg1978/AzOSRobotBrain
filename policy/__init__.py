# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Policy translator loader — returns the correct translator for the robot type."""

from protocol import ROBOT_WHEELED, ROBOT_DRONE, ROBOT_HUMANOID, ROBOT_TYPE_BY_NAME
from policy.wheeled import WheeledPolicy
from policy.drone import DronePolicy
from policy.humanoid import HumanoidPolicy


def get_translator(robot_type: int | str, config: dict | None = None):
    """Return the policy translator for the given robot type.

    Args:
        robot_type: Integer (ROBOT_* constant) or string ("wheeled", "drone", ...).
        config:     AzOSRobotBrain config dict (for per-type parameters).

    Returns:
        WheeledPolicy | DronePolicy | HumanoidPolicy
    """
    config = config or {}
    robot_cfg = config.get("robot", {})

    # Normalize string type. Uses protocol.ROBOT_TYPE_BY_NAME rather than a
    # local dict so policy selection and the server's safety-profile selection
    # can never disagree about what "ackermann" means.
    if isinstance(robot_type, str):
        robot_type = ROBOT_TYPE_BY_NAME.get(robot_type.lower(), ROBOT_WHEELED)

    if robot_type == ROBOT_WHEELED:
        whl = robot_cfg.get("wheeled", {})
        return WheeledPolicy(max_speed=whl.get("max_speed", 80))

    if robot_type == ROBOT_DRONE:
        drn = robot_cfg.get("drone", {})
        return DronePolicy(
            hover_throttle=drn.get("hover_throttle", 1450),
            max_tilt_deg=drn.get("max_tilt_deg", 35),
        )

    if robot_type == ROBOT_HUMANOID:
        hum = robot_cfg.get("humanoid", {})
        return HumanoidPolicy(num_joints=hum.get("num_joints", 12))

    # Ackermann: use wheeled policy (same 2-channel format, different semantics)
    whl = robot_cfg.get("wheeled", {})
    return WheeledPolicy(max_speed=whl.get("max_speed", 80))
