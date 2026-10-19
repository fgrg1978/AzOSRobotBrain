# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Translate LLM text actions into numeric motor commands."""

import re
from protocol import VelocityCmd

FLAG_EMERGENCY = 0x01
FLAG_ALERT = 0x02


def _extract_number(text: str, default: int = 50) -> int:
    """Extract first integer from text."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else default


def to_velocity_cmd(action_text: str, max_speed: int = 80) -> VelocityCmd:
    """Convert an LLM action string to a VelocityCmd.

    Args:
        action_text: Action from planner (e.g., "FORWARD 60", "TURN_RIGHT 45").
        max_speed: Maximum allowed speed percentage.

    Returns:
        VelocityCmd with speed_l, speed_r, flags.
    """
    action = action_text.strip().upper()

    if "EMERGENCY" in action or "E_STOP" in action:
        return VelocityCmd(speed_l=0, speed_r=0, flags=FLAG_EMERGENCY)

    if "STOP" in action:
        return VelocityCmd(speed_l=0, speed_r=0)

    if "FORWARD" in action:
        speed = min(_extract_number(action, 60), max_speed)
        return VelocityCmd(speed_l=speed, speed_r=speed)

    if "BACKWARD" in action or "REVERSE" in action:
        speed = min(_extract_number(action, 30), max_speed)
        return VelocityCmd(speed_l=-speed, speed_r=-speed)

    if "TURN_RIGHT" in action or "RIGHT" in action:
        degrees = _extract_number(action, 45)
        intensity = min(degrees * max_speed // 90, max_speed)
        return VelocityCmd(speed_l=intensity, speed_r=-intensity)

    if "TURN_LEFT" in action or "LEFT" in action:
        degrees = _extract_number(action, 45)
        intensity = min(degrees * max_speed // 90, max_speed)
        return VelocityCmd(speed_l=-intensity, speed_r=intensity)

    if "INVESTIGATE" in action:
        # Slow forward while investigating
        return VelocityCmd(speed_l=20, speed_r=20)

    if "ALERT" in action:
        return VelocityCmd(speed_l=0, speed_r=0, flags=FLAG_ALERT)

    # Unknown action — safe default: stop
    return VelocityCmd(speed_l=0, speed_r=0)
