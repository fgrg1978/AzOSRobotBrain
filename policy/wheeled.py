# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Wheeled robot policy translator — diff drive.

Converts skill names + args into ActuatorCmd for a differential drive robot.
Channels: [speed_l, speed_r]  (-max_speed .. +max_speed)
"""

import re
from protocol import ActuatorCmd, ACT_DIFF_DRIVE, FLAG_EMERGENCY, FLAG_ALERT


def _extract_number(text: str, default: int = 50) -> int:
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else default


class WheeledPolicy:
    """Policy translator for differential drive robots."""

    def __init__(self, max_speed: int = 80):
        self.max_speed = max_speed

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        """Clamp `value` to the inclusive [lo, hi] range.

        Applied to every wheel/PWM channel before it reaches ActuatorCmd —
        the LLM/plan-supplied speed/degrees args are untrusted input and
        must never be able to exceed the hardware's safe motor envelope.
        """
        return max(lo, min(hi, value))

    def translate(
        self, skill: str, args: dict | None = None, sensors: dict | None = None
    ) -> ActuatorCmd:
        """Translate a skill name into an ActuatorCmd.

        Args:
            skill:   Skill name (FORWARD, TURN_LEFT, STOP, ...) — case insensitive.
            args:    Optional skill arguments (speed, degrees, duration...).
            sensors: Latest sensor readings (for reactive skills).

        Returns:
            ActuatorCmd for diff drive (type=0, 2 channels).
        """
        args = args or {}
        s = skill.strip().upper()

        if s in ("EMERGENCY", "E_STOP"):
            return ActuatorCmd.wheeled(0, 0, flags=FLAG_EMERGENCY)

        if s in ("STOP", "WAIT"):
            return ActuatorCmd.wheeled(0, 0)

        if s == "FORWARD":
            speed = self._clamp(int(args.get("speed", 60)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s in ("BACKWARD", "REVERSE"):
            speed = self._clamp(-int(args.get("speed", 30)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "TURN_RIGHT":
            degrees = int(args.get("degrees", 45))
            intensity = self._clamp(degrees * self.max_speed // 90, -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(intensity, -intensity)

        if s == "TURN_LEFT":
            degrees = int(args.get("degrees", 45))
            intensity = self._clamp(degrees * self.max_speed // 90, -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(-intensity, intensity)

        if s == "INVESTIGATE":
            speed = self._clamp(int(args.get("speed", 20)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "TRACK":
            speed = self._clamp(int(args.get("speed", 30)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "ALERT":
            return ActuatorCmd.wheeled(0, 0, flags=FLAG_ALERT)

        if s == "SCAN_360":
            # Rotate in place at low speed — caller manages timing per step
            speed = self._clamp(int(args.get("speed", 25)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, -speed)

        if s == "FOLLOW_WALL":
            # Slow forward — caller keeps wall at distance via sensor feedback
            speed = self._clamp(int(args.get("speed", 30)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s in ("MAP_PERIMETER", "PATROL_PERIMETER", "EXPLORE_FRONTIER"):
            # Navigation skills — default slow forward, actual steering
            # is handled by the mapper/path planner sending TURN/FORWARD
            speed = self._clamp(int(args.get("speed", 30)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "NAVIGATE_PATH":
            # Follow planned path — steering based on heading error
            speed = int(args.get("speed", 40))
            steer = int(args.get("steer", 0))  # -100..+100
            left = self._clamp(speed + steer, -self.max_speed, self.max_speed)
            right = self._clamp(speed - steer, -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(left, right)

        if s == "DETERRENT":
            # Slow advance toward intruder during deterrent
            speed = self._clamp(int(args.get("speed", 15)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s in ("AIM_AT", "STAND_DOWN"):
            # No motor movement — turret/deterrent handled by DeterrentManager
            return ActuatorCmd.wheeled(0, 0)

        if s == "INVESTIGATE_ZONE":
            # Navigate toward zone — default moderate speed
            speed = self._clamp(int(args.get("speed", 40)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "RETURN_TO_DOCK":
            # Slow approach toward dock
            speed = self._clamp(int(args.get("speed", 20)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        if s == "UNDOCK":
            # Drive forward off dock
            speed = self._clamp(int(args.get("speed", 30)), -self.max_speed, self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)

        # Unknown skill — safe default
        return ActuatorCmd.wheeled(0, 0)

    def from_text(self, action_text: str) -> ActuatorCmd:
        """Translate a free-text LLM action string (legacy / direct LLM output).

        Supports: "FORWARD 60", "TURN_RIGHT 45", "STOP", "ALERT ...", etc.
        """
        action = action_text.strip().upper()

        if "EMERGENCY" in action or "E_STOP" in action:
            return ActuatorCmd.wheeled(0, 0, flags=FLAG_EMERGENCY)
        if "STOP" in action:
            return ActuatorCmd.wheeled(0, 0)
        if "FORWARD" in action:
            speed = min(_extract_number(action, 60), self.max_speed)
            return ActuatorCmd.wheeled(speed, speed)
        if "BACKWARD" in action or "REVERSE" in action:
            speed = min(_extract_number(action, 30), self.max_speed)
            return ActuatorCmd.wheeled(-speed, -speed)
        if "TURN_RIGHT" in action or "RIGHT" in action:
            degrees = _extract_number(action, 45)
            intensity = min(degrees * self.max_speed // 90, self.max_speed)
            return ActuatorCmd.wheeled(intensity, -intensity)
        if "TURN_LEFT" in action or "LEFT" in action:
            degrees = _extract_number(action, 45)
            intensity = min(degrees * self.max_speed // 90, self.max_speed)
            return ActuatorCmd.wheeled(-intensity, intensity)
        if "INVESTIGATE" in action:
            return ActuatorCmd.wheeled(20, 20)
        if "ALERT" in action:
            return ActuatorCmd.wheeled(0, 0, flags=FLAG_ALERT)

        return ActuatorCmd.wheeled(0, 0)
