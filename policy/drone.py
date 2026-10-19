# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Drone policy translator — quad rotor.

Converts skill names into ActuatorCmd for a quadrotor drone.
Channels: [throttle, roll, pitch, yaw]  (PWM 1000-2000)

NOTE: The PID attitude controller runs onboard (kernel RT task).
      This translator sends high-level setpoints, not raw motor commands.
      Full flight control requires an onboard EKF and attitude PID first.
"""

from protocol import ActuatorCmd, ACT_QUAD_ROTOR, FLAG_EMERGENCY, FLAG_ALERT

# PWM range: 1000 (min) - 2000 (max), 1500 = neutral
PWM_MIN = 1000
PWM_NEUTRAL = 1500
PWM_MAX = 2000
PWM_DISARM = 900  # below arm threshold


class DronePolicy:
    """Policy translator for quadrotor drones (STUB — requires AH+AK).

    Until EKF + attitude PID are implemented, only basic commands work.
    """

    def __init__(self, hover_throttle: int = 1450, max_tilt_deg: int = 35):
        self.hover_throttle = hover_throttle
        self.max_tilt_deg = max_tilt_deg

    def translate(
        self, skill: str, args: dict | None = None, sensors: dict | None = None
    ) -> ActuatorCmd:
        """Translate a skill into quad rotor ActuatorCmd.

        Channels: [throttle, roll, pitch, yaw]
        """
        args = args or {}
        s = skill.strip().upper()

        if s in ("EMERGENCY", "E_STOP", "KILL_MOTORS"):
            # CAUTION: only use on ground or as last resort
            return ActuatorCmd.drone(
                PWM_DISARM, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL, flags=FLAG_EMERGENCY
            )

        if s == "STOP":
            # For drone: STOP = hover, not kill motors
            return self._hover()

        if s == "HOVER":
            return self._hover()

        if s == "TAKEOFF":
            alt = int(args.get("altitude_m", 5))
            # Send climb throttle — onboard alt PID will stabilize
            throttle = min(self.hover_throttle + 100 * alt, PWM_MAX - 50)
            return ActuatorCmd.drone(throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL)

        if s == "LAND":
            # Descend slowly — onboard will cut motors on touchdown
            throttle = max(self.hover_throttle - 150, PWM_MIN + 100)
            return ActuatorCmd.drone(throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL)

        if s == "RETURN_HOME":
            # RTH is handled by onboard mission fallback (AC2)
            # Send RTH flag via mode — placeholder
            return self._hover(flags=0)

        if s == "ASCEND":
            meters = int(args.get("meters", 2))
            throttle = min(self.hover_throttle + 100 * meters, PWM_MAX - 50)
            return ActuatorCmd.drone(throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL)

        if s == "DESCEND":
            meters = int(args.get("meters", 2))
            throttle = max(self.hover_throttle - 100 * meters, PWM_MIN + 100)
            return ActuatorCmd.drone(throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL)

        if s == "YAW_RIGHT":
            degrees = int(args.get("degrees", 45))
            yaw = min(PWM_NEUTRAL + degrees * 3, PWM_MAX)
            return ActuatorCmd.drone(self.hover_throttle, PWM_NEUTRAL, PWM_NEUTRAL, yaw)

        if s == "YAW_LEFT":
            degrees = int(args.get("degrees", 45))
            yaw = max(PWM_NEUTRAL - degrees * 3, PWM_MIN)
            return ActuatorCmd.drone(self.hover_throttle, PWM_NEUTRAL, PWM_NEUTRAL, yaw)

        if s in ("SCAN_360", "ORBIT"):
            return ActuatorCmd.drone(
                self.hover_throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL + 100
            )

        if s == "ALERT":
            return self._hover(flags=FLAG_ALERT)

        # Unknown — hover (safe for drone)
        return self._hover()

    def _hover(self, flags: int = 0) -> ActuatorCmd:
        return ActuatorCmd.drone(
            self.hover_throttle, PWM_NEUTRAL, PWM_NEUTRAL, PWM_NEUTRAL, flags=flags
        )
