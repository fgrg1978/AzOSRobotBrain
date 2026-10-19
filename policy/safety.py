# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Safety profiles per robot type.

Each robot type has different safety constraints. Drones are strictly safer
than wheeled robots (lives at stake). Humanoids have torque and fall limits.

All thresholds come from config or named defaults — NO magic numbers.
"""

import math
import logging
from dataclasses import dataclass

from protocol import ROBOT_WHEELED, ROBOT_DRONE, ROBOT_HUMANOID, ROBOT_ACKERMANN

logger = logging.getLogger("brain.safety")

# ── Default thresholds (named constants) ─────────────────────────────────────

# Common defaults
DEFAULT_MIN_BATTERY_MV = 6500
DEFAULT_MAX_TILT_CDEG = 4500  # 45 degrees in centidegrees
DEFAULT_COMMS_TIMEOUT_S = 5.0

# Wheeled defaults
DEFAULT_OBSTACLE_STOP_MM = 200
DEFAULT_MAX_SPEED_PCT = 80

# Drone defaults (stricter)
DEFAULT_DRONE_MIN_BATTERY_MV = 7000  # drones need more margin
DEFAULT_DRONE_MAX_ALTITUDE_M = 50.0
DEFAULT_DRONE_MIN_SATELLITES = 6
DEFAULT_DRONE_MAX_WIND_MS = 10.0
DEFAULT_DRONE_LOW_BATTERY_RTL_MV = 7000
DEFAULT_DRONE_CRITICAL_BATTERY_LAND_MV = 6500
DEFAULT_DRONE_MAX_TILT_CDEG = 3500  # 35 degrees — stricter than wheeled
DEFAULT_DRONE_COMMS_TIMEOUT_S = 3.0  # shorter — drone must react fast

# Humanoid defaults
DEFAULT_MAX_JOINT_TORQUE_PCT = 70
DEFAULT_FALL_DETECT_THRESHOLD = 4000  # IMU accel threshold (mg)

# Ackermann defaults
DEFAULT_ACKERMANN_MAX_STEER_CDEG = 3500  # 35 degrees max steering angle


@dataclass
class SafetyProfile:
    """Safety constraints for a specific robot type.

    All values are loaded from config with sane defaults.
    Safety checks are ALWAYS active — never bypassed.
    """

    # ── Common (all robot types) ─────────────────────────────────────────────
    min_battery_mv: int
    max_tilt_cdeg: int  # IMU tilt threshold (centidegrees)
    comms_timeout_s: float  # max time without brain contact

    # ── Wheeled / Ackermann ──────────────────────────────────────────────────
    obstacle_stop_mm: int  # front rangefinder threshold
    max_speed_pct: int  # speed limiter (0-100)

    # ── Drone ────────────────────────────────────────────────────────────────
    max_altitude_m: float  # geofence ceiling
    min_satellites: int  # GPS fix quality for flight
    max_wind_ms: float  # abort threshold (m/s)
    low_battery_rtl_mv: int  # trigger Return-To-Launch
    critical_battery_land_mv: int  # trigger immediate landing

    # ── Humanoid ─────────────────────────────────────────────────────────────
    max_joint_torque_pct: int  # torque limiter (0-100)
    fall_detect_threshold: int  # IMU accel magnitude threshold (mg)

    # ── Ackermann ────────────────────────────────────────────────────────────
    max_steer_cdeg: int  # max steering angle (centidegrees)

    # ── Robot type this profile was created for ──────────────────────────────
    robot_type: int = ROBOT_WHEELED

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def wheeled(cls, config: dict) -> "SafetyProfile":
        """Create safety profile for wheeled (differential drive) robots."""
        safety = _resolve_safety_section(config, "wheeled")
        return cls(
            robot_type=ROBOT_WHEELED,
            # Common
            min_battery_mv=safety.get("min_battery_mv", DEFAULT_MIN_BATTERY_MV),
            max_tilt_cdeg=safety.get("max_tilt_cdeg", DEFAULT_MAX_TILT_CDEG),
            comms_timeout_s=safety.get("comms_timeout_s", DEFAULT_COMMS_TIMEOUT_S),
            # Wheeled
            obstacle_stop_mm=safety.get("obstacle_stop_mm", DEFAULT_OBSTACLE_STOP_MM),
            max_speed_pct=safety.get("max_speed_pct", DEFAULT_MAX_SPEED_PCT),
            # Drone (not used, but fields required)
            max_altitude_m=0,
            min_satellites=0,
            max_wind_ms=0,
            low_battery_rtl_mv=0,
            critical_battery_land_mv=0,
            # Humanoid (not used)
            max_joint_torque_pct=0,
            fall_detect_threshold=0,
            # Ackermann (not used)
            max_steer_cdeg=0,
        )

    @classmethod
    def drone(cls, config: dict) -> "SafetyProfile":
        """Create safety profile for quadrotor drones (stricter than wheeled)."""
        safety = _resolve_safety_section(config, "drone")
        return cls(
            robot_type=ROBOT_DRONE,
            # Common — stricter defaults for drone
            min_battery_mv=safety.get("min_battery_mv", DEFAULT_DRONE_MIN_BATTERY_MV),
            max_tilt_cdeg=safety.get("max_tilt_cdeg", DEFAULT_DRONE_MAX_TILT_CDEG),
            comms_timeout_s=safety.get("comms_timeout_s", DEFAULT_DRONE_COMMS_TIMEOUT_S),
            # Wheeled (not used for drone, but fields required)
            obstacle_stop_mm=0,
            max_speed_pct=0,
            # Drone
            max_altitude_m=safety.get("max_altitude_m", DEFAULT_DRONE_MAX_ALTITUDE_M),
            min_satellites=safety.get("min_satellites", DEFAULT_DRONE_MIN_SATELLITES),
            max_wind_ms=safety.get("max_wind_ms", DEFAULT_DRONE_MAX_WIND_MS),
            low_battery_rtl_mv=safety.get("low_battery_rtl_mv", DEFAULT_DRONE_LOW_BATTERY_RTL_MV),
            critical_battery_land_mv=safety.get(
                "critical_battery_land_mv", DEFAULT_DRONE_CRITICAL_BATTERY_LAND_MV
            ),
            # Humanoid (not used)
            max_joint_torque_pct=0,
            fall_detect_threshold=0,
            # Ackermann (not used)
            max_steer_cdeg=0,
        )

    @classmethod
    def humanoid(cls, config: dict) -> "SafetyProfile":
        """Create safety profile for humanoid robots."""
        safety = _resolve_safety_section(config, "humanoid")
        return cls(
            robot_type=ROBOT_HUMANOID,
            # Common
            min_battery_mv=safety.get("min_battery_mv", DEFAULT_MIN_BATTERY_MV),
            max_tilt_cdeg=safety.get("max_tilt_cdeg", DEFAULT_MAX_TILT_CDEG),
            comms_timeout_s=safety.get("comms_timeout_s", DEFAULT_COMMS_TIMEOUT_S),
            # Wheeled (not used)
            obstacle_stop_mm=0,
            max_speed_pct=0,
            # Drone (not used)
            max_altitude_m=0,
            min_satellites=0,
            max_wind_ms=0,
            low_battery_rtl_mv=0,
            critical_battery_land_mv=0,
            # Humanoid
            max_joint_torque_pct=safety.get("max_joint_torque_pct", DEFAULT_MAX_JOINT_TORQUE_PCT),
            fall_detect_threshold=safety.get(
                "fall_detect_threshold", DEFAULT_FALL_DETECT_THRESHOLD
            ),
            # Ackermann (not used)
            max_steer_cdeg=0,
        )

    @classmethod
    def ackermann(cls, config: dict) -> "SafetyProfile":
        """Create safety profile for Ackermann steering robots."""
        safety = _resolve_safety_section(config, "ackermann")
        return cls(
            robot_type=ROBOT_ACKERMANN,
            # Common
            min_battery_mv=safety.get("min_battery_mv", DEFAULT_MIN_BATTERY_MV),
            max_tilt_cdeg=safety.get("max_tilt_cdeg", DEFAULT_MAX_TILT_CDEG),
            comms_timeout_s=safety.get("comms_timeout_s", DEFAULT_COMMS_TIMEOUT_S),
            # Wheeled (shared with Ackermann)
            obstacle_stop_mm=safety.get("obstacle_stop_mm", DEFAULT_OBSTACLE_STOP_MM),
            max_speed_pct=safety.get("max_speed_pct", DEFAULT_MAX_SPEED_PCT),
            # Drone (not used)
            max_altitude_m=0,
            min_satellites=0,
            max_wind_ms=0,
            low_battery_rtl_mv=0,
            critical_battery_land_mv=0,
            # Humanoid (not used)
            max_joint_torque_pct=0,
            fall_detect_threshold=0,
            # Ackermann
            max_steer_cdeg=safety.get("max_steer_cdeg", DEFAULT_ACKERMANN_MAX_STEER_CDEG),
        )

    @classmethod
    def for_robot_type(cls, robot_type: int, config: dict) -> "SafetyProfile":
        """Factory: create the correct SafetyProfile for a given ROBOT_* constant."""
        if robot_type == ROBOT_DRONE:
            return cls.drone(config)
        if robot_type == ROBOT_HUMANOID:
            return cls.humanoid(config)
        if robot_type == ROBOT_ACKERMANN:
            return cls.ackermann(config)
        return cls.wheeled(config)

    # ── Safety check methods ─────────────────────────────────────────────────

    def check_common(self, pkt) -> list[str]:
        """Check safety conditions common to ALL robot types.

        Returns a list of violation descriptions (empty = safe).
        """
        violations = []

        # Low battery
        if pkt.battery_mv < self.min_battery_mv:
            violations.append(f"Low battery: {pkt.battery_mv}mV < {self.min_battery_mv}mV minimum")

        # IMU tilt (if accel data available)
        if hasattr(pkt, "accel_mg") and pkt.accel_mg:
            tilt_cdeg = _tilt_from_accel(pkt.accel_mg)
            if tilt_cdeg > self.max_tilt_cdeg:
                violations.append(
                    f"Excessive tilt: {tilt_cdeg} cdeg > {self.max_tilt_cdeg} cdeg limit"
                )

        return violations

    def check_wheeled(self, pkt) -> list[str]:
        """Check wheeled-specific safety conditions."""
        violations = self.check_common(pkt)

        # Obstacle too close
        if hasattr(pkt, "range_front_mm") and pkt.range_front_mm < self.obstacle_stop_mm:
            violations.append(
                f"Obstacle at {pkt.range_front_mm}mm < {self.obstacle_stop_mm}mm threshold"
            )

        return violations

    def check_drone(self, pkt) -> list[str]:
        """Check drone-specific safety conditions (stricter).

        Returns tuple (violations, action) where action is one of:
          None — no special action
          "rtl" — trigger Return-To-Launch
          "land" — trigger immediate landing
        """
        violations = self.check_common(pkt)

        # Altitude check (geofence ceiling)
        if hasattr(pkt, "gps_alt_cm"):
            altitude_m = pkt.gps_alt_cm / _CM_PER_M
            if altitude_m > self.max_altitude_m:
                violations.append(f"Altitude {altitude_m:.1f}m > {self.max_altitude_m}m ceiling")

        # GPS satellite count — drone should not fly without good fix
        # Note: satellite count comes via SensorCompact or extended drone packet
        # For now check if GPS data looks invalid (lat/lon both zero)
        if hasattr(pkt, "gps_lat_deg7") and hasattr(pkt, "gps_lon_deg7"):
            if pkt.gps_lat_deg7 == 0 and pkt.gps_lon_deg7 == 0:
                violations.append(
                    f"No GPS fix (lat=0, lon=0) — min {self.min_satellites} satellites required"
                )

        # Critical battery — immediate land
        if pkt.battery_mv < self.critical_battery_land_mv:
            violations.append(
                f"CRITICAL battery: {pkt.battery_mv}mV < {self.critical_battery_land_mv}mV — LAND NOW"
            )
        # Low battery — RTL
        elif pkt.battery_mv < self.low_battery_rtl_mv:
            violations.append(
                f"Low battery: {pkt.battery_mv}mV < {self.low_battery_rtl_mv}mV — RTL triggered"
            )

        return violations

    def check_humanoid(self, pkt) -> list[str]:
        """Check humanoid-specific safety conditions."""
        violations = self.check_common(pkt)

        # Fall detection via IMU acceleration magnitude
        if hasattr(pkt, "accel_mg") and pkt.accel_mg:
            accel_magnitude = _accel_magnitude(pkt.accel_mg)
            if accel_magnitude > self.fall_detect_threshold:
                violations.append(
                    f"Fall detected: accel {accel_magnitude}mg > {self.fall_detect_threshold}mg threshold"
                )

        return violations

    def check_ackermann(self, pkt) -> list[str]:
        """Check Ackermann-specific safety conditions."""
        # Ackermann shares obstacle check with wheeled
        return self.check_wheeled(pkt)

    def check(self, pkt) -> list[str]:
        """Run all applicable safety checks for this profile's robot type.

        Returns list of violation descriptions (empty = safe).
        """
        if self.robot_type == ROBOT_DRONE:
            return self.check_drone(pkt)
        if self.robot_type == ROBOT_HUMANOID:
            return self.check_humanoid(pkt)
        if self.robot_type == ROBOT_ACKERMANN:
            return self.check_ackermann(pkt)
        return self.check_wheeled(pkt)

    def drone_action(self, pkt) -> str | None:
        """For drones: determine the emergency action to take.

        Returns:
            None   — no special action needed
            "land" — critical battery, land immediately
            "rtl"  — low battery, return to launch
            "stop" — other violation (hover/descend)
        """
        if self.robot_type != ROBOT_DRONE:
            return None

        # Critical battery takes highest priority
        if pkt.battery_mv < self.critical_battery_land_mv:
            return "land"

        # Low battery triggers RTL
        if pkt.battery_mv < self.low_battery_rtl_mv:
            return "rtl"

        # Any other violation -> hover/stop
        violations = self.check(pkt)
        if violations:
            return "stop"

        return None


# ── Helper functions ─────────────────────────────────────────────────────────

# Conversion constant
_CM_PER_M = 100


def _resolve_safety_section(config: dict, robot_type_str: str) -> dict:
    """Resolve the safety config section for a robot type.

    Supports both per-type config:
        safety:
          wheeled:
            obstacle_stop_mm: 200
          drone:
            max_altitude_m: 50

    And flat (legacy) config:
        safety:
          obstacle_stop_mm: 200
          min_battery_mv: 6500

    Falls back to flat safety section as wheeled defaults.
    """
    safety = config.get("safety", {})

    # Check for per-type subsection
    per_type = safety.get(robot_type_str)
    if isinstance(per_type, dict):
        return per_type

    # Fallback: use flat safety section (legacy/wheeled defaults)
    # Filter out any subsection dicts to get only scalar values
    return {k: v for k, v in safety.items() if not isinstance(v, dict)}


def _tilt_from_accel(accel_mg: tuple[int, int, int]) -> int:
    """Compute tilt angle in centidegrees from accelerometer data.

    Uses atan2 of horizontal vs vertical acceleration.
    accel_mg is (ax, ay, az) in milli-g.
    """
    ax, ay, az = accel_mg
    horizontal = math.sqrt(ax * ax + ay * ay)
    if az == 0 and horizontal == 0:
        return 0
    tilt_rad = math.atan2(horizontal, abs(az))
    tilt_cdeg = int(tilt_rad * _RAD_TO_CDEG)
    return tilt_cdeg


# Conversion: radians to centidegrees
_RAD_TO_CDEG = 18000 / math.pi  # (180 * 100) / pi


def _accel_magnitude(accel_mg: tuple[int, int, int]) -> int:
    """Compute acceleration magnitude in milli-g."""
    ax, ay, az = accel_mg
    return int(math.sqrt(ax * ax + ay * ay + az * az))
