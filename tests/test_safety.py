# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for safety profiles (Phase AG)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol import (
    SensorPacket,
    SensorPacketDrone,
    SensorPacketHumanoid,
    ROBOT_WHEELED,
    ROBOT_DRONE,
    ROBOT_HUMANOID,
    ROBOT_ACKERMANN,
)
from policy.safety import (
    SafetyProfile,
    DEFAULT_MIN_BATTERY_MV,
    DEFAULT_OBSTACLE_STOP_MM,
    DEFAULT_DRONE_MIN_BATTERY_MV,
    DEFAULT_DRONE_MAX_ALTITUDE_M,
    DEFAULT_DRONE_LOW_BATTERY_RTL_MV,
    DEFAULT_DRONE_CRITICAL_BATTERY_LAND_MV,
    DEFAULT_DRONE_MIN_SATELLITES,
    DEFAULT_FALL_DETECT_THRESHOLD,
)

# ── Config fixtures ──────────────────────────────────────────────────────────

CONFIG_PER_TYPE = {
    "safety": {
        "wheeled": {
            "min_battery_mv": 6800,
            "obstacle_stop_mm": 250,
            "max_speed_pct": 70,
        },
        "drone": {
            "min_battery_mv": 7200,
            "max_altitude_m": 30.0,
            "min_satellites": 8,
            "low_battery_rtl_mv": 7200,
            "critical_battery_land_mv": 6600,
        },
        "humanoid": {
            "max_joint_torque_pct": 60,
            "fall_detect_threshold": 3500,
        },
    }
}

CONFIG_FLAT_LEGACY = {
    "safety": {
        "min_battery_mv": 6500,
        "obstacle_stop_mm": 200,
        "max_speed": 80,
    }
}

CONFIG_EMPTY = {}


# ── Sensor packet helpers ────────────────────────────────────────────────────


def _wheeled_pkt(battery_mv=8000, range_front_mm=500, accel_mg=(0, 0, 1000)):
    return SensorPacket(
        timestamp_ms=0,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=(0, 0, 0),
        odom_dist_mm=0,
        odom_hdg_cdeg=0,
        encoder_l=0,
        encoder_r=0,
        range_front_mm=range_front_mm,
        range_right_mm=500,
    )


def _drone_pkt(battery_mv=8000, gps_alt_cm=1000, gps_lat=1, gps_lon=1, accel_mg=(0, 0, 1000)):
    return SensorPacketDrone(
        timestamp_ms=0,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=(0, 0, 0),
        baro_pa=101325,
        mag_ut=(200, 50, 400),
        gps_lat_deg7=gps_lat,
        gps_lon_deg7=gps_lon,
        gps_alt_cm=gps_alt_cm,
        sonar_down_mm=100,
    )


def _humanoid_pkt(battery_mv=8000, accel_mg=(0, 0, 1000)):
    return SensorPacketHumanoid(
        timestamp_ms=0,
        battery_mv=battery_mv,
        accel_mg=accel_mg,
        gyro_mdps=(0, 0, 0),
        joint_angles=[0] * 12,
        foot_pressure_l=500,
        foot_pressure_r=500,
    )


# ── Factory method tests ─────────────────────────────────────────────────────


def test_wheeled_factory_defaults():
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    assert p.robot_type == ROBOT_WHEELED
    assert p.min_battery_mv == DEFAULT_MIN_BATTERY_MV
    assert p.obstacle_stop_mm == DEFAULT_OBSTACLE_STOP_MM


def test_wheeled_factory_config():
    p = SafetyProfile.wheeled(CONFIG_PER_TYPE)
    assert p.min_battery_mv == 6800
    assert p.obstacle_stop_mm == 250
    assert p.max_speed_pct == 70


def test_drone_factory_defaults():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    assert p.robot_type == ROBOT_DRONE
    assert p.min_battery_mv == DEFAULT_DRONE_MIN_BATTERY_MV
    assert p.max_altitude_m == DEFAULT_DRONE_MAX_ALTITUDE_M


def test_drone_factory_config():
    p = SafetyProfile.drone(CONFIG_PER_TYPE)
    assert p.min_battery_mv == 7200
    assert p.max_altitude_m == 30.0
    assert p.min_satellites == 8


def test_humanoid_factory_defaults():
    p = SafetyProfile.humanoid(CONFIG_EMPTY)
    assert p.robot_type == ROBOT_HUMANOID
    assert p.fall_detect_threshold == DEFAULT_FALL_DETECT_THRESHOLD


def test_humanoid_factory_config():
    p = SafetyProfile.humanoid(CONFIG_PER_TYPE)
    assert p.max_joint_torque_pct == 60
    assert p.fall_detect_threshold == 3500


def test_ackermann_factory():
    p = SafetyProfile.ackermann(CONFIG_EMPTY)
    assert p.robot_type == ROBOT_ACKERMANN
    assert p.obstacle_stop_mm == DEFAULT_OBSTACLE_STOP_MM


def test_for_robot_type_dispatches():
    assert SafetyProfile.for_robot_type(ROBOT_WHEELED, CONFIG_EMPTY).robot_type == ROBOT_WHEELED
    assert SafetyProfile.for_robot_type(ROBOT_DRONE, CONFIG_EMPTY).robot_type == ROBOT_DRONE
    assert SafetyProfile.for_robot_type(ROBOT_HUMANOID, CONFIG_EMPTY).robot_type == ROBOT_HUMANOID
    assert SafetyProfile.for_robot_type(ROBOT_ACKERMANN, CONFIG_EMPTY).robot_type == ROBOT_ACKERMANN


# ── Legacy flat config backward compat ───────────────────────────────────────


def test_flat_config_falls_back():
    """When config has flat safety: section (no per-type), use it as wheeled defaults."""
    p = SafetyProfile.wheeled(CONFIG_FLAT_LEGACY)
    assert p.min_battery_mv == 6500
    assert p.obstacle_stop_mm == 200


# ── Wheeled safety checks ───────────────────────────────────────────────────


def test_wheeled_safe():
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    pkt = _wheeled_pkt(battery_mv=8000, range_front_mm=500)
    assert p.check(pkt) == []


def test_wheeled_low_battery():
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    pkt = _wheeled_pkt(battery_mv=6000)
    violations = p.check(pkt)
    assert len(violations) >= 1
    assert "battery" in violations[0].lower()


def test_wheeled_obstacle():
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    pkt = _wheeled_pkt(range_front_mm=100)
    violations = p.check(pkt)
    assert len(violations) >= 1
    assert "obstacle" in violations[0].lower()


def test_wheeled_tilt():
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    # Accel mostly horizontal = tilted
    pkt = _wheeled_pkt(accel_mg=(1000, 0, 100))
    violations = p.check(pkt)
    assert any("tilt" in v.lower() for v in violations)


# ── Drone safety checks ─────────────────────────────────────────────────────


def test_drone_safe():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    pkt = _drone_pkt(battery_mv=8000, gps_alt_cm=1000)
    assert p.check(pkt) == []


def test_drone_altitude_exceeded():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    # 60m > 50m default ceiling
    pkt = _drone_pkt(gps_alt_cm=6000)
    violations = p.check(pkt)
    assert any("altitude" in v.lower() for v in violations)


def test_drone_no_gps():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    pkt = _drone_pkt(gps_lat=0, gps_lon=0)
    violations = p.check(pkt)
    assert any("gps" in v.lower() for v in violations)


def test_drone_critical_battery_land():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    pkt = _drone_pkt(battery_mv=6000)
    action = p.drone_action(pkt)
    assert action == "land"


def test_drone_low_battery_rtl():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    # Between critical (6500) and RTL (7000)
    pkt = _drone_pkt(battery_mv=6800)
    action = p.drone_action(pkt)
    assert action == "rtl"


def test_drone_battery_ok_no_action():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    pkt = _drone_pkt(battery_mv=8000)
    action = p.drone_action(pkt)
    assert action is None


def test_drone_altitude_violation_stops():
    p = SafetyProfile.drone(CONFIG_EMPTY)
    pkt = _drone_pkt(battery_mv=8000, gps_alt_cm=6000)
    action = p.drone_action(pkt)
    assert action == "stop"


def test_drone_stricter_than_wheeled():
    """Drone should have higher minimum battery than wheeled."""
    w = SafetyProfile.wheeled(CONFIG_EMPTY)
    d = SafetyProfile.drone(CONFIG_EMPTY)
    assert d.min_battery_mv >= w.min_battery_mv


# ── Humanoid safety checks ──────────────────────────────────────────────────


def test_humanoid_safe():
    p = SafetyProfile.humanoid(CONFIG_EMPTY)
    pkt = _humanoid_pkt(battery_mv=8000)
    assert p.check(pkt) == []


def test_humanoid_fall_detected():
    p = SafetyProfile.humanoid(CONFIG_EMPTY)
    # Very high accel = falling
    pkt = _humanoid_pkt(accel_mg=(3000, 3000, 3000))
    violations = p.check(pkt)
    assert any("fall" in v.lower() for v in violations)


def test_humanoid_low_battery():
    p = SafetyProfile.humanoid(CONFIG_EMPTY)
    pkt = _humanoid_pkt(battery_mv=6000)
    violations = p.check(pkt)
    assert any("battery" in v.lower() for v in violations)


# ── Multiple violations ─────────────────────────────────────────────────────


def test_multiple_violations_reported():
    """All violations should be reported, not just the first one."""
    p = SafetyProfile.wheeled(CONFIG_EMPTY)
    pkt = _wheeled_pkt(battery_mv=6000, range_front_mm=100)
    violations = p.check(pkt)
    assert len(violations) >= 2


# ── check() dispatches correctly ─────────────────────────────────────────────


def test_check_dispatches_by_type():
    """check() should call the right method based on robot_type."""
    p_w = SafetyProfile.wheeled(CONFIG_EMPTY)
    p_d = SafetyProfile.drone(CONFIG_EMPTY)
    p_h = SafetyProfile.humanoid(CONFIG_EMPTY)

    # Wheeled packet with obstacle should trigger wheeled check
    wpkt = _wheeled_pkt(range_front_mm=50)
    assert len(p_w.check(wpkt)) >= 1

    # Drone with bad altitude should trigger drone check
    dpkt = _drone_pkt(gps_alt_cm=6000)
    assert len(p_d.check(dpkt)) >= 1

    # Humanoid with fall should trigger humanoid check
    hpkt = _humanoid_pkt(accel_mg=(5000, 0, 0))
    assert len(p_h.check(hpkt)) >= 1


if __name__ == "__main__":
    test_wheeled_factory_defaults()
    test_wheeled_factory_config()
    test_drone_factory_defaults()
    test_drone_factory_config()
    test_humanoid_factory_defaults()
    test_humanoid_factory_config()
    test_ackermann_factory()
    test_for_robot_type_dispatches()
    test_flat_config_falls_back()
    test_wheeled_safe()
    test_wheeled_low_battery()
    test_wheeled_obstacle()
    test_wheeled_tilt()
    test_drone_safe()
    test_drone_altitude_exceeded()
    test_drone_no_gps()
    test_drone_critical_battery_land()
    test_drone_low_battery_rtl()
    test_drone_battery_ok_no_action()
    test_drone_altitude_violation_stops()
    test_drone_stricter_than_wheeled()
    test_humanoid_safe()
    test_humanoid_fall_detected()
    test_humanoid_low_battery()
    test_multiple_violations_reported()
    test_check_dispatches_by_type()
    print("All safety tests passed!")
