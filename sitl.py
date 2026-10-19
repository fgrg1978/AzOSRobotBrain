# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Standalone SITL (Software-In-The-Loop) robot simulator — B01.

Unified simulator for wheeled / drone / ackermann / humanoid robots, usable
without QEMU or real hardware. Connects to the brain TCP server just like a
real robot: exchanges SENSOR, CAMERA, STATUS packets and ActuatorCmds.

Two operating styles:

  1. Programmatic (tests / embedded):
        from sitl import SITLRobot, ROBOT_WHEELED
        robot = SITLRobot(robot_type=ROBOT_WHEELED)
        robot.step(dt=0.1)
        pkt = robot.sensor_packet_bytes()

  2. Network (CLI / full brain integration):
        python sitl.py --type wheeled --scenario forward_10m
        # opens TCP to 127.0.0.1:8000, completes the Noise handshake with
        # AZOS_BRAIN_LINK_KEY (the key the brain runs with), streams sensor frames

Scenario runner:

        from sitl import SITLRobot, run_scenario
        robot = SITLRobot(robot_type=ROBOT_WHEELED)
        run_scenario(robot, "forward_10m")

The existing `tools/sitl/sitl_wheeled.py` remains the canonical wheeled-only
sim (raycast world, JPEG frame rendering). This module is lighter, covers all
robot types, and is oriented toward unit-test usage — it delegates to the
wheeled sim when a scenario needs raycasting.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import protocol
from robot_link import RobotLink
from protocol import (
    ActuatorCmd,
    ACT_DIFF_DRIVE,
    ACT_QUAD_ROTOR,
    ACT_HUMANOID,
    ACT_ACKERMANN,
    CAMERA_FRAME,
    FLAG_EMERGENCY,
    ROBOT_ACKERMANN,
    ROBOT_DRONE,
    ROBOT_HUMANOID,
    ROBOT_WHEELED,
    SENSOR_PACKET,
    STATUS,
    SensorPacket,
    SensorPacketDrone,
    SensorPacketHumanoid,
    StatusPacket,
)

# ── Physics / limits ─────────────────────────────────────────────────────────

# Wheeled / ackermann
WHEEL_BASE_MM = 142  # Yahboom chassis 310
ACKERMANN_WB_MM = 250  # 1/10 RC-car wheelbase
TICKS_PER_M = 1000
MM_PER_M = 1000
MAX_SPEED_MM_S = 500  # 100 % throttle = 0.5 m/s
ACKERMANN_MAX_STEER_CDEG = 3500  # ±35 °

# Drone
DRONE_MAX_CLIMB_M_S = 3.0  # m/s at full throttle delta
DRONE_MAX_TILT_M_S2 = 5.0  # m/s² per PWM unit of roll/pitch
DRONE_MAX_YAW_DPS = 180.0  # deg/s at full yaw PWM
DRONE_PWM_NEUTRAL = 1500
DRONE_PWM_MIN = 1000
DRONE_PWM_MAX = 2000
DRONE_PWM_SPAN_HALF = DRONE_PWM_MAX - DRONE_PWM_NEUTRAL  # 500
DRONE_HOVER_PWM = 1450  # throttle at which gravity is balanced
DRONE_GRAVITY_M_S2 = 9.81

# Common
BATTERY_START_MV = 7400
BATTERY_MIN_MV = 0
BATTERY_DRAIN_MV_S = 0.5  # idle drain
BATTERY_LOAD_FACTOR = 3.0  # multiplier at full throttle
ENCODER_NOISE_TICKS = 2
ACCEL_NOISE_MG = 2
GYRO_NOISE_MDPS = 2
GRAVITY_MG_Z = 1000  # 1 g, +Z when level

DEFAULT_START_X_MM = 1000.0
DEFAULT_START_Y_MM = 1000.0
DEFAULT_START_Z_MM = 0.0
DEFAULT_START_HDG_DEG = 0.0
DEFAULT_ARENA_MM = 10000.0

# Humanoid
HUMANOID_DEFAULT_JOINTS = 12
HUMANOID_POSE_STAND_CDEG = 0

# GPS — default "field" location so the drone has a valid fix
DEFAULT_GPS_LAT_DEG7 = 400000000  # 40.0000000 °
DEFAULT_GPS_LON_DEG7 = -37000000  # -3.7000000 °
DEFAULT_GPS_ALT_CM = 10000  # 100 m MSL
GPS_DEG7_SCALE = 1e7
METERS_PER_DEG_LAT = 111320.0  # approximate equirectangular

# Barometer / sonar
BARO_SEA_LEVEL_PA = 101325
BARO_PA_PER_M = 12  # approximate near sea level
SONAR_MAX_MM = 4000

# Magnetometer — simple heading model in µT
MAG_FIELD_UT = 45

# Frame rates
DEFAULT_SENSOR_HZ = 20.0
DEFAULT_CAMERA_HZ = 2.0
DEFAULT_PHYSICS_HZ = 100.0
MIN_TICK_INTERVAL_S = 1e-6

# Camera (synthetic gradient)
CAM_WIDTH_PX = 160
CAM_HEIGHT_PX = 120
CAM_FORMAT_JPEG = 1
CAM_FORMAT_RAW_RGB = 2
CAM_HEADER_FMT = "<HHB"
CAM_HEADER_SIZE = struct.calcsize(CAM_HEADER_FMT)

# Arena wall clamp offset — keep robot inside bounds
ARENA_WALL_EPS_MM = 10.0

# Default network
DEFAULT_BRAIN_HOST = "127.0.0.1"
DEFAULT_BRAIN_PORT = 8000

# Scenario duration defaults (seconds)
SCEN_SHORT_S = 2.0
SCEN_MEDIUM_S = 5.0
SCEN_LONG_S = 10.0


# Conversion helpers
def _deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ── Synthetic camera ─────────────────────────────────────────────────────────


def synthetic_camera_payload(
    width: int = CAM_WIDTH_PX,
    height: int = CAM_HEIGHT_PX,
    seed: int = 0,
) -> bytes:
    """Return a fake PKT_CAMERA payload.

    Uses Pillow if available (JPEG gradient), otherwise falls back to a
    raw-RGB payload so the simulator works in minimal environments.
    Header: width(u16 LE) + height(u16 LE) + format(u8).
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        # Minimal raw-RGB fallback (deterministic gradient)
        pixels = bytearray(width * height * 3)
        for y in range(height):
            row_base = (y * 255) // max(1, height - 1)
            for x in range(width):
                col = (x * 255) // max(1, width - 1)
                idx = (y * width + x) * 3
                pixels[idx] = (col + seed) & 0xFF
                pixels[idx + 1] = (row_base + seed) & 0xFF
                pixels[idx + 2] = ((col + row_base) // 2) & 0xFF
        header = struct.pack(CAM_HEADER_FMT, width, height, CAM_FORMAT_RAW_RGB)
        return header + bytes(pixels)

    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        row_base = (y * 255) // max(1, height - 1)
        for x in range(width):
            col = (x * 255) // max(1, width - 1)
            px[x, y] = (
                (col + seed) & 0xFF,
                (row_base + seed) & 0xFF,
                ((col + row_base) // 2) & 0xFF,
            )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    header = struct.pack(CAM_HEADER_FMT, width, height, CAM_FORMAT_JPEG)
    return header + buf.getvalue()


# ── SITLRobot — unified simulator ────────────────────────────────────────────


@dataclass
class SITLRobot:
    """Unified robot simulator.

    Supports all four robot_type values. Only the physics relevant to the
    chosen type is updated on `step()`, but sensor packets are always
    produced in the correct wire format.
    """

    robot_type: int = ROBOT_WHEELED

    # Pose (right-hand, ENU: x=east_mm, y=north_mm, z=up_mm)
    x_mm: float = DEFAULT_START_X_MM
    y_mm: float = DEFAULT_START_Y_MM
    z_mm: float = DEFAULT_START_Z_MM
    hdg_deg: float = DEFAULT_START_HDG_DEG

    # Velocities (m/s / deg/s)
    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    yaw_rate_dps: float = 0.0

    # Actuator state (last command received)
    speed_l_pct: int = 0  # wheeled / ackermann
    speed_r_pct: int = 0  # wheeled
    ackermann_steer_cdeg: int = 0
    drone_throttle_pwm: int = DRONE_PWM_MIN
    drone_roll_pwm: int = DRONE_PWM_NEUTRAL
    drone_pitch_pwm: int = DRONE_PWM_NEUTRAL
    drone_yaw_pwm: int = DRONE_PWM_NEUTRAL
    humanoid_joints_cdeg: list[int] = field(default_factory=list)
    flags: int = 0

    # Derived / integrated sensors
    enc_l: int = 0
    enc_r: int = 0
    odom_dist_mm: int = 0
    odom_hdg_cdeg: int = 0
    battery_mv: float = BATTERY_START_MV

    # GPS origin (for drone lat/lon integration)
    gps_origin_lat_deg7: int = DEFAULT_GPS_LAT_DEG7
    gps_origin_lon_deg7: int = DEFAULT_GPS_LON_DEG7
    gps_origin_alt_cm: int = DEFAULT_GPS_ALT_CM

    # Arena (keeps simulator bounded for tests)
    arena_width_mm: float = DEFAULT_ARENA_MM
    arena_height_mm: float = DEFAULT_ARENA_MM

    # Internal
    _t_start: float = field(default_factory=time.monotonic)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    # ── Setup ────────────────────────────────────────────────────────────
    def __post_init__(self):
        if self.robot_type == ROBOT_HUMANOID and not self.humanoid_joints_cdeg:
            self.humanoid_joints_cdeg = [HUMANOID_POSE_STAND_CDEG] * HUMANOID_DEFAULT_JOINTS
        if self.robot_type == ROBOT_DRONE:
            # Drone starts disarmed / on the ground
            self.drone_throttle_pwm = DRONE_PWM_MIN

    # ── Command application ─────────────────────────────────────────────
    def apply_cmd(self, cmd: ActuatorCmd) -> None:
        """Apply an incoming ActuatorCmd to the simulated robot."""
        self.flags = cmd.flags
        if cmd.flags & FLAG_EMERGENCY:
            self._disarm()
            return

        if cmd.actuator_type == ACT_DIFF_DRIVE and len(cmd.channels) >= 2:
            self.speed_l_pct = int(_clamp(cmd.channels[0], -100, 100))
            self.speed_r_pct = int(_clamp(cmd.channels[1], -100, 100))

        elif cmd.actuator_type == ACT_ACKERMANN and len(cmd.channels) >= 2:
            self.speed_l_pct = int(_clamp(cmd.channels[0], -100, 100))
            self.ackermann_steer_cdeg = int(
                _clamp(cmd.channels[1], -ACKERMANN_MAX_STEER_CDEG, ACKERMANN_MAX_STEER_CDEG)
            )

        elif cmd.actuator_type == ACT_QUAD_ROTOR and len(cmd.channels) >= 4:
            self.drone_throttle_pwm = int(_clamp(cmd.channels[0], DRONE_PWM_MIN, DRONE_PWM_MAX))
            self.drone_roll_pwm = int(_clamp(cmd.channels[1], DRONE_PWM_MIN, DRONE_PWM_MAX))
            self.drone_pitch_pwm = int(_clamp(cmd.channels[2], DRONE_PWM_MIN, DRONE_PWM_MAX))
            self.drone_yaw_pwm = int(_clamp(cmd.channels[3], DRONE_PWM_MIN, DRONE_PWM_MAX))

        elif cmd.actuator_type == ACT_HUMANOID and cmd.channels:
            self.humanoid_joints_cdeg = [int(c) for c in cmd.channels]

    def _disarm(self) -> None:
        self.speed_l_pct = 0
        self.speed_r_pct = 0
        self.ackermann_steer_cdeg = 0
        self.drone_throttle_pwm = DRONE_PWM_MIN
        self.drone_roll_pwm = DRONE_PWM_NEUTRAL
        self.drone_pitch_pwm = DRONE_PWM_NEUTRAL
        self.drone_yaw_pwm = DRONE_PWM_NEUTRAL
        self.vx_m_s = self.vy_m_s = self.vz_m_s = 0.0
        self.yaw_rate_dps = 0.0

    # ── Physics ─────────────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        """Advance the simulation by dt seconds."""
        if dt < MIN_TICK_INTERVAL_S:
            return

        if self.robot_type == ROBOT_WHEELED:
            self._step_wheeled(dt)
        elif self.robot_type == ROBOT_ACKERMANN:
            self._step_ackermann(dt)
        elif self.robot_type == ROBOT_DRONE:
            self._step_drone(dt)
        elif self.robot_type == ROBOT_HUMANOID:
            # Humanoid stays still — poses are commanded joint angles
            pass

        # Battery drain common to all types
        self.battery_mv = max(
            BATTERY_MIN_MV,
            self.battery_mv
            - BATTERY_DRAIN_MV_S * dt * (1.0 + self._load_factor() * BATTERY_LOAD_FACTOR),
        )

    def _load_factor(self) -> float:
        if self.robot_type in (ROBOT_WHEELED, ROBOT_ACKERMANN):
            return (abs(self.speed_l_pct) + abs(self.speed_r_pct)) / 200.0
        if self.robot_type == ROBOT_DRONE:
            thr = self.drone_throttle_pwm - DRONE_PWM_MIN
            span = DRONE_PWM_MAX - DRONE_PWM_MIN
            return thr / span if span > 0 else 0.0
        return 0.0

    def _step_wheeled(self, dt: float) -> None:
        vl = self.speed_l_pct / 100.0 * MAX_SPEED_MM_S  # mm/s
        vr = self.speed_r_pct / 100.0 * MAX_SPEED_MM_S
        v_center = (vl + vr) / 2.0
        omega_rad = (vr - vl) / WHEEL_BASE_MM

        self.hdg_deg = (self.hdg_deg + math.degrees(omega_rad * dt)) % 360.0
        self.yaw_rate_dps = math.degrees(omega_rad)

        hdg_rad = _deg_to_rad(self.hdg_deg)
        dx = v_center * dt * math.cos(hdg_rad)
        dy = v_center * dt * math.sin(hdg_rad)
        self._move_clamped(dx, dy, 0.0)

        # Encoders: ticks per wheel
        self.enc_l += int(vl * dt / MM_PER_M * TICKS_PER_M)
        self.enc_r += int(vr * dt / MM_PER_M * TICKS_PER_M)
        self.odom_dist_mm += int(v_center * dt)
        self.odom_hdg_cdeg = int(self.hdg_deg * 100) % 36000

        self.vx_m_s = v_center * math.cos(hdg_rad) / MM_PER_M
        self.vy_m_s = v_center * math.sin(hdg_rad) / MM_PER_M

    def _step_ackermann(self, dt: float) -> None:
        v_mm_s = self.speed_l_pct / 100.0 * MAX_SPEED_MM_S
        steer_rad = _deg_to_rad(self.ackermann_steer_cdeg / 100.0)
        omega_rad = 0.0
        if ACKERMANN_WB_MM > 0:
            omega_rad = v_mm_s / ACKERMANN_WB_MM * math.tan(steer_rad)

        self.hdg_deg = (self.hdg_deg + math.degrees(omega_rad * dt)) % 360.0
        self.yaw_rate_dps = math.degrees(omega_rad)

        hdg_rad = _deg_to_rad(self.hdg_deg)
        dx = v_mm_s * dt * math.cos(hdg_rad)
        dy = v_mm_s * dt * math.sin(hdg_rad)
        self._move_clamped(dx, dy, 0.0)

        self.odom_dist_mm += int(v_mm_s * dt)
        self.odom_hdg_cdeg = int(self.hdg_deg * 100) % 36000
        self.vx_m_s = v_mm_s * math.cos(hdg_rad) / MM_PER_M
        self.vy_m_s = v_mm_s * math.sin(hdg_rad) / MM_PER_M

    def _step_drone(self, dt: float) -> None:
        # Thrust -> vertical acceleration (relative to hover throttle)
        pwm_delta = self.drone_throttle_pwm - DRONE_HOVER_PWM
        thrust_a = (pwm_delta / DRONE_PWM_SPAN_HALF) * DRONE_GRAVITY_M_S2
        self.vz_m_s += thrust_a * dt
        # Integrate altitude, clamp to ground
        dz_mm = self.vz_m_s * dt * MM_PER_M
        new_z = self.z_mm + dz_mm
        if new_z < 0.0:
            new_z = 0.0
            self.vz_m_s = 0.0
        self.z_mm = new_z

        # Roll / pitch -> horizontal acceleration in body frame
        roll_ratio = (self.drone_roll_pwm - DRONE_PWM_NEUTRAL) / DRONE_PWM_SPAN_HALF
        pitch_ratio = (self.drone_pitch_pwm - DRONE_PWM_NEUTRAL) / DRONE_PWM_SPAN_HALF
        ax_body = pitch_ratio * DRONE_MAX_TILT_M_S2
        ay_body = roll_ratio * DRONE_MAX_TILT_M_S2
        hdg_rad = _deg_to_rad(self.hdg_deg)
        # Body -> world
        ax_world = ax_body * math.cos(hdg_rad) - ay_body * math.sin(hdg_rad)
        ay_world = ax_body * math.sin(hdg_rad) + ay_body * math.cos(hdg_rad)
        self.vx_m_s += ax_world * dt
        self.vy_m_s += ay_world * dt

        # Yaw rate from yaw stick
        yaw_ratio = (self.drone_yaw_pwm - DRONE_PWM_NEUTRAL) / DRONE_PWM_SPAN_HALF
        self.yaw_rate_dps = yaw_ratio * DRONE_MAX_YAW_DPS
        self.hdg_deg = (self.hdg_deg + self.yaw_rate_dps * dt) % 360.0

        # Translate position
        dx_mm = self.vx_m_s * dt * MM_PER_M
        dy_mm = self.vy_m_s * dt * MM_PER_M
        self._move_clamped(dx_mm, dy_mm, 0.0)  # z already updated above

    def _move_clamped(self, dx_mm: float, dy_mm: float, dz_mm: float) -> None:
        self.x_mm = _clamp(
            self.x_mm + dx_mm, ARENA_WALL_EPS_MM, self.arena_width_mm - ARENA_WALL_EPS_MM
        )
        self.y_mm = _clamp(
            self.y_mm + dy_mm, ARENA_WALL_EPS_MM, self.arena_height_mm - ARENA_WALL_EPS_MM
        )
        self.z_mm = max(0.0, self.z_mm + dz_mm)

    # ── Sensor emulation ────────────────────────────────────────────────
    def timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t_start) * 1000)

    def _noise(self, magnitude: int) -> int:
        return self._rng.randint(-magnitude, magnitude) if magnitude > 0 else 0

    def _accel_mg(self) -> tuple[int, int, int]:
        return (
            self._noise(ACCEL_NOISE_MG),
            self._noise(ACCEL_NOISE_MG),
            GRAVITY_MG_Z + self._noise(ACCEL_NOISE_MG),
        )

    def _gyro_mdps(self) -> tuple[int, int, int]:
        return (
            self._noise(GYRO_NOISE_MDPS),
            self._noise(GYRO_NOISE_MDPS),
            int(self.yaw_rate_dps * 1000) + self._noise(GYRO_NOISE_MDPS),
        )

    def _gps_latlon(self) -> tuple[int, int, int]:
        """Integrate position deltas as GPS lat/lon using equirectangular."""
        lat_m = self.y_mm / MM_PER_M
        lon_m = self.x_mm / MM_PER_M
        lat_deg = lat_m / METERS_PER_DEG_LAT
        lat_base_rad = _deg_to_rad(self.gps_origin_lat_deg7 / GPS_DEG7_SCALE)
        meters_per_deg_lon = max(1.0, METERS_PER_DEG_LAT * math.cos(lat_base_rad))
        lon_deg = lon_m / meters_per_deg_lon
        return (
            self.gps_origin_lat_deg7 + int(lat_deg * GPS_DEG7_SCALE),
            self.gps_origin_lon_deg7 + int(lon_deg * GPS_DEG7_SCALE),
            self.gps_origin_alt_cm + int(self.z_mm / 10.0),
        )

    def sensor_packet(self):
        """Return the SensorPacket dataclass matching the current robot type."""
        ts = self.timestamp_ms()
        accel = self._accel_mg()
        gyro = self._gyro_mdps()
        batt = int(self.battery_mv)

        if self.robot_type == ROBOT_DRONE:
            lat, lon, alt_cm = self._gps_latlon()
            baro_pa = BARO_SEA_LEVEL_PA - int(self.z_mm / MM_PER_M * BARO_PA_PER_M)
            hdg_rad = _deg_to_rad(self.hdg_deg)
            mag = (
                int(MAG_FIELD_UT * math.cos(hdg_rad)),
                int(MAG_FIELD_UT * math.sin(hdg_rad)),
                0,
            )
            sonar = min(SONAR_MAX_MM, int(self.z_mm))
            return SensorPacketDrone(
                timestamp_ms=ts,
                battery_mv=batt,
                accel_mg=accel,
                gyro_mdps=gyro,
                baro_pa=baro_pa,
                mag_ut=mag,
                gps_lat_deg7=lat,
                gps_lon_deg7=lon,
                gps_alt_cm=alt_cm,
                sonar_down_mm=sonar,
            )

        if self.robot_type == ROBOT_HUMANOID:
            return SensorPacketHumanoid(
                timestamp_ms=ts,
                battery_mv=batt,
                accel_mg=accel,
                gyro_mdps=gyro,
                joint_angles=list(self.humanoid_joints_cdeg),
                foot_pressure_l=0,
                foot_pressure_r=0,
            )

        # Wheeled / ackermann both use SensorPacket (same wheeled layout)
        enc_l = self.enc_l + self._noise(ENCODER_NOISE_TICKS)
        enc_r = self.enc_r + self._noise(ENCODER_NOISE_TICKS)
        # Range sensors: arena bounds as a fake wall
        range_front = int(max(0.0, self.arena_width_mm - self.x_mm))
        range_right = int(max(0.0, self.arena_height_mm - self.y_mm))
        return SensorPacket(
            timestamp_ms=ts,
            battery_mv=batt,
            accel_mg=accel,
            gyro_mdps=gyro,
            odom_dist_mm=self.odom_dist_mm,
            odom_hdg_cdeg=self.odom_hdg_cdeg,
            encoder_l=enc_l,
            encoder_r=enc_r,
            range_front_mm=range_front,
            range_right_mm=range_right,
        )

    def sensor_packet_bytes(self) -> bytes:
        return self.sensor_packet().to_bytes()

    def status_packet(self, mode: int = 1, tasks_ok: int = 8) -> StatusPacket:
        uptime = int(time.monotonic() - self._t_start)
        return StatusPacket(
            mode=mode,
            tasks_ok=tasks_ok,
            canary_ok=tasks_ok,
            uptime_s=uptime,
            robot_type=self.robot_type,
        )

    def camera_payload(self, seed: int = 0) -> bytes:
        return synthetic_camera_payload(seed=seed)

    # ── Introspection (helpful for tests / viz) ─────────────────────────
    def pose(self) -> tuple[float, float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm, self.hdg_deg)


# ── Scenarios ────────────────────────────────────────────────────────────────

# A scenario is a function (SITLRobot) -> list of (t_s, ActuatorCmd) events.
# The runner advances physics, applying each event when its time is reached.

ScenarioEvent = tuple[float, ActuatorCmd]
ScenarioFn = Callable[[SITLRobot], list[ScenarioEvent]]


def _scen_idle(_: SITLRobot) -> list[ScenarioEvent]:
    return []


def _scen_forward_10m(robot: SITLRobot) -> list[ScenarioEvent]:
    if robot.robot_type == ROBOT_DRONE:
        # Forward flight at constant low pitch
        return [
            (
                0.0,
                ActuatorCmd.drone(
                    DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL + 100, DRONE_PWM_NEUTRAL
                ),
            ),
            (
                SCEN_LONG_S,
                ActuatorCmd.drone(
                    DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL
                ),
            ),
        ]
    if robot.robot_type == ROBOT_ACKERMANN:
        return [
            (0.0, ActuatorCmd(actuator_type=ACT_ACKERMANN, channels=[100, 0])),
            (SCEN_MEDIUM_S, ActuatorCmd.stop(ACT_ACKERMANN)),
        ]
    # wheeled default
    return [(0.0, ActuatorCmd.wheeled(100, 100)), (SCEN_MEDIUM_S, ActuatorCmd.wheeled(0, 0))]


def _scen_spin_in_place(robot: SITLRobot) -> list[ScenarioEvent]:
    if robot.robot_type == ROBOT_DRONE:
        return [
            (
                0.0,
                ActuatorCmd.drone(
                    DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL + 200
                ),
            ),
            (
                SCEN_SHORT_S,
                ActuatorCmd.drone(
                    DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL
                ),
            ),
        ]
    return [(0.0, ActuatorCmd.wheeled(60, -60)), (SCEN_SHORT_S, ActuatorCmd.wheeled(0, 0))]


def _scen_takeoff_hover_land(robot: SITLRobot) -> list[ScenarioEvent]:
    # Useful only for drone — provide a no-op fallback otherwise
    if robot.robot_type != ROBOT_DRONE:
        return _scen_idle(robot)
    return [
        (
            0.0,
            ActuatorCmd.drone(
                DRONE_PWM_NEUTRAL + 200, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL
            ),
        ),  # climb
        (
            SCEN_SHORT_S,
            ActuatorCmd.drone(
                DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL
            ),
        ),  # hover
        (
            SCEN_MEDIUM_S,
            ActuatorCmd.drone(
                DRONE_HOVER_PWM - 100, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL
            ),
        ),  # descend
    ]


def _scen_emergency_stop(_: SITLRobot) -> list[ScenarioEvent]:
    return [
        (0.0, ActuatorCmd.wheeled(100, 100)),
        (SCEN_SHORT_S, ActuatorCmd.stop()),
    ]


SCENARIOS: dict[str, ScenarioFn] = {
    "idle": _scen_idle,
    "forward_10m": _scen_forward_10m,
    "spin_in_place": _scen_spin_in_place,
    "takeoff_hover_land": _scen_takeoff_hover_land,
    "emergency_stop": _scen_emergency_stop,
}


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS.keys())


def run_scenario(
    robot: SITLRobot,
    name: str,
    duration_s: float = SCEN_LONG_S,
    dt: float = 1.0 / DEFAULT_PHYSICS_HZ,
) -> SITLRobot:
    """Play a named scenario against the given robot (in-memory, no network).

    Returns the robot (for chaining). Raises KeyError if the scenario is unknown.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}' — known: {list_scenarios()}")
    events = SCENARIOS[name](robot)
    events_sorted = sorted(events, key=lambda e: e[0])

    t = 0.0
    idx = 0
    steps = max(1, int(duration_s / dt))
    for _ in range(steps):
        # Fire any due events
        while idx < len(events_sorted) and events_sorted[idx][0] <= t:
            robot.apply_cmd(events_sorted[idx][1])
            idx += 1
        robot.step(dt)
        t += dt
    return robot


# ── Network client ───────────────────────────────────────────────────────────


class SITLNetClient:
    """Connects a SITLRobot to a brain TCP server.

    Runs four async loops: physics, sensor, camera, recv.
    """

    def __init__(
        self,
        robot: SITLRobot,
        host: str = DEFAULT_BRAIN_HOST,
        port: int = DEFAULT_BRAIN_PORT,
        sensor_hz: float = DEFAULT_SENSOR_HZ,
        camera_hz: float = DEFAULT_CAMERA_HZ,
        physics_hz: float = DEFAULT_PHYSICS_HZ,
        duration_s: float = 0.0,
    ):
        self.robot = robot
        self.host = host
        self.port = port
        self.sensor_interval = 1.0 / sensor_hz if sensor_hz > 0 else 0.0
        self.camera_interval = 1.0 / camera_hz if camera_hz > 0 else 0.0
        self.physics_interval = 1.0 / physics_hz if physics_hz > 0 else 0.0
        self.duration_s = duration_s
        self._running = False

    async def run(self) -> None:
        link = await RobotLink.connect(self.host, self.port)
        self._running = True
        # Initial STATUS
        link.queue_packet(STATUS, self.robot.status_packet().to_bytes())
        await link.writer.drain()

        tasks = [
            asyncio.create_task(self._physics_loop()),
            asyncio.create_task(self._sensor_loop(link)),
            asyncio.create_task(self._camera_loop(link)),
            asyncio.create_task(self._recv_loop(link)),
        ]
        try:
            if self.duration_s > 0:
                await asyncio.sleep(self.duration_s)
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._running = False
            for t in tasks:
                t.cancel()
            link.close()
            try:
                await link.wait_closed()
            except Exception:
                pass

    async def _physics_loop(self) -> None:
        last = time.monotonic()
        while self._running:
            now = time.monotonic()
            self.robot.step(now - last)
            last = now
            await asyncio.sleep(self.physics_interval)

    async def _sensor_loop(self, link: RobotLink) -> None:
        if self.sensor_interval <= 0:
            return
        while self._running:
            link.queue_packet(SENSOR_PACKET, self.robot.sensor_packet_bytes())
            try:
                await link.writer.drain()
            except (ConnectionError, BrokenPipeError):
                self._running = False
                return
            await asyncio.sleep(self.sensor_interval)

    async def _camera_loop(self, link: RobotLink) -> None:
        if self.camera_interval <= 0:
            return
        frame_seed = 0
        while self._running:
            link.queue_packet(CAMERA_FRAME, self.robot.camera_payload(frame_seed))
            frame_seed = (frame_seed + 1) & 0xFF
            try:
                await link.writer.drain()
            except (ConnectionError, BrokenPipeError):
                self._running = False
                return
            await asyncio.sleep(self.camera_interval)

    async def _recv_loop(self, link: RobotLink) -> None:
        while self._running:
            try:
                res = await link.read_packet()
            except asyncio.IncompleteReadError:
                self._running = False
                return
            if res is None:
                continue
            pkt_type, payload = res
            if pkt_type == protocol.ACTUATOR_CMD:
                self.robot.apply_cmd(ActuatorCmd.from_bytes(payload))


# ── CLI ──────────────────────────────────────────────────────────────────────

_TYPE_STR_TO_INT = {
    "wheeled": ROBOT_WHEELED,
    "drone": ROBOT_DRONE,
    "humanoid": ROBOT_HUMANOID,
    "ackermann": ROBOT_ACKERMANN,
}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Standalone SITL robot simulator (B01)")
    ap.add_argument("--type", choices=list(_TYPE_STR_TO_INT.keys()), default="wheeled")
    ap.add_argument("--host", default=DEFAULT_BRAIN_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_BRAIN_PORT)
    ap.add_argument(
        "--scenario",
        default="",
        help=f"run a pre-defined scenario offline (no network). "
        f"Known: {', '.join(list_scenarios())}",
    )
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = forever)")
    ap.add_argument("--sensor-hz", type=float, default=DEFAULT_SENSOR_HZ)
    ap.add_argument("--camera-hz", type=float, default=DEFAULT_CAMERA_HZ)
    args = ap.parse_args(argv)

    robot = SITLRobot(robot_type=_TYPE_STR_TO_INT[args.type])

    if args.scenario:
        run_scenario(robot, args.scenario, duration_s=args.duration or SCEN_LONG_S)
        print(f"[SITL] Scenario '{args.scenario}' done. Pose: {robot.pose()}")
        return 0

    client = SITLNetClient(
        robot,
        args.host,
        args.port,
        sensor_hz=args.sensor_hz,
        camera_hz=args.camera_hz,
        duration_s=args.duration,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
