# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Robot Description Format (AT2) — YAML declarative robot specification.

Replaces hardcoded pin assignments. Robot capabilities are loaded from YAML
and used by the brain to understand what the robot can do.

Example robot.yaml:
    name: patrol_bot_1
    type: wheeled
    chassis:
      wheel_base_mm: 142
      wheel_diameter_mm: 65
      max_speed_pct: 80
    sensors:
      - type: imu
        model: mpu6050
        bus: i2c
        address: 0x68
        rate_hz: 100
      - type: rangefinder
        model: hcsr04
        gpio_trig: 5
        gpio_echo: 6
        position: front
        max_range_mm: 4000
      - type: camera
        model: ov2640
        interface: csi
        resolution: [640, 480]
        fps: 15
      - type: pir
        gpio: 12
        position: front
      - type: gps
        model: neo6m
        bus: uart
        baud: 9600
    actuators:
      - type: motor
        model: dc_brushed
        pwm_channel: 0
        dir_gpio: 2
        encoder_gpio: 3
        side: left
      - type: motor
        model: dc_brushed
        pwm_channel: 1
        dir_gpio: 4
        encoder_gpio: 7
        side: right
      - type: buzzer
        gpio: 8
      - type: led
        type: ws2812
        gpio: 9
        count: 8
    payload:
      - type: spotlight
        gpio: 10
        watts: 10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

# ── Constants ────────────────────────────────────────────────────────────────

MAX_SENSORS = 16
MAX_ACTUATORS = 16
MAX_PAYLOADS = 8
DEFAULT_ROBOT_YAML = "robot.yaml"

GPIO_UNSET = -1
ADDRESS_UNSET = 0
RATE_UNSET = 0
RANGE_UNSET = 0
RESOLUTION_UNSET = (0, 0)
FPS_UNSET = 0
BAUD_UNSET = 0
PWM_CHANNEL_UNSET = -1
COUNT_DEFAULT = 1
WHEEL_BASE_DEFAULT_MM = 200
WHEEL_DIAMETER_DEFAULT_MM = 65
MAX_SPEED_DEFAULT_PCT = 80
TRACK_WIDTH_DEFAULT_MM = 0
WATTS_UNSET = 0


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class SensorDesc:
    """Description of a single sensor on the robot."""

    type: str  # imu, rangefinder, camera, pir, gps, lidar, etc.
    model: str = ""
    bus: str = ""  # i2c, spi, uart, gpio, csi
    address: int = ADDRESS_UNSET
    gpio: int = GPIO_UNSET
    gpio_trig: int = GPIO_UNSET
    gpio_echo: int = GPIO_UNSET
    position: str = ""  # front, rear, left, right, top
    rate_hz: int = RATE_UNSET
    max_range_mm: int = RANGE_UNSET
    resolution: tuple[int, int] = RESOLUTION_UNSET
    fps: int = FPS_UNSET
    baud: int = BAUD_UNSET


@dataclass
class ActuatorDesc:
    """Description of a single actuator on the robot."""

    type: str  # motor, servo, buzzer, led
    model: str = ""
    pwm_channel: int = PWM_CHANNEL_UNSET
    dir_gpio: int = GPIO_UNSET
    encoder_gpio: int = GPIO_UNSET
    gpio: int = GPIO_UNSET
    side: str = ""  # left, right
    count: int = COUNT_DEFAULT


@dataclass
class ChassisDesc:
    """Physical chassis dimensions and constraints."""

    wheel_base_mm: int = WHEEL_BASE_DEFAULT_MM
    wheel_diameter_mm: int = WHEEL_DIAMETER_DEFAULT_MM
    max_speed_pct: int = MAX_SPEED_DEFAULT_PCT
    track_width_mm: int = TRACK_WIDTH_DEFAULT_MM


@dataclass
class PayloadDesc:
    """Description of an optional payload device."""

    type: str  # spotlight, siren, laser, speaker, gripper, sprayer
    gpio: int = GPIO_UNSET
    watts: int = WATTS_UNSET


@dataclass
class RobotDescription:
    """Complete declarative description of a robot's hardware."""

    name: str = "robot"
    type: str = "wheeled"  # wheeled, drone, humanoid, ackermann
    chassis: ChassisDesc = field(default_factory=ChassisDesc)
    sensors: list[SensorDesc] = field(default_factory=list)
    actuators: list[ActuatorDesc] = field(default_factory=list)
    payloads: list[PayloadDesc] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> RobotDescription:
        """Load robot description from a YAML file."""
        resolved = os.path.expanduser(path)
        with open(resolved, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotDescription:
        """Build RobotDescription from a plain dict (e.g., parsed YAML)."""
        desc = cls()
        desc.name = d.get("name", desc.name)
        desc.type = d.get("type", desc.type)

        # Chassis
        chassis_raw = d.get("chassis", {})
        if chassis_raw:
            desc.chassis = ChassisDesc(
                wheel_base_mm=chassis_raw.get("wheel_base_mm", WHEEL_BASE_DEFAULT_MM),
                wheel_diameter_mm=chassis_raw.get("wheel_diameter_mm", WHEEL_DIAMETER_DEFAULT_MM),
                max_speed_pct=chassis_raw.get("max_speed_pct", MAX_SPEED_DEFAULT_PCT),
                track_width_mm=chassis_raw.get("track_width_mm", TRACK_WIDTH_DEFAULT_MM),
            )

        # Sensors
        for s_raw in d.get("sensors", []):
            if len(desc.sensors) >= MAX_SENSORS:
                break
            resolution = s_raw.get("resolution", list(RESOLUTION_UNSET))
            if isinstance(resolution, list):
                resolution = tuple(resolution)
            desc.sensors.append(
                SensorDesc(
                    type=s_raw.get("type", ""),
                    model=s_raw.get("model", ""),
                    bus=s_raw.get("bus", ""),
                    address=s_raw.get("address", ADDRESS_UNSET),
                    gpio=s_raw.get("gpio", GPIO_UNSET),
                    gpio_trig=s_raw.get("gpio_trig", GPIO_UNSET),
                    gpio_echo=s_raw.get("gpio_echo", GPIO_UNSET),
                    position=s_raw.get("position", ""),
                    rate_hz=s_raw.get("rate_hz", RATE_UNSET),
                    max_range_mm=s_raw.get("max_range_mm", RANGE_UNSET),
                    resolution=resolution,
                    fps=s_raw.get("fps", FPS_UNSET),
                    baud=s_raw.get("baud", BAUD_UNSET),
                )
            )

        # Actuators
        for a_raw in d.get("actuators", []):
            if len(desc.actuators) >= MAX_ACTUATORS:
                break
            desc.actuators.append(
                ActuatorDesc(
                    type=a_raw.get("type", ""),
                    model=a_raw.get("model", ""),
                    pwm_channel=a_raw.get("pwm_channel", PWM_CHANNEL_UNSET),
                    dir_gpio=a_raw.get("dir_gpio", GPIO_UNSET),
                    encoder_gpio=a_raw.get("encoder_gpio", GPIO_UNSET),
                    gpio=a_raw.get("gpio", GPIO_UNSET),
                    side=a_raw.get("side", ""),
                    count=a_raw.get("count", COUNT_DEFAULT),
                )
            )

        # Payloads
        for p_raw in d.get("payload", d.get("payloads", [])):
            if len(desc.payloads) >= MAX_PAYLOADS:
                break
            desc.payloads.append(
                PayloadDesc(
                    type=p_raw.get("type", ""),
                    gpio=p_raw.get("gpio", GPIO_UNSET),
                    watts=p_raw.get("watts", WATTS_UNSET),
                )
            )

        return desc

    # ── Query helpers ────────────────────────────────────────────────────────

    def has_sensor(self, sensor_type: str) -> bool:
        """Check if the robot has a sensor of the given type."""
        return any(s.type == sensor_type for s in self.sensors)

    def get_sensor(self, sensor_type: str) -> SensorDesc | None:
        """Return the first sensor matching the given type, or None."""
        for s in self.sensors:
            if s.type == sensor_type:
                return s
        return None

    def get_sensors(self, sensor_type: str) -> list[SensorDesc]:
        """Return all sensors matching the given type."""
        return [s for s in self.sensors if s.type == sensor_type]

    def has_actuator(self, actuator_type: str, side: str = "") -> bool:
        """Check if the robot has an actuator of the given type (and side)."""
        for a in self.actuators:
            if a.type == actuator_type:
                if side and a.side != side:
                    continue
                return True
        return False

    def motor_count(self) -> int:
        """Return the number of motor actuators."""
        return sum(1 for a in self.actuators if a.type == "motor")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the description back to a plain dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "chassis": asdict(self.chassis),
            "sensors": [asdict(s) for s in self.sensors],
            "actuators": [asdict(a) for a in self.actuators],
        }
        if self.payloads:
            result["payload"] = [asdict(p) for p in self.payloads]
        # Convert resolution tuples to lists for YAML compatibility
        for s in result["sensors"]:
            if isinstance(s.get("resolution"), tuple):
                s["resolution"] = list(s["resolution"])
        return result
