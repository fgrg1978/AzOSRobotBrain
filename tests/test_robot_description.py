# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/robot_description.py — YAML robot description format."""

import os
import tempfile

import yaml

from planner.robot_description import (
    RobotDescription,
    SensorDesc,
    ActuatorDesc,
    ChassisDesc,
    PayloadDesc,
    MAX_SENSORS,
    MAX_ACTUATORS,
    MAX_PAYLOADS,
    GPIO_UNSET,
    ADDRESS_UNSET,
    RESOLUTION_UNSET,
    WHEEL_BASE_DEFAULT_MM,
    WHEEL_DIAMETER_DEFAULT_MM,
    MAX_SPEED_DEFAULT_PCT,
    DEFAULT_ROBOT_YAML,
)

# ── Sample YAML data ────────────────────────────────────────────────────────

SAMPLE_YAML = {
    "name": "test_bot",
    "type": "wheeled",
    "chassis": {
        "wheel_base_mm": 142,
        "wheel_diameter_mm": 65,
        "max_speed_pct": 80,
    },
    "sensors": [
        {"type": "imu", "model": "mpu6050", "bus": "i2c", "address": 0x68, "rate_hz": 100},
        {
            "type": "rangefinder",
            "model": "hcsr04",
            "gpio_trig": 5,
            "gpio_echo": 6,
            "position": "front",
            "max_range_mm": 4000,
        },
        {"type": "camera", "model": "ov2640", "bus": "csi", "resolution": [640, 480], "fps": 15},
        {"type": "pir", "gpio": 12, "position": "front"},
        {"type": "gps", "model": "neo6m", "bus": "uart", "baud": 9600},
    ],
    "actuators": [
        {
            "type": "motor",
            "model": "dc_brushed",
            "pwm_channel": 0,
            "dir_gpio": 2,
            "encoder_gpio": 3,
            "side": "left",
        },
        {
            "type": "motor",
            "model": "dc_brushed",
            "pwm_channel": 1,
            "dir_gpio": 4,
            "encoder_gpio": 7,
            "side": "right",
        },
        {"type": "buzzer", "gpio": 8},
        {"type": "led", "gpio": 9, "count": 8},
    ],
    "payload": [
        {"type": "spotlight", "gpio": 10, "watts": 10},
    ],
}


def _write_yaml(data: dict) -> str:
    """Write data to a temporary YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path


# ── from_dict tests ──────────────────────────────────────────────────────────


class TestFromDict:
    def test_basic_fields(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.name == "test_bot"
        assert desc.type == "wheeled"

    def test_chassis(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.chassis.wheel_base_mm == 142
        assert desc.chassis.wheel_diameter_mm == 65
        assert desc.chassis.max_speed_pct == 80

    def test_sensor_count(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert len(desc.sensors) == 5

    def test_actuator_count(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert len(desc.actuators) == 4

    def test_payload_count(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert len(desc.payloads) == 1

    def test_imu_sensor(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        imu = desc.get_sensor("imu")
        assert imu is not None
        assert imu.model == "mpu6050"
        assert imu.bus == "i2c"
        assert imu.address == 0x68
        assert imu.rate_hz == 100

    def test_rangefinder_sensor(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        rf = desc.get_sensor("rangefinder")
        assert rf is not None
        assert rf.gpio_trig == 5
        assert rf.gpio_echo == 6
        assert rf.position == "front"
        assert rf.max_range_mm == 4000

    def test_camera_resolution(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        cam = desc.get_sensor("camera")
        assert cam is not None
        assert cam.resolution == (640, 480)
        assert cam.fps == 15

    def test_gps_baud(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        gps = desc.get_sensor("gps")
        assert gps is not None
        assert gps.baud == 9600

    def test_motor_actuators(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.has_actuator("motor", "left")
        assert desc.has_actuator("motor", "right")
        assert desc.motor_count() == 2

    def test_buzzer_actuator(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.has_actuator("buzzer")

    def test_payload_spotlight(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert len(desc.payloads) == 1
        assert desc.payloads[0].type == "spotlight"
        assert desc.payloads[0].watts == 10

    def test_empty_dict(self):
        desc = RobotDescription.from_dict({})
        assert desc.name == "robot"
        assert desc.type == "wheeled"
        assert len(desc.sensors) == 0


# ── from_yaml tests ──────────────────────────────────────────────────────────


class TestFromYaml:
    def test_load_yaml_file(self):
        path = _write_yaml(SAMPLE_YAML)
        try:
            desc = RobotDescription.from_yaml(path)
            assert desc.name == "test_bot"
            assert len(desc.sensors) == 5
            assert len(desc.actuators) == 4
        finally:
            os.unlink(path)

    def test_empty_yaml(self):
        path = _write_yaml({})
        try:
            desc = RobotDescription.from_yaml(path)
            assert desc.name == "robot"
        finally:
            os.unlink(path)


# ── Query helper tests ───────────────────────────────────────────────────────


class TestQueryHelpers:
    def test_has_sensor_true(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.has_sensor("imu")
        assert desc.has_sensor("camera")
        assert desc.has_sensor("pir")

    def test_has_sensor_false(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert not desc.has_sensor("lidar")

    def test_get_sensor_none(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.get_sensor("lidar") is None

    def test_get_sensors_multiple(self):
        data = {
            "sensors": [
                {"type": "rangefinder", "position": "front"},
                {"type": "rangefinder", "position": "rear"},
            ]
        }
        desc = RobotDescription.from_dict(data)
        rfs = desc.get_sensors("rangefinder")
        assert len(rfs) == 2

    def test_has_actuator_with_side(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.has_actuator("motor", "left")
        assert not desc.has_actuator("motor", "top")

    def test_has_actuator_without_side(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        assert desc.has_actuator("motor")
        assert desc.has_actuator("led")
        assert not desc.has_actuator("servo")


# ── Limits tests ─────────────────────────────────────────────────────────────


class TestLimits:
    def test_max_sensors_enforced(self):
        data = {"sensors": [{"type": f"sensor_{i}"} for i in range(MAX_SENSORS + 5)]}
        desc = RobotDescription.from_dict(data)
        assert len(desc.sensors) == MAX_SENSORS

    def test_max_actuators_enforced(self):
        data = {"actuators": [{"type": f"act_{i}"} for i in range(MAX_ACTUATORS + 5)]}
        desc = RobotDescription.from_dict(data)
        assert len(desc.actuators) == MAX_ACTUATORS

    def test_max_payloads_enforced(self):
        data = {"payload": [{"type": f"pay_{i}"} for i in range(MAX_PAYLOADS + 5)]}
        desc = RobotDescription.from_dict(data)
        assert len(desc.payloads) == MAX_PAYLOADS


# ── Serialization tests ─────────────────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        d = desc.to_dict()
        desc2 = RobotDescription.from_dict(d)
        assert desc2.name == desc.name
        assert desc2.type == desc.type
        assert desc2.chassis.wheel_base_mm == desc.chassis.wheel_base_mm
        assert len(desc2.sensors) == len(desc.sensors)
        assert len(desc2.actuators) == len(desc.actuators)

    def test_to_dict_has_required_keys(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        d = desc.to_dict()
        assert "name" in d
        assert "type" in d
        assert "chassis" in d
        assert "sensors" in d
        assert "actuators" in d

    def test_to_dict_payload_present(self):
        desc = RobotDescription.from_dict(SAMPLE_YAML)
        d = desc.to_dict()
        assert "payload" in d
        assert len(d["payload"]) == 1

    def test_to_dict_no_payload_when_empty(self):
        desc = RobotDescription.from_dict({"name": "bare"})
        d = desc.to_dict()
        assert "payload" not in d


# ── Default values tests ────────────────────────────────────────────────────


class TestDefaults:
    def test_default_chassis(self):
        desc = RobotDescription()
        assert desc.chassis.wheel_base_mm == WHEEL_BASE_DEFAULT_MM
        assert desc.chassis.wheel_diameter_mm == WHEEL_DIAMETER_DEFAULT_MM
        assert desc.chassis.max_speed_pct == MAX_SPEED_DEFAULT_PCT

    def test_sensor_defaults(self):
        s = SensorDesc(type="test")
        assert s.gpio == GPIO_UNSET
        assert s.address == ADDRESS_UNSET
        assert s.resolution == RESOLUTION_UNSET

    def test_actuator_defaults(self):
        a = ActuatorDesc(type="test")
        assert a.gpio == GPIO_UNSET
        assert a.pwm_channel == -1
        assert a.count == 1
