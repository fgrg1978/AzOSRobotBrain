# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/transforms.py — coordinate transform tree."""

import math

from planner.transforms import (
    Transform,
    Frame,
    TransformTree,
    MAX_FRAMES,
    ROOT_FRAME_NAME,
    DEGREES_TO_RADIANS,
    DEFAULT_IMU_OFFSET_Z_MM,
    DEFAULT_CAMERA_OFFSET_X_MM,
    DEFAULT_CAMERA_OFFSET_Z_MM,
    DEFAULT_RANGE_FRONT_OFFSET_X_MM,
    DEFAULT_RANGE_FRONT_OFFSET_Z_MM,
    DEFAULT_RANGE_SIDE_OFFSET_Y_MM,
    RANGE_RIGHT_YAW_DEG,
)

# Tolerance for floating-point comparisons
FLOAT_TOLERANCE = 1e-6


def _approx(a: float, b: float, tol: float = FLOAT_TOLERANCE) -> bool:
    return abs(a - b) < tol


# ── Transform basic math ────────────────────────────────────────────────────


class TestTransform:
    def test_identity_compose(self):
        t = Transform()
        t2 = Transform(x_mm=100.0, y_mm=50.0, z_mm=30.0, yaw_deg=45.0)
        result = t.compose(t2)
        assert _approx(result.x_mm, 100.0)
        assert _approx(result.y_mm, 50.0)
        assert _approx(result.z_mm, 30.0)
        assert _approx(result.yaw_deg, 45.0)

    def test_translation_compose(self):
        t1 = Transform(x_mm=100.0)
        t2 = Transform(x_mm=50.0)
        result = t1.compose(t2)
        assert _approx(result.x_mm, 150.0)
        assert _approx(result.y_mm, 0.0)

    def test_rotation_compose(self):
        # Rotate 90 degrees, then translate forward (x=100)
        # After 90deg yaw, forward (x) becomes leftward (y)
        t1 = Transform(yaw_deg=90.0)
        t2 = Transform(x_mm=100.0)
        result = t1.compose(t2)
        assert _approx(result.x_mm, 0.0, tol=0.01)
        assert _approx(result.y_mm, 100.0, tol=0.01)
        assert _approx(result.yaw_deg, 90.0)

    def test_inverse_identity(self):
        t = Transform()
        inv = t.inverse()
        assert _approx(inv.x_mm, 0.0)
        assert _approx(inv.y_mm, 0.0)
        assert _approx(inv.z_mm, 0.0)
        assert _approx(inv.yaw_deg, 0.0)

    def test_inverse_translation(self):
        t = Transform(x_mm=100.0, y_mm=50.0, z_mm=30.0)
        inv = t.inverse()
        assert _approx(inv.x_mm, -100.0)
        assert _approx(inv.y_mm, -50.0)
        assert _approx(inv.z_mm, -30.0)

    def test_inverse_roundtrip(self):
        t = Transform(x_mm=100.0, y_mm=50.0, z_mm=30.0, yaw_deg=45.0)
        inv = t.inverse()
        result = t.compose(inv)
        assert _approx(result.x_mm, 0.0, tol=0.01)
        assert _approx(result.y_mm, 0.0, tol=0.01)
        assert _approx(result.z_mm, 0.0, tol=0.01)
        assert _approx(result.yaw_deg, 0.0, tol=0.01)

    def test_apply_to_point_identity(self):
        t = Transform()
        x, y, z = t.apply_to_point(10.0, 20.0, 30.0)
        assert _approx(x, 10.0)
        assert _approx(y, 20.0)
        assert _approx(z, 30.0)

    def test_apply_to_point_translation(self):
        t = Transform(x_mm=100.0, y_mm=50.0, z_mm=10.0)
        x, y, z = t.apply_to_point(0.0, 0.0, 0.0)
        assert _approx(x, 100.0)
        assert _approx(y, 50.0)
        assert _approx(z, 10.0)

    def test_apply_to_point_rotation_90(self):
        t = Transform(yaw_deg=90.0)
        x, y, z = t.apply_to_point(100.0, 0.0, 0.0)
        assert _approx(x, 0.0, tol=0.01)
        assert _approx(y, 100.0, tol=0.01)
        assert _approx(z, 0.0)

    def test_z_is_additive(self):
        t1 = Transform(z_mm=10.0)
        t2 = Transform(z_mm=20.0)
        result = t1.compose(t2)
        assert _approx(result.z_mm, 30.0)


# ── TransformTree ────────────────────────────────────────────────────────────


class TestTransformTree:
    def test_root_exists(self):
        tree = TransformTree()
        assert ROOT_FRAME_NAME in tree.list_frames()

    def test_add_frame(self):
        tree = TransformTree()
        tree.add_frame("imu_link", ROOT_FRAME_NAME, Transform(z_mm=50.0))
        assert "imu_link" in tree.list_frames()

    def test_identity_transform_same_frame(self):
        tree = TransformTree()
        tf = tree.get_transform(ROOT_FRAME_NAME, ROOT_FRAME_NAME)
        assert tf is not None
        assert _approx(tf.x_mm, 0.0)
        assert _approx(tf.yaw_deg, 0.0)

    def test_parent_to_child(self):
        tree = TransformTree()
        tree.add_frame("cam", ROOT_FRAME_NAME, Transform(x_mm=100.0, z_mm=80.0))
        tf = tree.get_transform(ROOT_FRAME_NAME, "cam")
        assert tf is not None
        assert _approx(tf.x_mm, 100.0)
        assert _approx(tf.z_mm, 80.0)

    def test_child_to_parent(self):
        tree = TransformTree()
        tree.add_frame("cam", ROOT_FRAME_NAME, Transform(x_mm=100.0, z_mm=80.0))
        tf = tree.get_transform("cam", ROOT_FRAME_NAME)
        assert tf is not None
        assert _approx(tf.x_mm, -100.0)
        assert _approx(tf.z_mm, -80.0)

    def test_sibling_transform(self):
        tree = TransformTree()
        tree.add_frame("imu", ROOT_FRAME_NAME, Transform(z_mm=50.0))
        tree.add_frame("cam", ROOT_FRAME_NAME, Transform(x_mm=100.0, z_mm=80.0))
        # imu -> base_link -> cam
        tf = tree.get_transform("imu", "cam")
        assert tf is not None
        assert _approx(tf.x_mm, 100.0)
        assert _approx(tf.z_mm, 30.0)  # 80 - 50

    def test_unknown_frame_returns_none(self):
        tree = TransformTree()
        tf = tree.get_transform(ROOT_FRAME_NAME, "nonexistent")
        assert tf is None

    def test_transform_point(self):
        tree = TransformTree()
        tree.add_frame("sensor", ROOT_FRAME_NAME, Transform(x_mm=100.0, y_mm=50.0))
        # A point at (10, 0, 0) in sensor frame
        pt = tree.transform_point((10.0, 0.0, 0.0), "sensor", ROOT_FRAME_NAME)
        assert pt is not None
        # sensor is at (100, 50) from base, so point in base = (-100+10, -50, 0)
        # Actually: sensor->base = inverse of base->sensor
        # base->sensor = (100, 50, 0, 0deg), inverse = (-100, -50, 0)
        # point (10,0,0) in sensor frame => in base frame:
        # apply inverse: (-100+10, -50+0, 0) = (-90, -50, 0)
        assert _approx(pt[0], -90.0)
        assert _approx(pt[1], -50.0)

    def test_transform_point_with_rotation(self):
        tree = TransformTree()
        tree.add_frame("right_range", ROOT_FRAME_NAME, Transform(y_mm=-75.0, yaw_deg=-90.0))
        # A point at (100, 0, 0) in right_range frame (i.e., 100mm along
        # the sensor's x-axis). The sensor is rotated -90deg from base,
        # so its x-axis points in the -y direction of base.
        # Inverse transform: yaw=+90, translation becomes (-75, ~0).
        # Applying +90 rotation to (100,0) gives (0, 100).
        # Result in base: (-75+0, 0+100, 0) = (-75, 100, 0).
        pt = tree.transform_point((100.0, 0.0, 0.0), "right_range", ROOT_FRAME_NAME)
        assert pt is not None
        assert _approx(pt[0], -75.0, tol=0.1)
        assert _approx(pt[1], 100.0, tol=0.1)

    def test_max_frames(self):
        tree = TransformTree()
        # Root already takes 1 slot
        for i in range(MAX_FRAMES - 1):
            ok = tree.add_frame(f"f{i}", ROOT_FRAME_NAME, Transform())
            assert ok
        # Next should fail
        ok = tree.add_frame("overflow", ROOT_FRAME_NAME, Transform())
        assert not ok

    def test_list_frames(self):
        tree = TransformTree()
        tree.add_frame("a", ROOT_FRAME_NAME, Transform())
        tree.add_frame("b", ROOT_FRAME_NAME, Transform())
        frames = tree.list_frames()
        assert ROOT_FRAME_NAME in frames
        assert "a" in frames
        assert "b" in frames
        assert len(frames) == 3

    def test_chain_three_deep(self):
        tree = TransformTree()
        tree.add_frame("arm", ROOT_FRAME_NAME, Transform(x_mm=200.0))
        tree.add_frame("gripper", "arm", Transform(x_mm=100.0))
        tf = tree.get_transform(ROOT_FRAME_NAME, "gripper")
        assert tf is not None
        assert _approx(tf.x_mm, 300.0)

    def test_chain_reverse(self):
        tree = TransformTree()
        tree.add_frame("arm", ROOT_FRAME_NAME, Transform(x_mm=200.0))
        tree.add_frame("gripper", "arm", Transform(x_mm=100.0))
        tf = tree.get_transform("gripper", ROOT_FRAME_NAME)
        assert tf is not None
        assert _approx(tf.x_mm, -300.0)


# ── from_robot_description ───────────────────────────────────────────────────


class TestFromRobotDescription:
    def _make_desc(self):
        """Create a minimal mock RobotDescription."""
        from planner.robot_description import RobotDescription

        return RobotDescription.from_dict(
            {
                "name": "test",
                "type": "wheeled",
                "sensors": [
                    {"type": "imu", "model": "mpu6050"},
                    {"type": "camera", "model": "ov2640"},
                    {"type": "rangefinder", "position": "front"},
                    {"type": "rangefinder", "position": "right"},
                    {"type": "gps", "model": "neo6m"},
                ],
            }
        )

    def test_frames_created(self):
        desc = self._make_desc()
        tree = TransformTree.from_robot_description(desc)
        frames = tree.list_frames()
        assert ROOT_FRAME_NAME in frames
        assert "imu_link" in frames
        assert "camera_link" in frames
        assert "rangefinder_front" in frames
        assert "rangefinder_right" in frames
        assert "gps_link" in frames

    def test_imu_offset(self):
        desc = self._make_desc()
        tree = TransformTree.from_robot_description(desc)
        tf = tree.get_transform(ROOT_FRAME_NAME, "imu_link")
        assert tf is not None
        assert _approx(tf.z_mm, DEFAULT_IMU_OFFSET_Z_MM)

    def test_camera_offset(self):
        desc = self._make_desc()
        tree = TransformTree.from_robot_description(desc)
        tf = tree.get_transform(ROOT_FRAME_NAME, "camera_link")
        assert tf is not None
        assert _approx(tf.x_mm, DEFAULT_CAMERA_OFFSET_X_MM)
        assert _approx(tf.z_mm, DEFAULT_CAMERA_OFFSET_Z_MM)

    def test_rangefinder_front_offset(self):
        desc = self._make_desc()
        tree = TransformTree.from_robot_description(desc)
        tf = tree.get_transform(ROOT_FRAME_NAME, "rangefinder_front")
        assert tf is not None
        assert _approx(tf.x_mm, DEFAULT_RANGE_FRONT_OFFSET_X_MM)

    def test_rangefinder_right_yaw(self):
        desc = self._make_desc()
        tree = TransformTree.from_robot_description(desc)
        tf = tree.get_transform(ROOT_FRAME_NAME, "rangefinder_right")
        assert tf is not None
        assert _approx(tf.yaw_deg, RANGE_RIGHT_YAW_DEG)
        assert _approx(tf.y_mm, -DEFAULT_RANGE_SIDE_OFFSET_Y_MM)
