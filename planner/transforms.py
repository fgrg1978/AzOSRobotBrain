# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Coordinate Transforms (AT3) — static frame transforms for robots.

Simple TF-lite: tree of transforms from base_link to sensor frames.
Used to convert sensor readings to the robot's base frame.

Frame tree example:
    base_link
    +-- imu_link        (0, 0, 50mm, 0deg)
    +-- camera_link     (100mm, 0, 80mm, 0deg)
    +-- lidar_link      (0, 0, 120mm, 0deg)
    +-- range_front     (150mm, 0, 30mm, 0deg)
    +-- range_right     (0, -75mm, 30mm, 90deg)

Only yaw rotation is supported (2D rotation on x,y; z is additive).
This keeps the math simple and sufficient for ground robots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Constants ────────────────────────────────────────────────────────────────

MAX_FRAMES = 32
ROOT_FRAME_NAME = "base_link"
ROOT_PARENT = ""
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi

# Default sensor frame offsets (mm) — used when building from RobotDescription
# and no explicit offset is provided. (x=forward, y=left, z=up)
DEFAULT_IMU_OFFSET_Z_MM = 50.0
DEFAULT_CAMERA_OFFSET_X_MM = 100.0
DEFAULT_CAMERA_OFFSET_Z_MM = 80.0
DEFAULT_LIDAR_OFFSET_Z_MM = 120.0
DEFAULT_RANGE_FRONT_OFFSET_X_MM = 150.0
DEFAULT_RANGE_FRONT_OFFSET_Z_MM = 30.0
DEFAULT_RANGE_SIDE_OFFSET_Y_MM = 75.0
DEFAULT_RANGE_SIDE_OFFSET_Z_MM = 30.0
RANGE_RIGHT_YAW_DEG = -90.0
RANGE_LEFT_YAW_DEG = 90.0
RANGE_REAR_YAW_DEG = 180.0
DEFAULT_GPS_OFFSET_Z_MM = 100.0


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Transform:
    """3D transform: translation (mm) + rotation (yaw only, degrees)."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    yaw_deg: float = 0.0

    def compose(self, other: Transform) -> Transform:
        """Compose self followed by other: self * other.

        Applies self's rotation to other's translation, then adds.
        """
        yaw_rad = self.yaw_deg * DEGREES_TO_RADIANS
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # Rotate other's translation by self's yaw
        rx = cos_y * other.x_mm - sin_y * other.y_mm
        ry = sin_y * other.x_mm + cos_y * other.y_mm
        return Transform(
            x_mm=self.x_mm + rx,
            y_mm=self.y_mm + ry,
            z_mm=self.z_mm + other.z_mm,
            yaw_deg=self.yaw_deg + other.yaw_deg,
        )

    def inverse(self) -> Transform:
        """Return the inverse transform."""
        yaw_rad = -self.yaw_deg * DEGREES_TO_RADIANS
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # Rotate the negated translation by the inverse yaw
        rx = cos_y * (-self.x_mm) - sin_y * (-self.y_mm)
        ry = sin_y * (-self.x_mm) + cos_y * (-self.y_mm)
        return Transform(
            x_mm=rx,
            y_mm=ry,
            z_mm=-self.z_mm,
            yaw_deg=-self.yaw_deg,
        )

    def apply_to_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Apply this transform to a 3D point."""
        yaw_rad = self.yaw_deg * DEGREES_TO_RADIANS
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        rx = cos_y * x - sin_y * y + self.x_mm
        ry = sin_y * x + cos_y * y + self.y_mm
        rz = z + self.z_mm
        return (rx, ry, rz)


@dataclass
class Frame:
    """A named coordinate frame with a parent relationship."""

    name: str
    parent: str  # parent frame name ("" = root)
    transform: Transform


class TransformTree:
    """Static transform tree for robot coordinate frames.

    Frames form a tree rooted at base_link. Transforms can be composed
    to convert points between any two frames in the tree.
    """

    def __init__(self) -> None:
        self._frames: dict[str, Frame] = {}
        # Add root frame
        self.add_frame(ROOT_FRAME_NAME, ROOT_PARENT, Transform())

    def add_frame(self, name: str, parent: str, tf: Transform) -> bool:
        """Add a frame to the tree. Returns False if tree is full."""
        if len(self._frames) >= MAX_FRAMES and name not in self._frames:
            return False
        self._frames[name] = Frame(name=name, parent=parent, transform=tf)
        return True

    def _path_to_root(self, frame_name: str) -> list[str] | None:
        """Return the chain of frame names from the given frame up to root."""
        path: list[str] = []
        current = frame_name
        visited: set[str] = set()
        while current:
            if current in visited:
                return None  # cycle detected
            if current not in self._frames:
                return None  # frame not found
            visited.add(current)
            path.append(current)
            current = self._frames[current].parent
        return path

    def get_transform(self, from_frame: str, to_frame: str) -> Transform | None:
        """Get composed transform from one frame to another.

        Finds the common ancestor, composes transforms up from from_frame
        to the ancestor, then down to to_frame.
        """
        if from_frame == to_frame:
            return Transform()

        path_from = self._path_to_root(from_frame)
        path_to = self._path_to_root(to_frame)
        if path_from is None or path_to is None:
            return None

        # Find lowest common ancestor
        set_from = set(path_from)
        ancestor: str | None = None
        for name in path_to:
            if name in set_from:
                ancestor = name
                break
        if ancestor is None:
            return None

        # Compose from from_frame up to ancestor (each inverted)
        tf = Transform()
        for name in path_from:
            if name == ancestor:
                break
            frame = self._frames[name]
            tf = tf.compose(frame.transform.inverse())

        # Compose from ancestor down to to_frame
        # First collect the path from ancestor to to_frame
        down_path: list[str] = []
        for name in path_to:
            if name == ancestor:
                break
            down_path.append(name)
        down_path.reverse()

        for name in down_path:
            frame = self._frames[name]
            tf = tf.compose(frame.transform)

        return tf

    def transform_point(
        self, point_mm: tuple[float, float, float], from_frame: str, to_frame: str
    ) -> tuple[float, float, float] | None:
        """Transform a 3D point between frames."""
        tf = self.get_transform(from_frame, to_frame)
        if tf is None:
            return None
        return tf.apply_to_point(point_mm[0], point_mm[1], point_mm[2])

    def list_frames(self) -> list[str]:
        """Return all frame names in the tree."""
        return list(self._frames.keys())

    @classmethod
    def from_robot_description(cls, desc: "object") -> TransformTree:
        """Build transform tree from a RobotDescription.

        Assigns default offsets based on sensor type and position.
        The desc argument should be a planner.robot_description.RobotDescription.
        """
        tree = cls()

        for sensor in getattr(desc, "sensors", []):
            s_type = sensor.type
            position = sensor.position
            frame_name = _sensor_frame_name(s_type, position)

            if s_type == "imu":
                tf = Transform(z_mm=DEFAULT_IMU_OFFSET_Z_MM)
            elif s_type == "camera":
                tf = Transform(x_mm=DEFAULT_CAMERA_OFFSET_X_MM, z_mm=DEFAULT_CAMERA_OFFSET_Z_MM)
            elif s_type == "lidar":
                tf = Transform(z_mm=DEFAULT_LIDAR_OFFSET_Z_MM)
            elif s_type == "rangefinder":
                tf = _rangefinder_transform(position)
            elif s_type == "gps":
                tf = Transform(z_mm=DEFAULT_GPS_OFFSET_Z_MM)
            else:
                tf = Transform()

            tree.add_frame(frame_name, ROOT_FRAME_NAME, tf)

        return tree


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sensor_frame_name(sensor_type: str, position: str) -> str:
    """Generate a frame name from sensor type and position."""
    if position:
        return f"{sensor_type}_{position}"
    return f"{sensor_type}_link"


def _rangefinder_transform(position: str) -> Transform:
    """Return a default transform for a rangefinder at the given position."""
    if position == "front":
        return Transform(
            x_mm=DEFAULT_RANGE_FRONT_OFFSET_X_MM,
            z_mm=DEFAULT_RANGE_FRONT_OFFSET_Z_MM,
        )
    elif position == "right":
        return Transform(
            y_mm=-DEFAULT_RANGE_SIDE_OFFSET_Y_MM,
            z_mm=DEFAULT_RANGE_SIDE_OFFSET_Z_MM,
            yaw_deg=RANGE_RIGHT_YAW_DEG,
        )
    elif position == "left":
        return Transform(
            y_mm=DEFAULT_RANGE_SIDE_OFFSET_Y_MM,
            z_mm=DEFAULT_RANGE_SIDE_OFFSET_Z_MM,
            yaw_deg=RANGE_LEFT_YAW_DEG,
        )
    elif position == "rear":
        return Transform(
            x_mm=-DEFAULT_RANGE_FRONT_OFFSET_X_MM,
            z_mm=DEFAULT_RANGE_FRONT_OFFSET_Z_MM,
            yaw_deg=RANGE_REAR_YAW_DEG,
        )
    return Transform(z_mm=DEFAULT_RANGE_FRONT_OFFSET_Z_MM)
