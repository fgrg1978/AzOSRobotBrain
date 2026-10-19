# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""First-person camera renderer for the SITL simulator.

Generates synthetic JPEG images from the robot's perspective using raycasting
(Wolfenstein 3D style). The VLM can then analyze these images as if they came
from a real camera.

Rendering approach:
  - Cast rays across the horizontal FOV from the robot's position
  - For each ray, find the nearest wall/obstacle via the World's raycast
  - Draw vertical strips: closer objects appear taller, farther ones shorter
  - Walls (boundary) are gray, obstacles are brown/red, floor is dark green
  - Sky is light blue gradient

Output: JPEG bytes ready to send as PKT_CAMERA payload.
"""

import io
import math

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


# ── Render constants ─────────────────────────────────────────────────────────

IMAGE_WIDTH = 320  # pixels
IMAGE_HEIGHT = 240  # pixels
FOV_DEG = 90  # horizontal field of view
JPEG_QUALITY = 60  # JPEG compression quality (lower = smaller)
MAX_RENDER_DIST = 5000  # mm — beyond this, just draw sky
WALL_HEIGHT_REF = 800  # mm — reference wall height for projection
FOCAL_LENGTH = IMAGE_WIDTH / (2.0 * math.tan(math.radians(FOV_DEG / 2.0)))

# Colors (RGB)
COLOR_SKY_TOP = (135, 180, 220)  # light blue
COLOR_SKY_BOT = (180, 210, 240)  # lighter near horizon
COLOR_FLOOR_NEAR = (60, 80, 50)  # dark green
COLOR_FLOOR_FAR = (90, 110, 80)  # lighter green at horizon
COLOR_WALL = (160, 160, 160)  # gray boundary walls
COLOR_OBSTACLE = (140, 90, 60)  # brown obstacles
COLOR_WALL_DARK = (120, 120, 120)  # shadow side of wall
COLOR_OBS_DARK = (100, 65, 40)  # shadow side of obstacle

# Number of rays to cast (one per horizontal pixel column)
NUM_RAYS = IMAGE_WIDTH

# Camera header for PKT_CAMERA: width(u16 LE) + height(u16 LE) + format(u8)
CAMERA_HDR_SIZE = 5
CAMERA_FMT_JPEG = 1


def _available() -> bool:
    """Check if Pillow is installed."""
    return Image is not None


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linear interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _shade_by_distance(base_color: tuple, dark_color: tuple, dist: float) -> tuple:
    """Darken color based on distance (fog effect)."""
    t = min(dist / MAX_RENDER_DIST, 1.0)
    # Blend toward dark variant, then toward gray fog
    c = _lerp_color(base_color, dark_color, t * 0.5)
    fog = (140, 140, 150)
    return _lerp_color(c, fog, t * 0.7)


def render_frame(
    robot_x: float, robot_y: float, robot_hdg_deg: float, world, obstacles_only: bool = False
) -> bytes:
    """Render a first-person view and return JPEG bytes.

    Args:
        robot_x, robot_y: Robot position in mm.
        robot_hdg_deg: Robot heading in degrees (0=East, 90=North).
        world: World object with raycast() and .obstacles / .walls lists.
        obstacles_only: If True, skip wall rendering (debug).

    Returns:
        JPEG image bytes (no protocol header).
    """
    if not _available():
        # Fallback: 1x1 black JPEG
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), COLOR_SKY_TOP)
    draw = ImageDraw.Draw(img)

    # Draw sky gradient (top half)
    horizon_y = IMAGE_HEIGHT // 2
    for y in range(horizon_y):
        t = y / horizon_y
        color = _lerp_color(COLOR_SKY_TOP, COLOR_SKY_BOT, t)
        draw.line([(0, y), (IMAGE_WIDTH - 1, y)], fill=color)

    # Draw floor gradient (bottom half)
    for y in range(horizon_y, IMAGE_HEIGHT):
        t = (y - horizon_y) / (IMAGE_HEIGHT - horizon_y)
        color = _lerp_color(COLOR_FLOOR_FAR, COLOR_FLOOR_NEAR, t)
        draw.line([(0, y), (IMAGE_WIDTH - 1, y)], fill=color)

    # Cast rays and draw wall strips
    half_fov = FOV_DEG / 2.0
    all_targets = world.obstacles + world.walls

    for col in range(NUM_RAYS):
        # Ray angle: sweep from left to right across FOV
        ray_offset = (col / (NUM_RAYS - 1) - 0.5) * FOV_DEG
        ray_angle = robot_hdg_deg + ray_offset

        # Find distance to nearest surface
        min_dist = math.inf
        hit_is_wall = True

        # Check obstacles first
        for obs in world.obstacles:
            d = obs.ray_dist(robot_x, robot_y, ray_angle)
            if d < min_dist:
                min_dist = d
                hit_is_wall = False

        # Check boundary walls
        for wall in world.walls:
            d = wall.ray_dist(robot_x, robot_y, ray_angle)
            if d < min_dist:
                min_dist = d
                hit_is_wall = True

        if min_dist >= MAX_RENDER_DIST or min_dist <= 0:
            continue

        # Fix fisheye: correct distance by cos of angle offset
        corrected_dist = min_dist * math.cos(math.radians(ray_offset))
        if corrected_dist <= 0:
            continue

        # Calculate wall strip height on screen
        strip_height = (WALL_HEIGHT_REF * FOCAL_LENGTH) / corrected_dist
        strip_height = min(strip_height, IMAGE_HEIGHT)

        strip_top = int(horizon_y - strip_height / 2)
        strip_bot = int(horizon_y + strip_height / 2)

        # Choose color based on hit type
        if hit_is_wall:
            color = _shade_by_distance(COLOR_WALL, COLOR_WALL_DARK, min_dist)
        else:
            color = _shade_by_distance(COLOR_OBSTACLE, COLOR_OBS_DARK, min_dist)

        draw.line([(col, strip_top), (col, strip_bot)], fill=color)

    # Encode as JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def render_camera_payload(robot_x: float, robot_y: float, robot_hdg_deg: float, world) -> bytes:
    """Render and return the full PKT_CAMERA payload (header + JPEG).

    Payload format: width(u16 LE) + height(u16 LE) + format(u8) + JPEG bytes
    """
    import struct

    jpeg = render_frame(robot_x, robot_y, robot_hdg_deg, world)
    header = struct.pack("<HHB", IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FMT_JPEG)
    return header + jpeg
