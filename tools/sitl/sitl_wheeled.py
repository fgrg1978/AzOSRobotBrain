# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""SITL (Software-In-The-Loop) simulator for a wheeled differential-drive robot.

Simulates the VF2 robot over TCP — connects to the brain server just like a real
robot would, sends SensorPackets, receives ActuatorCmds.

Physics model (2D, top-down):
  - Position (x_mm, y_mm), heading_cdeg (0=North, 90=East)
  - Velocity from actuator channels (speed_l, speed_r), range -100..100 %
  - Encoders: integer ticks, 1000 ticks/m
  - Rangefinders: ray-cast against rectangular obstacles
  - Battery: slow drain, starts at 7400 mV

Usage:
    python tools/sitl/sitl_wheeled.py [--host 127.0.0.1] [--port 9000]
                                       [--scenario scenarios/empty.yaml]
                                       [--hz 20] [--cam-hz 2] [--duration 60]
"""

import argparse
import asyncio
import math
import struct
import sys
import time
import os
import yaml

_SITL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_SITL_DIR))
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SITL_DIR)

import protocol
from robot_link import RobotLink
from protocol import (
    parse_packet,
    SensorPacket,
    StatusPacket,
    ActuatorCmd,
    SENSOR_PACKET,
    CAMERA_FRAME,
    STATUS,
    ACTUATOR_CMD,
    ROBOT_WHEELED,
    FLAG_EMERGENCY,
)
from camera import render_camera_payload

# ── Physics constants ─────────────────────────────────────────────────────────

WHEEL_BASE_MM = 142  # Yahboom chassis 310
TICKS_PER_M = 1000  # encoder resolution
MM_PER_M = 1000  # millimeters per meter (for unit conversion)
MAX_SPEED_MM_S = 500  # 100% = 500 mm/s
BATTERY_START = 7400  # mV
BATTERY_DRAIN = 0.5  # mV per second (idle) — real drain would be higher under load
SENSOR_NOISE = 2  # ± ticks noise on encoders

# LiDAR simulation
LIDAR_NUM_RAYS = 360  # one ray per degree
LIDAR_MAX_MM = 12000  # LD19 max range
LIDAR_NOISE_MM = 10  # ± mm noise per ray


# ── World / obstacle model ────────────────────────────────────────────────────


class Rect:
    """Axis-aligned rectangular obstacle."""

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h

    def contains(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def ray_dist(self, ox: float, oy: float, angle_deg: float) -> float:
        """Distance from (ox, oy) in angle_deg direction to this rect edge. inf if no hit."""
        dx = math.cos(math.radians(angle_deg))
        dy = math.sin(math.radians(angle_deg))
        # Slab method
        with_inf = lambda a, b: (b - a) / (dx if abs(dx) > 1e-9 else 1e-9)
        tx1 = (self.x1 - ox) / (dx if abs(dx) > 1e-9 else 1e-9)
        tx2 = (self.x2 - ox) / (dx if abs(dx) > 1e-9 else 1e-9)
        ty1 = (self.y1 - oy) / (dy if abs(dy) > 1e-9 else 1e-9)
        ty2 = (self.y2 - oy) / (dy if abs(dy) > 1e-9 else 1e-9)
        tmin = max(min(tx1, tx2), min(ty1, ty2))
        tmax = min(max(tx1, tx2), max(ty1, ty2))
        if tmax < 0 or tmin > tmax:
            return math.inf
        return tmin if tmin > 0 else (tmax if tmax > 0 else math.inf)


class World:
    def __init__(self, obstacles: list[Rect], width_mm: float = 10000, height_mm: float = 10000):
        self.obstacles = obstacles
        self.width = width_mm
        self.height = height_mm
        # Add boundary walls
        T = 100  # wall thickness
        self.walls = [
            Rect(-T, -T, T, height_mm + T),  # left
            Rect(width_mm, -T, T, height_mm + T),  # right
            Rect(-T, -T, width_mm + T, T),  # bottom
            Rect(-T, height_mm, width_mm + T, T),  # top
        ]

    def raycast(self, x: float, y: float, angle_deg: float) -> float:
        """Return distance mm to nearest obstacle/wall in given direction."""
        dists = [r.ray_dist(x, y, angle_deg) for r in self.obstacles + self.walls]
        return min(dists) if dists else math.inf

    def collides(self, x: float, y: float, radius: float = 100) -> bool:
        for r in self.obstacles + self.walls:
            if r.contains(x, y):
                return True
        return False

    @classmethod
    def from_scenario(cls, scenario: dict) -> "World":
        obs = []
        for o in scenario.get("obstacles", []):
            obs.append(Rect(o["x"], o["y"], o["w"], o["h"]))
        return cls(
            obstacles=obs,
            width_mm=scenario.get("width_mm", 10000),
            height_mm=scenario.get("height_mm", 10000),
        )


# ── Robot state ───────────────────────────────────────────────────────────────


class RobotSim:
    def __init__(
        self, world: World, start_x: float = 1000, start_y: float = 1000, start_hdg_deg: float = 0
    ):
        self.world = world
        self.x = float(start_x)  # mm
        self.y = float(start_y)  # mm
        self.hdg_deg = float(start_hdg_deg)  # degrees, 0=East

        self.speed_l = 0  # -100..100 %
        self.speed_r = 0  # -100..100 %
        self.flags = 0

        self.enc_l = 0  # encoder ticks
        self.enc_r = 0
        self.odom_dist_mm = 0
        self.odom_hdg_cdeg = 0

        self.battery_mv = BATTERY_START
        self.timestamp_ms = int(time.time() * 1000)

        self._last_tick = time.monotonic()

    def apply_cmd(self, cmd: ActuatorCmd):
        if cmd.flags & FLAG_EMERGENCY:
            self.speed_l = 0
            self.speed_r = 0
        elif len(cmd.channels) >= 2:
            self.speed_l = max(-100, min(100, cmd.channels[0]))
            self.speed_r = max(-100, min(100, cmd.channels[1]))
        self.flags = cmd.flags

    def step(self):
        """Advance physics by wall-clock elapsed time."""
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        vl = self.speed_l / 100.0 * MAX_SPEED_MM_S  # mm/s
        vr = self.speed_r / 100.0 * MAX_SPEED_MM_S

        v_center = (vl + vr) / 2.0
        omega = (vr - vl) / WHEEL_BASE_MM  # rad/s

        # Update heading
        delta_hdg_rad = omega * dt
        self.hdg_deg = (self.hdg_deg + math.degrees(delta_hdg_rad)) % 360

        # Update position
        dx = v_center * dt * math.cos(math.radians(self.hdg_deg))
        dy = v_center * dt * math.sin(math.radians(self.hdg_deg))
        nx = self.x + dx
        ny = self.y + dy

        # Collision: stop if blocked
        if not self.world.collides(nx, ny):
            self.x = nx
            self.y = ny
        else:
            self.speed_l = 0
            self.speed_r = 0

        # Encoders
        ticks_l = int(vl * dt / MM_PER_M * TICKS_PER_M)
        ticks_r = int(vr * dt / MM_PER_M * TICKS_PER_M)
        self.enc_l += ticks_l
        self.enc_r += ticks_r

        # Odometry integration
        dist_delta = v_center * dt
        self.odom_dist_mm += int(dist_delta)
        self.odom_hdg_cdeg = int(self.hdg_deg * 100) % 36000

        # Battery drain
        load_factor = (abs(self.speed_l) + abs(self.speed_r)) / 200.0
        self.battery_mv -= BATTERY_DRAIN * dt * (1 + load_factor * 3)
        self.battery_mv = max(0, self.battery_mv)

        self.timestamp_ms = int(time.time() * 1000)

    @property
    def range_front_mm(self) -> int:
        d = self.world.raycast(self.x, self.y, self.hdg_deg)
        return int(min(d, 65535))

    @property
    def range_right_mm(self) -> int:
        d = self.world.raycast(self.x, self.y, (self.hdg_deg - 90) % 360)
        return int(min(d, 65535))

    def lidar_scan(self) -> list[tuple[int, int]]:
        """Simulate 360° LiDAR scan. Returns [(angle_cdeg, distance_mm), ...]."""
        import random

        scan = []
        for deg in range(LIDAR_NUM_RAYS):
            abs_angle = self.hdg_deg + deg
            dist = self.world.raycast(self.x, self.y, abs_angle)
            dist = min(dist, LIDAR_MAX_MM)
            if dist < LIDAR_MAX_MM:
                dist += random.randint(-LIDAR_NOISE_MM, LIDAR_NOISE_MM)
                dist = max(0, dist)
            # angle relative to robot heading, in centidegrees
            angle_cdeg = deg * 100
            scan.append((angle_cdeg, int(dist)))
        return scan

    def sensor_packet(self) -> SensorPacket:
        import random

        noise = lambda: random.randint(-SENSOR_NOISE, SENSOR_NOISE)
        return SensorPacket(
            timestamp_ms=self.timestamp_ms,
            battery_mv=int(self.battery_mv),
            accel_mg=(noise(), noise(), 1000 + noise()),  # simulated flat
            gyro_mdps=(noise(), noise(), noise()),
            odom_dist_mm=self.odom_dist_mm,
            odom_hdg_cdeg=self.odom_hdg_cdeg,
            encoder_l=self.enc_l,
            encoder_r=self.enc_r,
            range_front_mm=self.range_front_mm,
            range_right_mm=self.range_right_mm,
        )

    def status_packet(self) -> StatusPacket:
        return StatusPacket(
            mode=1,
            tasks_ok=8,
            canary_ok=8,
            uptime_s=int(time.time()),
            robot_type=ROBOT_WHEELED,
        )

    def __repr__(self) -> str:
        return (
            f"RobotSim(x={self.x:.0f} y={self.y:.0f} "
            f"hdg={self.hdg_deg:.1f}° "
            f"batt={self.battery_mv:.0f}mV "
            f"front={self.range_front_mm}mm)"
        )


# ── SITL client ───────────────────────────────────────────────────────────────


class SITLClient:
    def __init__(
        self,
        robot: RobotSim,
        host: str,
        port: int,
        sensor_hz: float = 20,
        cam_hz: float = 2,
        duration: float = 0,
        state_file: str = "/tmp/sitl_state.json",
    ):
        self.robot = robot
        self.host = host
        self.port = port
        self.sensor_hz = sensor_hz
        self.cam_hz = cam_hz
        self.duration = duration  # 0 = run forever
        self.state_file = state_file
        self._running = False

    async def connect_and_run(self):
        print(f"[SITL] Connecting to {self.host}:{self.port}...")
        link = await RobotLink.connect(self.host, self.port)
        print(f"[SITL] Connected. Robot: {self.robot}")
        self._running = True

        # Send initial STATUS
        status = self.robot.status_packet()
        link.queue_packet(STATUS, status.to_bytes())
        await link.writer.drain()

        start = time.monotonic()
        tasks = [
            asyncio.create_task(self._sensor_loop(link)),
            asyncio.create_task(self._camera_loop(link)),
            asyncio.create_task(self._recv_loop(link)),
            asyncio.create_task(self._physics_loop()),
        ]

        if self.duration > 0:
            await asyncio.sleep(self.duration)
            self._running = False
            for t in tasks:
                t.cancel()
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        link.close()
        await link.wait_closed()
        print(f"[SITL] Done. Final state: {self.robot}")

    async def _physics_loop(self):
        interval = 0.01  # 100 Hz physics
        export_interval = 0.1  # write state file at 10 Hz
        last_export = 0.0
        while self._running:
            self.robot.step()
            now = time.monotonic()
            if self.state_file and now - last_export >= export_interval:
                self._export_state()
                last_export = now
            await asyncio.sleep(interval)

    def _export_state(self):
        """Write robot state to JSON file for external visualizers."""
        import json

        state = {
            "x": self.robot.x,
            "y": self.robot.y,
            "hdg_deg": self.robot.hdg_deg,
            "speed_l": self.robot.speed_l,
            "speed_r": self.robot.speed_r,
            "battery_mv": self.robot.battery_mv,
            "range_front_mm": self.robot.range_front_mm,
            "range_right_mm": self.robot.range_right_mm,
            "enc_l": self.robot.enc_l,
            "enc_r": self.robot.enc_r,
            "odom_dist_mm": self.robot.odom_dist_mm,
            "timestamp_ms": self.robot.timestamp_ms,
            "lidar_rays": self.robot.lidar_scan(),
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except OSError:
            pass

    async def _sensor_loop(self, link: RobotLink):
        interval = 1.0 / self.sensor_hz
        while self._running:
            pkt = self.robot.sensor_packet()
            link.queue_packet(SENSOR_PACKET, pkt.to_bytes())
            await link.writer.drain()
            await asyncio.sleep(interval)

    async def _camera_loop(self, link: RobotLink):
        """Send rendered first-person camera frames at cam_hz."""
        if self.cam_hz <= 0:
            return
        interval = 1.0 / self.cam_hz
        frame_count = 0
        while self._running:
            payload = render_camera_payload(
                self.robot.x,
                self.robot.y,
                self.robot.hdg_deg,
                self.robot.world,
            )
            link.queue_packet(CAMERA_FRAME, payload)
            await link.writer.drain()
            frame_count += 1
            if frame_count == 1:
                jpeg_size = len(payload) - 5  # subtract header
                print(f"[SITL] Camera active: {self.cam_hz} Hz, " f"JPEG ~{jpeg_size} bytes/frame")
            await asyncio.sleep(interval)

    async def _recv_loop(self, link: RobotLink):
        while self._running:
            try:
                result = await link.read_packet()
                if result is None:
                    print("[SITL] Invalid packet from server")
                    continue
                pkt_type, payload = result
                if pkt_type == ACTUATOR_CMD:
                    cmd = ActuatorCmd.from_bytes(payload)
                    self.robot.apply_cmd(cmd)
                    print(f"[SITL] CMD: ch={cmd.channels} flags={cmd.flags:#04x} | {self.robot}")
            except asyncio.IncompleteReadError:
                print("[SITL] Server disconnected")
                self._running = False
                break
            except Exception as e:
                print(f"[SITL] recv error: {e}")
                break


# ── Entry point ───────────────────────────────────────────────────────────────


def load_scenario(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="SITL Wheeled Robot Simulator")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--scenario", default="")
    ap.add_argument("--hz", type=float, default=20)
    ap.add_argument("--cam-hz", type=float, default=2)
    ap.add_argument(
        "--duration", type=float, default=0, help="Run for N seconds then exit (0=forever)"
    )
    ap.add_argument(
        "--state-file",
        default="/tmp/sitl_state.json",
        help="JSON file to export robot state for viz.py",
    )
    args = ap.parse_args()

    scenario = load_scenario(args.scenario)
    world = World.from_scenario(scenario)
    start = scenario.get("start", {})
    robot = RobotSim(
        world,
        start_x=start.get("x_mm", 1000),
        start_y=start.get("y_mm", 1000),
        start_hdg_deg=start.get("hdg_deg", 0),
    )
    # Optional battery override from scenario (e.g. low_battery.yaml)
    if "battery_mv_override" in scenario:
        robot.battery_mv = float(scenario["battery_mv_override"])

    client = SITLClient(
        robot,
        args.host,
        args.port,
        sensor_hz=args.hz,
        cam_hz=args.cam_hz,
        duration=args.duration,
        state_file=args.state_file,
    )
    asyncio.run(client.connect_and_run())


if __name__ == "__main__":
    main()
