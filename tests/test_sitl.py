# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for SITL physics and protocol integration (no network needed)."""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import protocol
from protocol import ActuatorCmd, ACT_DIFF_DRIVE, FLAG_EMERGENCY
from tools.sitl.sitl_wheeled import RobotSim, World, Rect

# ── World / raycast ────────────────────────────────────────────────────────────


def test_world_raycast_wall():
    world = World(obstacles=[], width_mm=5000, height_mm=5000)
    # Robot at center facing East — should hit east wall at ~2500 mm
    robot = RobotSim(world, start_x=2500, start_y=2500, start_hdg_deg=0)
    d = world.raycast(2500, 2500, 0)
    assert 2400 < d < 2700  # wall at x=5000 + 100 thick, from x=2500 → ~2500


def test_world_collision():
    obs = [Rect(1000, 1000, 500, 500)]
    world = World(obstacles=obs)
    assert world.collides(1200, 1200)
    assert not world.collides(200, 200)


def test_world_obstacle_raycast():
    obs = [Rect(1000, 500, 100, 1000)]  # vertical wall
    world = World(obstacles=obs, width_mm=5000, height_mm=5000)
    # From x=200, facing East (0°) — obstacle at x=1000
    d = world.raycast(200, 1000, 0)
    assert 750 < d < 850  # ~800 mm to x=1000


# ── RobotSim physics ───────────────────────────────────────────────────────────


def _make_robot(x=1000, y=1000, hdg=0) -> RobotSim:
    world = World(obstacles=[], width_mm=10000, height_mm=10000)
    return RobotSim(world, start_x=x, start_y=y, start_hdg_deg=hdg)


def test_robot_stop_by_default():
    robot = _make_robot()
    assert robot.speed_l == 0
    assert robot.speed_r == 0


def test_robot_apply_cmd_forward():
    robot = _make_robot()
    cmd = ActuatorCmd.wheeled(60, 60)
    robot.apply_cmd(cmd)
    assert robot.speed_l == 60
    assert robot.speed_r == 60


def test_robot_apply_cmd_emergency_stops():
    robot = _make_robot()
    cmd = ActuatorCmd.wheeled(60, 60)
    robot.apply_cmd(cmd)
    stop = ActuatorCmd.stop(n_channels=2)
    robot.apply_cmd(stop)
    assert robot.speed_l == 0
    assert robot.speed_r == 0


def test_robot_apply_cmd_clamps():
    robot = _make_robot()
    cmd = ActuatorCmd.wheeled(200, -200)
    robot.apply_cmd(cmd)
    assert robot.speed_l == 100
    assert robot.speed_r == -100


def test_robot_moves_forward():
    import time

    robot = _make_robot(x=5000, y=5000, hdg=0)  # facing East
    cmd = ActuatorCmd.wheeled(100, 100)
    robot.apply_cmd(cmd)

    # Step 1 second of physics at 100 Hz
    for _ in range(100):
        robot._last_tick -= 0.01  # fake 10ms elapsed
        robot.step()

    assert robot.x > 5200  # should have moved East


def test_robot_turns_right():
    import time

    robot = _make_robot(x=5000, y=5000, hdg=0)
    # Right turn: left faster than right → CW
    cmd = ActuatorCmd.wheeled(60, -60)
    robot.apply_cmd(cmd)

    for _ in range(50):
        robot._last_tick -= 0.01
        robot.step()

    assert robot.hdg_deg > 10  # heading increased (CW)


def test_robot_sensor_packet():
    robot = _make_robot()
    pkt = robot.sensor_packet()
    assert pkt.battery_mv > 0
    assert pkt.range_front_mm > 0
    from protocol import SensorPacket

    assert isinstance(pkt, SensorPacket)


def test_robot_sensor_packet_roundtrip():
    robot = _make_robot()
    pkt = robot.sensor_packet()
    data = pkt.to_bytes()
    from protocol import SensorPacket

    pkt2 = SensorPacket.from_bytes(data)
    assert pkt2.battery_mv == pkt.battery_mv
    assert pkt2.range_front_mm == pkt.range_front_mm


def test_robot_status_packet():
    robot = _make_robot()
    st = robot.status_packet()
    from protocol import ROBOT_WHEELED

    assert st.robot_type == ROBOT_WHEELED


def test_robot_collision_stops_motion():
    # Place robot right next to a wall
    obs = [Rect(1200, 0, 100, 5000)]  # vertical wall at x=1200
    world = World(obstacles=obs, width_mm=10000, height_mm=10000)
    robot = RobotSim(world, start_x=1000, start_y=2500, start_hdg_deg=0)
    cmd = ActuatorCmd.wheeled(100, 100)
    robot.apply_cmd(cmd)

    for _ in range(200):
        robot._last_tick -= 0.01
        robot.step()

    # Robot should not have passed through the wall
    assert robot.x < 1200


def test_robot_battery_drains():
    robot = _make_robot()
    initial_batt = robot.battery_mv
    cmd = ActuatorCmd.wheeled(100, 100)
    robot.apply_cmd(cmd)

    for _ in range(200):
        robot._last_tick -= 0.01
        robot.step()

    assert robot.battery_mv < initial_batt


# ── Scenario loading ───────────────────────────────────────────────────────────


def test_world_from_scenario_empty():
    scenario = {"obstacles": [], "width_mm": 8000, "height_mm": 6000}
    world = World.from_scenario(scenario)
    assert world.width == 8000
    assert world.height == 6000
    assert len(world.obstacles) == 0


def test_world_from_scenario_with_obstacles():
    scenario = {
        "obstacles": [{"x": 1000, "y": 1000, "w": 500, "h": 500}],
        "width_mm": 5000,
        "height_mm": 5000,
    }
    world = World.from_scenario(scenario)
    assert len(world.obstacles) == 1
    assert world.collides(1200, 1200)


# ── B01 standalone SITLRobot (sitl.py) ─────────────────────────────────────────

from sitl import (
    SITLRobot,
    SITLNetClient,
    synthetic_camera_payload,
    run_scenario,
    list_scenarios,
    SCENARIOS,
    DRONE_HOVER_PWM,
    DRONE_PWM_NEUTRAL,
    DRONE_PWM_MAX,
    SCEN_MEDIUM_S,
)
from protocol import (
    ROBOT_WHEELED,
    ROBOT_DRONE,
    ROBOT_HUMANOID,
    ROBOT_ACKERMANN,
    SensorPacketDrone,
    SensorPacketHumanoid,
    ACT_QUAD_ROTOR,
    ACT_ACKERMANN,
    STATUS,
    CAMERA_FRAME,
    SENSOR_PACKET,
    build_packet,
    parse_packet,
    read_packet,
)

# --- Construction / typing ---------------------------------------------------


def test_sitl_default_is_wheeled():
    r = SITLRobot()
    assert r.robot_type == ROBOT_WHEELED
    assert r.battery_mv > 0


def test_sitl_humanoid_default_joints():
    r = SITLRobot(robot_type=ROBOT_HUMANOID)
    assert len(r.humanoid_joints_cdeg) == 12
    assert all(j == 0 for j in r.humanoid_joints_cdeg)


def test_sitl_drone_starts_on_ground():
    r = SITLRobot(robot_type=ROBOT_DRONE)
    assert r.z_mm == 0.0
    assert r.vz_m_s == 0.0


# --- Physics: wheeled --------------------------------------------------------


def test_sitl_wheeled_moves_forward():
    r = SITLRobot(robot_type=ROBOT_WHEELED, x_mm=5000, y_mm=5000, hdg_deg=0)
    r.apply_cmd(ActuatorCmd.wheeled(100, 100))
    for _ in range(100):
        r.step(0.01)
    assert r.x_mm > 5100  # moved east
    assert abs(r.y_mm - 5000) < 5  # barely any Y drift


def test_sitl_wheeled_turns_right_on_skid():
    r = SITLRobot(robot_type=ROBOT_WHEELED, x_mm=5000, y_mm=5000, hdg_deg=0)
    r.apply_cmd(ActuatorCmd.wheeled(60, -60))
    for _ in range(100):
        r.step(0.01)
    assert r.hdg_deg > 10


def test_sitl_wheeled_encoders_advance():
    r = SITLRobot(robot_type=ROBOT_WHEELED)
    r.apply_cmd(ActuatorCmd.wheeled(100, 100))
    for _ in range(100):
        r.step(0.01)
    assert r.enc_l > 0
    assert r.enc_r > 0


# --- Physics: ackermann ------------------------------------------------------


def test_sitl_ackermann_straight():
    r = SITLRobot(robot_type=ROBOT_ACKERMANN, x_mm=1000, y_mm=5000, hdg_deg=0)
    r.apply_cmd(ActuatorCmd(actuator_type=ACT_ACKERMANN, channels=[100, 0]))
    for _ in range(100):
        r.step(0.01)
    assert r.x_mm > 1100
    assert abs(r.hdg_deg) < 1


def test_sitl_ackermann_steer_turns():
    r = SITLRobot(robot_type=ROBOT_ACKERMANN, x_mm=1000, y_mm=5000, hdg_deg=0)
    # Full right steer
    r.apply_cmd(ActuatorCmd(actuator_type=ACT_ACKERMANN, channels=[100, 3500]))
    for _ in range(200):
        r.step(0.01)
    assert r.hdg_deg > 5


# --- Physics: drone ----------------------------------------------------------


def test_sitl_drone_climbs_at_full_throttle():
    r = SITLRobot(robot_type=ROBOT_DRONE)
    r.apply_cmd(
        ActuatorCmd.drone(DRONE_PWM_MAX, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL)
    )
    for _ in range(100):
        r.step(0.01)
    assert r.z_mm > 100  # climbed at least 10 cm


def test_sitl_drone_hover_altitude_stable():
    r = SITLRobot(robot_type=ROBOT_DRONE, z_mm=1000, vz_m_s=0.0)
    r.apply_cmd(
        ActuatorCmd.drone(DRONE_HOVER_PWM, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL)
    )
    for _ in range(100):
        r.step(0.01)
    # At hover PWM (= hover constant), net vertical accel = 0 → altitude stays
    assert abs(r.z_mm - 1000) < 5


def test_sitl_drone_ground_clamp():
    r = SITLRobot(robot_type=ROBOT_DRONE)
    # Leave throttle at min; z must never go below 0
    for _ in range(100):
        r.step(0.01)
    assert r.z_mm == 0.0


# --- Emergency / flags -------------------------------------------------------


def test_sitl_emergency_stops_motion():
    r = SITLRobot(robot_type=ROBOT_WHEELED)
    r.apply_cmd(ActuatorCmd.wheeled(100, 100))
    assert r.speed_l_pct == 100
    r.apply_cmd(ActuatorCmd.stop())
    assert r.speed_l_pct == 0
    assert r.speed_r_pct == 0


def test_sitl_drone_emergency_disarms():
    r = SITLRobot(robot_type=ROBOT_DRONE)
    r.apply_cmd(
        ActuatorCmd.drone(DRONE_PWM_MAX, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL, DRONE_PWM_NEUTRAL)
    )
    r.apply_cmd(ActuatorCmd.stop(ACT_QUAD_ROTOR, n_channels=4))
    assert r.drone_throttle_pwm == 1000
    assert r.vz_m_s == 0.0


# --- Sensor packets ----------------------------------------------------------


def test_sitl_wheeled_sensor_packet_roundtrip():
    r = SITLRobot(robot_type=ROBOT_WHEELED)
    r.step(0.1)
    pkt = r.sensor_packet()
    data = pkt.to_bytes()
    from protocol import SensorPacket

    pkt2 = SensorPacket.from_bytes(data)
    assert pkt2.battery_mv == pkt.battery_mv
    assert pkt2.encoder_l == pkt.encoder_l


def test_sitl_drone_sensor_packet_type():
    r = SITLRobot(robot_type=ROBOT_DRONE, z_mm=500)
    pkt = r.sensor_packet()
    assert isinstance(pkt, SensorPacketDrone)
    data = pkt.to_bytes()
    pkt2 = SensorPacketDrone.from_bytes(data)
    assert pkt2.gps_lat_deg7 == pkt.gps_lat_deg7
    assert pkt2.sonar_down_mm == pkt.sonar_down_mm
    assert pkt2.baro_pa == pkt.baro_pa


def test_sitl_humanoid_sensor_packet_type():
    r = SITLRobot(robot_type=ROBOT_HUMANOID)
    pkt = r.sensor_packet()
    assert isinstance(pkt, SensorPacketHumanoid)
    data = pkt.to_bytes()
    pkt2 = SensorPacketHumanoid.from_bytes(data)
    assert pkt2.joint_angles == pkt.joint_angles


def test_sitl_status_packet_has_correct_type():
    for rt in (ROBOT_WHEELED, ROBOT_DRONE, ROBOT_HUMANOID, ROBOT_ACKERMANN):
        r = SITLRobot(robot_type=rt)
        st = r.status_packet()
        assert st.robot_type == rt


def test_sitl_battery_drains():
    r = SITLRobot(robot_type=ROBOT_WHEELED)
    batt0 = r.battery_mv
    r.apply_cmd(ActuatorCmd.wheeled(100, 100))
    for _ in range(200):
        r.step(0.01)
    assert r.battery_mv < batt0


# --- Camera ------------------------------------------------------------------


def test_sitl_camera_payload_has_header():
    payload = synthetic_camera_payload()
    # Header: 2B width + 2B height + 1B format
    import struct as _s

    w, h, fmt = _s.unpack_from("<HHB", payload, 0)
    assert w == 160
    assert h == 120
    assert fmt in (1, 2)


def test_sitl_camera_packet_builds():
    r = SITLRobot()
    packet = build_packet(CAMERA_FRAME, r.camera_payload())
    parsed = parse_packet(packet)
    assert parsed is not None
    pkt_type, payload = parsed
    assert pkt_type == CAMERA_FRAME
    assert len(payload) > 5  # header + some pixels


# --- Scenarios ---------------------------------------------------------------


def test_list_scenarios_nonempty():
    names = list_scenarios()
    assert "forward_10m" in names
    assert "emergency_stop" in names
    assert "idle" in names


def test_run_scenario_forward_wheeled():
    r = SITLRobot(robot_type=ROBOT_WHEELED, x_mm=1000, y_mm=5000, hdg_deg=0)
    run_scenario(r, "forward_10m", duration_s=SCEN_MEDIUM_S, dt=0.01)
    assert r.x_mm > 1000


def test_run_scenario_emergency_stop():
    r = SITLRobot(robot_type=ROBOT_WHEELED, x_mm=1000, y_mm=5000, hdg_deg=0)
    run_scenario(r, "emergency_stop", duration_s=SCEN_MEDIUM_S, dt=0.01)
    # After emergency: both wheels zero
    assert r.speed_l_pct == 0
    assert r.speed_r_pct == 0


def test_run_scenario_takeoff_drone():
    r = SITLRobot(robot_type=ROBOT_DRONE)
    run_scenario(r, "takeoff_hover_land", duration_s=1.0, dt=0.01)
    assert r.z_mm > 0.0  # climbed above ground


def test_run_scenario_unknown_raises():
    r = SITLRobot()
    import pytest

    with pytest.raises(KeyError):
        run_scenario(r, "does_not_exist")


# --- Network client (integration w/ asyncio echo) ----------------------------


def test_sitl_netclient_sends_status_and_sensors(monkeypatch):
    """The SITL client completes the brain's real Noise handshake (the brain's
    `perform_handshake` is the server) and its packets arrive opened.

    Canary: drop the client's handshake and the brain's `read_packet` refuses
    the unlinked stream, so no packet is received."""
    import asyncio as _a
    from tests.mock_kernel import arm_brain, reset_protocol_state

    reset_protocol_state()
    arm_brain(monkeypatch)
    received: list[tuple[int, bytes]] = []

    async def server_handler(reader, writer):
        try:
            if not await protocol.perform_handshake(reader, writer):
                writer.close()
                return
            for _ in range(3):
                res = await read_packet(reader)
                if res is None:
                    continue
                received.append(res)
        except _a.IncompleteReadError:
            pass
        writer.close()
        await writer.wait_closed()

    async def main():
        server = await _a.start_server(server_handler, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        robot = SITLRobot(robot_type=ROBOT_WHEELED)
        client = SITLNetClient(
            robot, host=host, port=port, sensor_hz=50, camera_hz=0, duration_s=0.3
        )
        await client.run()
        server.close()
        await server.wait_closed()

    try:
        _a.run(main())
    finally:
        reset_protocol_state()
    types = [r[0] for r in received]
    assert STATUS in types
    assert SENSOR_PACKET in types


def test_sitl_netclient_refuses_to_dial_without_the_link_key(monkeypatch):
    """No key, no link: the client never opens a plaintext connection."""
    import asyncio as _a

    monkeypatch.delenv("AZOS_BRAIN_LINK_KEY", raising=False)
    client = SITLNetClient(SITLRobot(robot_type=ROBOT_WHEELED), host="127.0.0.1", port=1, duration_s=0.1)
    with pytest.raises(RuntimeError, match="AZOS_BRAIN_LINK_KEY"):
        _a.run(client.run())


if __name__ == "__main__":
    test_world_raycast_wall()
    test_world_collision()
    test_world_obstacle_raycast()
    test_robot_stop_by_default()
    test_robot_apply_cmd_forward()
    test_robot_apply_cmd_emergency_stops()
    test_robot_apply_cmd_clamps()
    test_robot_moves_forward()
    test_robot_turns_right()
    test_robot_sensor_packet()
    test_robot_sensor_packet_roundtrip()
    test_robot_status_packet()
    test_robot_collision_stops_motion()
    test_robot_battery_drains()
    test_world_from_scenario_empty()
    test_world_from_scenario_with_obstacles()
    test_sitl_default_is_wheeled()
    test_sitl_wheeled_moves_forward()
    test_sitl_drone_climbs_at_full_throttle()
    test_sitl_drone_ground_clamp()
    test_sitl_camera_payload_has_header()
    test_list_scenarios_nonempty()
    test_run_scenario_forward_wheeled()
    test_run_scenario_takeoff_drone()
    print("All SITL tests passed!")
