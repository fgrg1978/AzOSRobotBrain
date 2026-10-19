# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Integration test: mock brain server + SITL physics (no LLM, no hardware).

Tests the full packet pipeline:
  RobotSim → TCP → MockBrainServer → ActuatorCmd → RobotSim.apply_cmd
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import protocol
from protocol import (
    build_packet,
    parse_packet,
    SensorPacket,
    StatusPacket,
    ActuatorCmd,
    SENSOR_PACKET,
    STATUS,
    ACTUATOR_CMD,
    ROBOT_WHEELED,
    FLAG_EMERGENCY,
)
from tools.sitl.sitl_wheeled import RobotSim, World

# ── Mock brain: just echoes a fixed ActuatorCmd on every SensorPacket ─────────


async def _mock_brain(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    responses: list[ActuatorCmd],
    received_packets: list,
):
    """Fake server that records sensor packets and sends canned responses."""
    try:
        while True:
            result = await asyncio.wait_for(protocol.read_packet(reader), timeout=2.0)
            if result is None:
                break
            pkt_type, payload = result
            received_packets.append(pkt_type)

            if pkt_type == SENSOR_PACKET and responses:
                cmd = responses.pop(0)
                await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())

            if not responses and pkt_type == SENSOR_PACKET:
                break  # done — close connection
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_sensor_packet_sent_to_server():
    """Robot sends SensorPackets; server receives them."""

    received = []
    responses = [ActuatorCmd.stop(n_channels=2)]  # one response then done

    async def run():
        server = await asyncio.start_server(
            lambda r, w: _mock_brain(r, w, responses, received), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]

        world = World(obstacles=[])
        robot = RobotSim(world, 1000, 1000, 0)

        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Send STATUS first
            st = robot.status_packet()
            writer.write(build_packet(STATUS, st.to_bytes()))
            await writer.drain()

            # Send one SensorPacket
            sp = robot.sensor_packet()
            writer.write(build_packet(SENSOR_PACKET, sp.to_bytes()))
            await writer.drain()

            # Read back the ActuatorCmd
            result = await asyncio.wait_for(protocol.read_packet(reader), timeout=2.0)
            writer.close()

        return result, received

    result, received = asyncio.run(run())
    assert result is not None
    pkt_type, payload = result
    assert pkt_type == ACTUATOR_CMD
    assert STATUS in received
    assert SENSOR_PACKET in received


def test_actuator_cmd_applied_to_robot():
    """ActuatorCmd received from server is applied to robot physics."""

    cmd_to_send = ActuatorCmd.wheeled(60, 60)
    received_cmd = []
    responses = [cmd_to_send]

    async def run():
        server = await asyncio.start_server(
            lambda r, w: _mock_brain(r, w, responses, []), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        world = World(obstacles=[])
        robot = RobotSim(world, 1000, 1000, 0)

        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build_packet(SENSOR_PACKET, robot.sensor_packet().to_bytes()))
            await writer.drain()

            result = await asyncio.wait_for(protocol.read_packet(reader), timeout=2.0)
            if result:
                _, payload = result
                cmd = ActuatorCmd.from_bytes(payload)
                robot.apply_cmd(cmd)
                received_cmd.append(cmd)
            writer.close()

    asyncio.run(run())
    assert received_cmd
    assert received_cmd[0].channels == [60, 60]


def test_emergency_stop_cmd_from_server():
    """Emergency stop from server halts robot speeds."""

    responses = [ActuatorCmd.stop(n_channels=2)]

    async def run():
        server = await asyncio.start_server(
            lambda r, w: _mock_brain(r, w, responses, []), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        world = World(obstacles=[])
        robot = RobotSim(world, 1000, 1000, 0)
        robot.speed_l = 80
        robot.speed_r = 80

        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build_packet(SENSOR_PACKET, robot.sensor_packet().to_bytes()))
            await writer.drain()

            result = await asyncio.wait_for(protocol.read_packet(reader), timeout=2.0)
            if result:
                _, payload = result
                cmd = ActuatorCmd.from_bytes(payload)
                robot.apply_cmd(cmd)
            writer.close()

        return robot

    robot = asyncio.run(run())
    assert robot.speed_l == 0
    assert robot.speed_r == 0


def test_status_packet_roundtrip():
    """StatusPacket serialization survives the wire."""
    world = World(obstacles=[])
    robot = RobotSim(world, 0, 0, 0)
    st = robot.status_packet()
    data = st.to_bytes()
    st2 = StatusPacket.from_bytes(data)
    assert st2.robot_type == ROBOT_WHEELED
    assert st2.tasks_ok == st.tasks_ok


def test_sensor_packet_values_preserved():
    """SensorPacket values are preserved through serialization."""
    world = World(obstacles=[], width_mm=5000, height_mm=5000)
    robot = RobotSim(world, 2500, 2500, 0)
    # Set known values
    robot.battery_mv = 7200.0
    robot.enc_l = 12345
    robot.enc_r = 12300
    robot.odom_dist_mm = 1000

    sp = robot.sensor_packet()
    data = sp.to_bytes()
    sp2 = SensorPacket.from_bytes(data)

    assert sp2.battery_mv == 7200
    assert sp2.encoder_l == 12345
    assert sp2.encoder_r == 12300
    assert sp2.odom_dist_mm == 1000


def test_multiple_sensor_packets():
    """Server handles multiple consecutive SensorPackets."""
    packet_count = [0]
    received = []

    async def counting_brain(reader, writer):
        try:
            for _ in range(3):
                result = await asyncio.wait_for(protocol.read_packet(reader), timeout=1.0)
                if result:
                    received.append(result[0])
        except asyncio.TimeoutError:
            pass
        finally:
            writer.close()

    async def run():
        server = await asyncio.start_server(counting_brain, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        world = World(obstacles=[])
        robot = RobotSim(world, 1000, 1000, 0)

        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            for _ in range(3):
                writer.write(build_packet(SENSOR_PACKET, robot.sensor_packet().to_bytes()))
            await writer.drain()
            await asyncio.sleep(0.2)
            writer.close()

    asyncio.run(run())
    assert received.count(SENSOR_PACKET) == 3


if __name__ == "__main__":
    test_sensor_packet_sent_to_server()
    test_actuator_cmd_applied_to_robot()
    test_emergency_stop_cmd_from_server()
    test_status_packet_roundtrip()
    test_sensor_packet_values_preserved()
    test_multiple_sensor_packets()
    print("All integration tests passed!")
