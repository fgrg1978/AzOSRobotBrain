# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""AzOS Robot Brain Server — receives sensor data from robot, sends motor commands.

Architecture:
  Robot --TCP--> server.py --API--> LM Studio (VLM + LLM)
                              |
                              +--> ModeManager ──> TaskPlanner ──> SkillRunner
                              |                                        |
                              +--> policy ─────────────────────> ActuatorCmd
                              |
                              +--> Notifier (Pushover/Telegram/Email/Webhook)
                              +--> TelegramBot (bidirectional commands)
                              +--> APIServer   (HTTP REST)
"""

import asyncio
import logging
import os
import struct
import time
import yaml

import metrics as _metrics_mod
import protocol
from protocol import (
    SensorPacket,
    sensor_packet_from_bytes,
    ActuatorCmd,
    VelocityCmd,
    StatusPacket,
    SENSOR_PACKET,
    CAMERA_FRAME,
    STATUS,
    ACTUATOR_CMD,
    ROBOT_WHEELED,
    ROBOT_DRONE,
    ROBOT_HUMANOID,
    ROBOT_TYPE_BY_NAME,
    ROBOT_TYPE_NAME_BY_ID,
    ROBOT_TYPE_DEFAULT_NAME,
)
from perception.vision import VisionPerception, VLM_TIMEOUT_S
from planner.decide import Planner, LLM_TIMEOUT_S
from planner.modes import ModeManager
from planner.task_planner import TaskPlanner
from planner.skills import get_skills
from policy import get_translator
from policy.safety import SafetyProfile
from executor.skill_runner import SkillRunner
from perception.slam import SLAM, OccupancyGrid
from planner.path import PathPlanner
from planner.mapper import PerimeterMapper
from planner.patrol import PatrolController
from planner.led import LedController
from planner.power import PowerManager
from planner.sensors import SensorFusion
from planner.alert import AlertPipeline
from planner.deterrent import DeterrentManager, DeterrentLevel
from perception.rtsp_monitor import RtspMonitor, RtspEvent, cameras_from_config
from planner.docking import DockManager, DockState, DockInfo, dock_from_config
from planner.zones import ZoneManager
from planner.tracker import IntrusionTracker
from planner.experience import ExperienceStore
from planner.meta import MetaReviewer
from planner.robot_description import RobotDescription
from planner.transforms import TransformTree
from planner.battery import BatteryMonitor
from planner.gps_mission import GpsMission, Geofence, GpsPosition
from planner.logger import MissionLogger
from planner.payload import PayloadManager
from planner.fleet import FleetPlanner
from planner.offline import OfflineManager
from fleet import FleetManager
from notifications import Notifier
from telegram_bot import TelegramBot
from api import APIServer
from camera_link import CameraListener, peer_host

logger = logging.getLogger("brain.server")


class RobotState:
    """Tracks the latest sensor data from the robot."""

    def __init__(self):
        self.sensors: dict = {}
        self.odom: dict = {"dist_mm": 0, "heading_cdeg": 0}
        self.last_image: bytes = b""
        self.last_sensor_time: float = 0
        self.last_image_time: float = 0
        self.status: dict = {}
        self.connected: bool = False

    def reset(self):
        """Clear all sensor/image state (but not `connected`, which the
        caller manages). B-C5: must be called on every new connection —
        otherwise a reconnecting robot's first cycle decides against stale
        pre-disconnect sensors/odometry, and SLAM computes its odom delta
        against a pre-reboot baseline the kernel has already forgotten.
        """
        self.sensors = {}
        self.odom = {"dist_mm": 0, "heading_cdeg": 0}
        self.last_image = b""
        self.last_sensor_time = 0
        self.last_image_time = 0
        self.status = {}

    def update_sensors(self, pkt):
        self.sensors = {
            "battery_mv": pkt.battery_mv,
        }
        if hasattr(pkt, "range_front_mm"):
            self.sensors["range_front_mm"] = pkt.range_front_mm
            self.sensors["range_right_mm"] = pkt.range_right_mm
        if hasattr(pkt, "accel_mg"):
            self.sensors["accel_mg"] = pkt.accel_mg
            self.sensors["gyro_mdps"] = pkt.gyro_mdps
        if hasattr(pkt, "odom_dist_mm"):
            self.odom = {
                "dist_mm": pkt.odom_dist_mm,
                "heading_cdeg": pkt.odom_hdg_cdeg,
            }
        self.last_sensor_time = time.time()

    def update_image(self, image_data: bytes):
        self.last_image = image_data
        self.last_image_time = time.time()

    def update_status(self, pkt: StatusPacket):
        self.status = {
            "mode": pkt.mode,
            "tasks_ok": pkt.tasks_ok,
            "canary_ok": pkt.canary_ok,
            "uptime_s": pkt.uptime_s,
        }


# RFC-0036: consecutive perception failures before the brain declares itself
# "blind" and arms degraded mode. >1 so a single dropped frame / transient VLM
# hiccup does not trigger containment.
DEGRADE_PERCEPTION_FAIL_THRESHOLD: int = 3

# Hard ceiling on a single _navigate_to() call from _task_worker (RETURN_TO_DOCK
# / INVESTIGATE_ZONE). Without this, a stuck path-follow (e.g. SLAM giving no
# progress, or the robot wedged against an obstacle) blocks the task queue —
# and every subsequent queued task — forever. This is a backstop, not the
# primary navigation timing (PatrolController has its own per-waypoint-reach
# logic); it's set generously above any real navigation leg.
NAVIGATION_TIMEOUT_S: float = 120.0

# B-A15: deadline for reading one frame off the robot link. `read_packet` has
# no internal timeout and `run()` sets none on the socket, so a peer that opens
# TCP/9000 and then sends nothing parks in `readexactly()` forever — and since
# handle_robot() refuses a second connection while `self._writer` is set (B-A1),
# that single idle socket locks the real robot out until the process restarts.
#
# This is deliberately NOT `safety_profile.comms_timeout_s`: that knob is a
# *staleness* threshold ("how old may the last sensor packet be before we treat
# the robot as out of contact", 3.0s on a drone) and is checked independently
# in _safety_check. Reusing it here would drop links over ordinary jitter well
# before the safety layer has anything to say. This is the far looser "this
# peer is not talking at all" backstop.
#
# NOTE: read_packet is not an atomic read (header then body), so a timeout can
# land mid-frame with bytes already consumed. The stream is unrecoverable at
# that point — _packet_stream closes the connection rather than resyncing.
LINK_READ_TIMEOUT_S: float = 30.0

# ── Actuation-link bind address ──────────────────────────────────────────────

#: The robot link listens on every interface: real robots are remote, and the
#: Noise handshake (keyed by `AZOS_BRAIN_LINK_KEY`) is what admits a peer.
LINK_BIND_ANY: str = "0.0.0.0"


class BrainServer:
    """Main server — orchestrates perception, planning, execution, and notifications."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        lm = self.config["lmstudio"]
        robot_type_str = self.config["robot"].get("type", "wheeled")

        # Perception + reactive planner (VLM/LLM)
        self.vision = VisionPerception(lm["host"], lm["port"], lm["vlm_model"])
        self.planner = Planner(lm["host"], lm["port"], lm["llm_model"])

        # RFC-0036: brain-triggered degraded mode. When perception fails for
        # several cycles in a row the brain has effectively gone blind — a
        # situational hazard the kernel cannot perceive — so it arms degraded
        # mode (capability containment) and clears it on recovery.
        self._perception_fail_streak = 0
        self._degraded_sent = False

        # Experience store (persistent plan outcome memory)
        exp_dir = self.config.get("experience", {}).get("dir", "data/experience")
        self.experience = ExperienceStore(exp_dir, robot_type=robot_type_str)

        # Meta-reviewer (LLM heuristic extraction)
        self.meta_reviewer = MetaReviewer(
            lm["host"],
            lm["port"],
            lm["llm_model"],
            experience=self.experience,
            robot_type=robot_type_str,
        )

        # Task planner (LLM → skill plan) — with experience + meta hooks
        self.task_planner = TaskPlanner(
            lm["host"],
            lm["port"],
            lm["llm_model"],
            robot_type=robot_type_str,
            experience=self.experience,
            meta=self.meta_reviewer,
        )

        # Policy translator.
        #
        # B-A14: robot_type is the operator's declaration, taken from config
        # and never from the wire. It used to be hardwired to ROBOT_WHEELED
        # here and then overwritten by whatever robot_type the first STATUS
        # packet claimed (_on_status). Two bugs fell out of that:
        #   1. With `robot: type: drone` in config the server booted with a
        #      DRONE policy but a WHEELED safety profile (built from
        #      self.robot_type at :268) until a STATUS happened to arrive.
        #   2. Any peer that reached TCP/9000 could rewrite it. handle_robot()
        #      resets self.state and self._prev_odom per connection (B-C5) but
        #      not self.robot_type, so one 8-byte STATUS from an attacker
        #      persisted across the disconnect and mis-parsed / mis-safety-
        #      checked the real robot's next session.
        # Config is now the single source of truth; _on_status validates
        # against it instead of assigning from it.
        self._configured_robot_type = ROBOT_TYPE_BY_NAME.get(
            str(robot_type_str).lower(), ROBOT_WHEELED
        )
        self.robot_type = self._configured_robot_type
        self.policy = get_translator(robot_type_str, self.config)

        # Mode manager
        self.mode_manager = ModeManager(self.config)

        # Executor
        self.runner: SkillRunner | None = None

        # Notifications
        self.notifier = Notifier(self.config.get("notifications", {}))

        # SLAM + navigation
        slam_cfg = self.config.get("slam", {})
        grid = OccupancyGrid(
            resolution_mm=slam_cfg.get("map_resolution_mm", 50),
            size_cells=slam_cfg.get("map_size_cells", 1000),
        )
        self.slam = SLAM(grid)
        self.path_planner = PathPlanner(grid)
        self.mapper = PerimeterMapper(self.slam, self.path_planner)
        self.patrol: PatrolController | None = None  # created on connection

        # LED controller
        self.led = LedController(protocol.send_packet)

        # Power mode manager (ECO/ALERT)
        self.power = PowerManager(protocol.send_packet)

        # Multi-sensor fusion (PIR/sound/IR)
        self.sensor_fusion = SensorFusion()

        # Alert pipeline (buzzer, evidence, notifications)
        alert_cfg = self.config.get("alert", {})
        self.alert_pipeline = AlertPipeline(
            send_packet=protocol.send_packet,
            notifier=self.notifier,
            evidence_dir=alert_cfg.get("evidence_dir", "data/evidence"),
            cooldown_s=alert_cfg.get("cooldown_s", 30),
            evidence_frames=alert_cfg.get("evidence_frames", 10),
        )

        # Deterrent system (escalating response)
        self.deterrent = DeterrentManager(protocol.send_packet)

        # RTSP camera network (fixed surveillance cameras)
        rtsp_cameras = cameras_from_config(self.config)
        self.rtsp_monitor = (
            RtspMonitor(
                cameras=rtsp_cameras,
                vision=self.vision,
                on_threat=self._on_rtsp_threat,
                detect_labels=self.config.get("modes", {})
                .get("guardia", {})
                .get("detect", ["person", "vehicle", "fire"]),
            )
            if rtsp_cameras
            else None
        )

        # Zones of interest manager
        self.zone_manager = ZoneManager()

        # Intruder tracker
        self.tracker = IntrusionTracker()

        # Auto-docking manager
        dock_info = dock_from_config(self.config)
        self.dock_manager = DockManager(
            dock=dock_info,
            on_dock_needed=self._on_dock_needed,
            on_undock_ready=self._on_undock_ready,
            on_critical=self._on_battery_critical,
        )

        # Robot state
        self.state = RobotState()
        self.safety_profile = SafetyProfile.for_robot_type(self.robot_type, self.config)
        battery_mah = self.config.get("battery", {}).get("nominal_mah", 3600)
        self.battery_monitor = BatteryMonitor(nominal_mah=battery_mah)
        self._writer: asyncio.StreamWriter | None = None
        self._prev_odom: dict = {"dist_mm": 0, "heading_cdeg": 0}
        # C1: camera frames on their own connection (camera_link.py), paired
        # with the established control connection by source address.
        self._control_peer: str | None = None
        self.camera = CameraListener(
            lambda: self._control_peer, self._on_camera_frame, LINK_READ_TIMEOUT_S
        )

        # GPS mission + geofence (E03)
        self.gps_mission: GpsMission | None = None
        geofence_cfg = self.config.get("geofence", {})
        if geofence_cfg.get("enabled") and geofence_cfg.get("polygon"):
            self.geofence = Geofence(
                polygon=geofence_cfg["polygon"], margin_m=geofence_cfg.get("margin_m", 5.0)
            )
        else:
            self.geofence = None

        # Mission logger (E06)
        log_dir = self.config.get("logging", {}).get("dir", "data/missions")
        self.mission_logger = MissionLogger(log_dir=log_dir)

        # Payload manager (E04)
        payload_cfg = self.config.get("payload", {})
        self.payload_manager = PayloadManager(payload_cfg) if payload_cfg else None

        # Fleet planner (E07) — zone assignment / nearest-dispatch
        fleet_cfg = self.config.get("fleet", {})
        self.fleet_planner = FleetPlanner(fleet_cfg) if fleet_cfg.get("enabled") else None

        # Fleet manager (E07) — per-connection robot registry + command fanout
        self.fleet_manager = FleetManager(send_fn=protocol.send_packet)

        # Offline autonomy manager (E05)
        offline_cfg = self.config.get("offline", {})
        self.offline_manager = OfflineManager(offline_cfg)

        # Current task description (for experience recording)
        self._current_task_desc: str = ""

        # Task queue (for /task API and Telegram /task command)
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Telegram bot (enabled only if configured)
        tg_cfg = self.config.get("notifications", {}).get("telegram", {})
        self.tg_bot = TelegramBot(tg_cfg, self) if tg_cfg.get("enabled") else None

        # HTTP API
        api_cfg = self.config.get("api", {})
        self.api = (
            APIServer(self, port=api_cfg.get("port", 8080))
            if api_cfg.get("enabled", True)
            else None
        )

        # Robot description (declarative YAML hardware spec)
        robot_desc_path = os.path.join(os.path.dirname(__file__), "robot.yaml")
        if os.path.exists(robot_desc_path):
            self.robot_description = RobotDescription.from_yaml(robot_desc_path)
            logger.info(
                "[Brain] Loaded robot description: %s (%s)",
                self.robot_description.name,
                self.robot_description.type,
            )
        else:
            self.robot_description = None

        # Transform tree (sensor frame offsets from base_link)
        if self.robot_description:
            self.transforms = TransformTree.from_robot_description(self.robot_description)
            logger.info("[Brain] Transform tree: %d frames", len(self.transforms.list_frames()))
        else:
            self.transforms = TransformTree()  # default with just base_link

    # ── ActuatorCmd sender (used by SkillRunner) ──────────────────────────────

    async def _send_actuator_cmd(self, cmd: ActuatorCmd):
        if self._writer is None:
            logger.error(
                "[BRAIN] _send_actuator_cmd: no active connection — "
                "dropping ActuatorCmd (type=%d channels=%r)",
                cmd.actuator_type,
                cmd.channels,
            )
            return
        await protocol.send_packet(self._writer, ACTUATOR_CMD, cmd.to_bytes())

    async def _send_predict_cmd(self, pred):
        """RFC-0034: send the brain's predicted NEXT command (PKT_PREDICT) so the
        kernel can speculate ahead of the confirmed command. Best-effort — a
        dropped prediction just means no speculation that cycle."""
        if self._writer:
            await protocol.send_packet(self._writer, protocol.PREDICT_CMD, pred.to_bytes())

    async def _send_degrade_cmd(self, reason: int):
        """RFC-0036: arm (reason>0) or clear (reason=0) kernel degraded mode. The
        kernel contains the userspace blast radius at the capability chokepoint
        while its in-kernel control loop keeps safe-stopping."""
        if self._writer:
            await protocol.send_packet(
                self._writer, protocol.DEGRADE_CMD, protocol.DegradeCmd(reason=reason).to_bytes()
            )

    # ── Robot connection handler ──────────────────────────────────────────────

    async def handle_robot(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")

        # B-A1: self._writer/self.runner/self.patrol/self.state are plain
        # instance attributes shared by whatever connection is active — there
        # was never a concurrency-safe path for two robots to be connected to
        # the same BrainServer at once. A second connection silently stole
        # self._writer (the first robot's commands then went to the second
        # robot), and the first robot's eventual disconnect `finally` block
        # unconditionally tore down the second robot's runner/secure channel.
        # Multi-robot fleets run one BrainServer per robot (see
        # FleetManager/data-plane in fleet.py) — enforce that invariant
        # explicitly instead of silently corrupting shared state. No `await`
        # between this check and `self._writer = writer` below, so this is
        # race-free under asyncio's cooperative scheduling.
        if self._writer is not None:
            print(
                f"[BRAIN] Rejecting connection from {addr}: already serving a "
                f"robot (one robot per BrainServer instance — see fleet.py "
                f"for multi-robot deployment)"
            )
            writer.close()
            await writer.wait_closed()
            return

        print(f"[BRAIN] Robot connected from {addr}")
        self.state.connected = True
        self._writer = writer

        # B-C5: clear stale sensor/odom/image state from any previous
        # connection before this one starts. Without this, the first
        # perception cycle after a reconnect decides against pre-disconnect
        # sensor data, and _feed_slam computes its odometry delta against a
        # baseline the kernel (which may have rebooted) has already forgotten.
        self.state.reset()
        self._prev_odom = {"dist_mm": 0, "heading_cdeg": 0}

        # RFC-0019: if encryption is armed, complete the responder handshake
        # (the robot speaks first) BEFORE any packet is sent (the LED set_state below would otherwise
        # write plaintext). On failure, drop the connection — no fallback.
        if protocol.encrypt_link_armed():
            if not await protocol.perform_handshake(reader, writer):
                print(
                    f"[BRAIN] secure_channel: handshake failed for {addr}, " "dropping connection"
                )
                self.state.connected = False
                self._writer = None
                writer.close()
                return
        # Only now may a camera connection pair with this one.
        self._control_peer = peer_host(writer)

        # Register in fleet (E07). Use peer addr as provisional id; a later
        # STATUS packet may refine the robot_type.
        robot_id = f"{addr[0]}:{addr[1]}" if addr else "robot_unknown"
        self._current_robot_id = robot_id
        self.fleet_manager.register(
            robot_id=robot_id,
            robot_type=self.robot_type,
            name=self.config.get("robot", {}).get("name", robot_id),
            writer=writer,
        )

        # Set LED to monitoring (green) on connect
        await self.led.set_state("monitoring", writer)

        # Create SkillRunner bound to this connection (with experience callback)
        self.runner = SkillRunner(
            self.policy,
            self._send_actuator_cmd,
            on_plan_done=self._on_plan_done,
            send_predict=self._send_predict_cmd,
            is_comms_stale=lambda: (
                time.time() - self.state.last_sensor_time > self.safety_profile.comms_timeout_s
            ),
        )

        # Create PatrolController bound to this connection
        self.patrol = PatrolController(
            slam=self.slam,
            path_planner=self.path_planner,
            mapper=self.mapper,
            send_cmd=self._send_actuator_cmd,
            policy=self.policy,
            on_waypoint_reached=self._on_waypoint_reached,
            on_detection=self._on_detection,
            is_connected=lambda: self.state.connected,
        )

        self._packet_parse_errors = 0
        try:
            async for pkt_type, payload in self._packet_stream(reader):
                try:
                    await self._dispatch(pkt_type, payload, writer)
                except (ValueError, struct.error) as e:
                    # Malformed/truncated packet body (e.g. sensor_packet_
                    # from_bytes below MIN_SIZE). Drop just this packet and
                    # keep the connection — one bad frame shouldn't force a
                    # full robot reconnect + handshake.
                    self._packet_parse_errors += 1
                    logger.warning(
                        "[BRAIN] Packet parse error (type=0x%02x, total=%d): %s",
                        pkt_type,
                        self._packet_parse_errors,
                        e,
                    )
                    continue

        except asyncio.IncompleteReadError:
            print("[BRAIN] Robot disconnected")
        except Exception as e:
            print(f"[BRAIN] Error: {e}")
        finally:
            self.state.connected = False
            self._writer = None
            self._control_peer = None
            self.camera.close()
            # Drop this connection's encrypted channel so a reconnect runs a
            # fresh RFC-0019 handshake with new ephemeral keys (forward secrecy).
            protocol.drop_link(reader, writer)
            # Mark disconnected in the fleet registry (keeps record for history)
            current_id = getattr(self, "_current_robot_id", None)
            if current_id:
                self.fleet_manager.mark_disconnected(current_id)
            if self.runner:
                self.runner.interrupt("connection closed")
            writer.close()
            await writer.wait_closed()

    async def _packet_stream(self, reader: asyncio.StreamReader):
        """Async generator that yields (pkt_type, payload) packets."""
        while True:
            # B-A15: bound the read. Without a deadline a peer that connects
            # and then sends nothing blocks here forever, and because
            # handle_robot() serves one robot at a time that idle socket denies
            # service to the real robot indefinitely.
            try:
                result = await asyncio.wait_for(
                    protocol.read_packet(reader), timeout=LINK_READ_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                # Close, never resume: read_packet reads a header and then a
                # body, so the timeout may have fired between the two with
                # header bytes already consumed. Continuing the loop would
                # parse the next frame at a mid-frame offset. Log explicitly —
                # letting this propagate to handle_robot's `except Exception`
                # printed a bare, empty "[BRAIN] Error:" line.
                print(
                    f"[BRAIN] No packet within {LINK_READ_TIMEOUT_S:.0f}s — "
                    f"dropping idle connection"
                )
                break
            if result is None:
                print("[BRAIN] Invalid packet")
                break
            yield result

    async def _dispatch(self, pkt_type: int, payload: bytes, writer: asyncio.StreamWriter):
        """Route a received packet to the appropriate handler."""

        if pkt_type == SENSOR_PACKET:
            _metrics_mod.M.pkt_dispatch_handled_total.labels(type=f"0x{pkt_type:02x}").inc()
            pkt = sensor_packet_from_bytes(self.robot_type, payload)
            self.state.update_sensors(pkt)
            self._feed_slam(pkt)
            # Fleet heartbeat (E07): refresh last_seen + battery
            current_id = getattr(self, "_current_robot_id", None)
            if current_id:
                self.fleet_manager.heartbeat(
                    current_id,
                    battery_mv=getattr(pkt, "battery_mv", None),
                )
            # E05: Track connection health
            self.offline_manager.on_sensor_received(
                battery_pct=self.battery_monitor.state.capacity_pct
            )
            await self._safety_check(pkt, writer)
            # Check power mode timeout (ALERT → ECO after timeout)
            deescalated = await self.power.check_timeout(writer)
            if deescalated:
                await self.led.set_state("monitoring", writer)
            # Check deterrent safety timeout
            if self.deterrent.active:
                await self.deterrent.check_timeout(writer)
            # Log sensor event (E06)
            self.mission_logger.log_event(
                "sensor",
                {
                    "battery_mv": pkt.battery_mv,
                    "dist_mm": getattr(pkt, "odom_dist_mm", 0),
                    "heading_cdeg": getattr(pkt, "odom_hdg_cdeg", 0),
                },
            )

            # Geofence check (E03). GPS only exists on the drone SensorPacket
            # variant (gps_lat_deg7/gps_lon_deg7, degrees x1e7); wheeled/
            # humanoid packets don't carry GPS so hasattr() naturally skips
            # them here.
            if self.geofence and hasattr(pkt, "gps_lat_deg7") and pkt.gps_lat_deg7 != 0:
                pos = GpsPosition(lat=pkt.gps_lat_deg7 / 1e7, lon=pkt.gps_lon_deg7 / 1e7)
                if pos.has_fix and not self.geofence.contains(pos):
                    logger.warning("[GEOFENCE] Robot outside boundary!")
                    asyncio.create_task(
                        self.notifier.alert(
                            "Robot outside geofence boundary", title="Geofence Violation"
                        )
                    )
                    # Physical response, not just a notification: hard E-Stop
                    # with the geofence reason so the kernel's L0 safety
                    # layer can act even if userspace/the LLM loop is wedged.
                    await protocol.send_packet(
                        writer,
                        protocol.ESTOP_CMD,
                        protocol.EStopCmd(reason=protocol.ESTOP_REASON_GEOFENCE).to_bytes(),
                    )

            # Update detailed battery monitor (E09)
            self.battery_monitor.update(
                voltage_mv=pkt.battery_mv,
                current_ma=getattr(pkt, "current_ma", 0),
                mah_used=getattr(pkt, "mah_used", 0),
                capacity_pct=getattr(pkt, "capacity_pct", -1),
                sag_flag=getattr(pkt, "sag_flag", False),
                failsafe=getattr(pkt, "failsafe_level", 0),
            )
            # Battery monitoring → auto-dock
            new_dock_state = self.dock_manager.update_battery(pkt.battery_mv)
            if new_dock_state == DockState.LOW_BATTERY:
                await self._on_dock_needed()
            elif new_dock_state == DockState.CRITICAL:
                await self._on_battery_critical()
            elif new_dock_state == DockState.CHARGED:
                await self._on_undock_ready()
            # Process digital sensor triggers (PIR/sound/IR)
            if hasattr(pkt, "sensor_flags") and pkt.sensor_flags:
                await self._process_sensor_triggers(pkt.sensor_flags, writer)

        elif pkt_type == CAMERA_FRAME:
            await self._handle_camera_frame(payload, writer)

        elif pkt_type == STATUS:
            _metrics_mod.M.pkt_dispatch_handled_total.labels(type=f"0x{pkt_type:02x}").inc()
            pkt = StatusPacket.from_bytes(payload)
            self.state.update_status(pkt)
            # Fleet heartbeat (E07): STATUS is the canonical heartbeat packet
            current_id = getattr(self, "_current_robot_id", None)
            if current_id:
                self.fleet_manager.heartbeat(current_id)
            await self._on_status(pkt)

        else:
            # RFC-0039 step 1 ("PKT_CAPS reserved and undeclared types
            # recorded"): this used to be a bare if/elif chain with no final
            # else. OTA_ACK and SENSOR_COMPACT are declared packet-type
            # constants in protocol.py and listed as inbound in BRAIN.TOML,
            # but neither has ever had a branch here — they, and any
            # genuinely unknown type, fell through and vanished with no log
            # line and no counter, indistinguishable from a handled packet to
            # anyone watching from outside the process. This does not decode
            # or act on the payload (that is rollout steps 2-4, not this
            # slice) — it only makes the silence visible.
            logger.warning(
                "[BRAIN] Unhandled inbound packet type 0x%02x (%d byte payload) — "
                "no dispatch handler, packet dropped",
                pkt_type,
                len(payload),
            )
            _metrics_mod.M.pkt_dispatch_ignored_total.labels(type=f"0x{pkt_type:02x}").inc()

    # ── Patrol callbacks ─────────────────────────────────────────────────────

    async def _on_waypoint_reached(self, wp, index: int):
        """Called when patrol reaches a waypoint — trigger VLM scan."""
        logger.info(
            "[Patrol] Reached waypoint %d: (%d, %d) %s", index, wp.x_mm, wp.y_mm, wp.label or ""
        )

        # Skip scan if zone manager says not this pass (normal waypoints)
        if not self.zone_manager.should_scan(wp):
            return

        # LED: yellow while scanning
        await self.led.set_state("detecting", self._writer)

        # If we have a recent image, run VLM to check for detections
        if self.state.last_image:
            try:
                # Blocking OpenAI-client call (LM Studio) — must not stall
                # the event loop. Off-thread + hard timeout so a stuck/dead
                # VLM backend degrades to "skip this scan", not "wedge the
                # whole brain".
                scene = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vision.describe,
                        self.state.last_image,
                        context=f"patrol waypoint {index} ({wp.label})",
                    ),
                    timeout=VLM_TIMEOUT_S,
                )
                print(f"[VLM@wp{index}] {scene}")

                # Zone tagging (during mapping) and state change detection
                if not wp.zone_type:
                    self.zone_manager.tag_waypoint(wp, scene)
                else:
                    changed, change_desc = self.zone_manager.check_state(
                        wp,
                        scene,
                    )
                    if changed:
                        print(f"[ZONE CHANGE] {change_desc}")
                        actions = (
                            self.mode_manager.on_detect_actions()
                            if self.mode_manager.current
                            else []
                        )
                        await self.alert_pipeline.raise_alert(
                            trigger_label=f"zone_change_wp{index}",
                            detection_label=f"zone_{wp.zone_type}",
                            vlm_description=change_desc,
                            image_data=self.state.last_image,
                            writer=self._writer,
                            actions=actions,
                        )

                # Check for detections
                mode = self.mode_manager.current
                if mode:
                    detected = False
                    actions = self.mode_manager.on_detect_actions()
                    for label in mode.detect:
                        if label.lower() in scene.lower():
                            print(f"[DETECT] '{label}' at waypoint {index}")
                            detected = True
                            await self.led.set_state("confirmed", self._writer)
                            await self.power.report_threat(self._writer)

                            # Use alert pipeline for full response
                            await self.alert_pipeline.raise_alert(
                                trigger_label=f"patrol_wp{index}",
                                detection_label=label,
                                vlm_description=scene,
                                image_data=self.state.last_image,
                                writer=self._writer,
                                actions=actions,
                            )

                            # Start deterrent if configured
                            if "deterrent" in actions:
                                await self.deterrent.start(self._writer)
                                await self.led.set_state("panic", self._writer)

                    # If VLM says clear, report to power/deterrent for de-escalation
                    if not detected and "clear" in scene.lower():
                        await self.power.report_clear(self._writer)
                        if self.deterrent.active:
                            stood_down = await self.deterrent.report_clear(self._writer)
                            if stood_down:
                                await self.led.set_state("monitoring", self._writer)

            except Exception as e:
                logger.error("[Patrol] VLM error at waypoint %d: %s", index, e)

        # LED: back to green (monitoring)
        await self.led.set_state("monitoring", self._writer)

    async def _on_detection(self, label: str, image_data: bytes):
        """Called on a confirmed detection during patrol."""
        await self.led.set_state("confirmed", self._writer)
        asyncio.create_task(
            self.notifier.alert(f"Detection: {label}", title="Patrol Alert", image=image_data)
        )

    # ── Auto-docking callbacks ──────────────────────────────────────────

    async def _on_dock_needed(self):
        """Battery low — interrupt patrol and navigate to dock."""
        logger.warning("[Dock] Battery low — returning to dock")
        if self._writer:
            await self.led.set_state("low_battery", self._writer)
        if self.patrol:
            self.patrol.stop()
        await self.task_queue.put("RETURN_TO_DOCK")
        asyncio.create_task(
            self.notifier.alert(
                f"Battery low ({self.dock_manager.battery_mv}mV) — returning to dock",
                title="Low Battery",
            )
        )

    async def _on_undock_ready(self):
        """Battery full — undock and resume patrol."""
        logger.info("[Dock] Battery full — undocking")
        await self.task_queue.put("UNDOCK")

    async def _on_battery_critical(self):
        """Battery critical — emergency dock."""
        logger.critical("[Dock] Battery CRITICAL — emergency dock")
        if self._writer:
            await self.led.set_state("low_battery", self._writer)
        if self.patrol:
            self.patrol.stop()
        if self.runner:
            self.runner.interrupt("battery critical")
        await self.task_queue.put("RETURN_TO_DOCK")
        asyncio.create_task(
            self.notifier.alert(
                f"CRITICAL battery ({self.dock_manager.battery_mv}mV) — emergency dock!",
                title="BATTERY CRITICAL",
            )
        )

    # ── RTSP camera threat callback ─────────────────────────────────────

    async def _on_rtsp_threat(self, event: RtspEvent):
        """Called when an RTSP camera confirms a threat — dispatch robot."""
        logger.info(
            "[RTSP] Threat on '%s': %s (%s)",
            event.camera_name,
            event.detection_label,
            event.vlm_description,
        )
        # Escalate to ALERT mode
        if self._writer:
            await self.power.trigger_alert(
                f"rtsp_{event.camera_name}",
                self._writer,
            )
            await self.led.set_state("detecting", self._writer)

        # Alert pipeline (buzzer + evidence + notifications)
        mode = self.mode_manager.current
        actions = self.mode_manager.on_detect_actions() if mode else []
        if self._writer:
            await self.alert_pipeline.raise_alert(
                trigger_label=f"rtsp_{event.camera_name}",
                detection_label=event.detection_label,
                vlm_description=event.vlm_description,
                image_data=event.image_data,
                writer=self._writer,
                actions=actions,
            )

        # Dispatch robot to investigate zone
        if event.zone_waypoint:
            await self.task_queue.put(f"INVESTIGATE_ZONE {event.zone_waypoint}")

    # ── Sensor triggers (PIR/sound/IR) ────────────────────────────────────

    async def _process_sensor_triggers(self, sensor_flags: int, writer: asyncio.StreamWriter):
        """Process digital sensor triggers — escalate to ALERT + VLM confirm."""
        triggers = self.sensor_fusion.process_flags(sensor_flags)
        if not triggers:
            return

        # Escalate to ALERT mode
        first = triggers[0]
        await self.power.trigger_alert(first.label, writer)
        await self.led.set_state("detecting", writer)

        # Run VLM to confirm detection (if we have an image)
        if self.state.last_image:
            try:
                trigger_names = ", ".join(t.label for t in triggers)
                # Blocking OpenAI-client call — see _on_waypoint_reached for
                # why this is off-thread + timeout-bounded.
                scene = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vision.describe,
                        self.state.last_image,
                        context=f"sensor trigger: {trigger_names}",
                    ),
                    timeout=VLM_TIMEOUT_S,
                )
                print(f"[VLM@sensor] {scene}")

                # Check for actual threats
                mode = self.mode_manager.current
                actions = self.mode_manager.on_detect_actions() if mode else []
                detected = False

                if mode:
                    for label in mode.detect:
                        if label.lower() in scene.lower():
                            detected = True
                            await self.led.set_state("confirmed", writer)

                            # Raise alert through pipeline
                            event = await self.alert_pipeline.raise_alert(
                                trigger_label=first.label,
                                detection_label=label,
                                vlm_description=scene,
                                image_data=self.state.last_image,
                                writer=writer,
                                actions=actions,
                            )
                            if event:
                                self.sensor_fusion.mark_all_alerted(triggers)
                                # Start deterrent if configured
                                if "deterrent" in actions:
                                    await self.deterrent.start(writer)
                                    await self.led.set_state("panic", writer)
                            break

                # VLM says clear — report to power/deterrent
                if not detected:
                    if "clear" in scene.lower():
                        await self.power.report_clear(writer)
                        if self.deterrent.active:
                            stood_down = await self.deterrent.report_clear(writer)
                            if stood_down:
                                await self.led.set_state("monitoring", writer)
                    await self.led.set_state("monitoring", writer)

            except Exception as e:
                logger.error("[Sensor] VLM confirmation error: %s", e)
                await self.led.set_state("monitoring", writer)
        else:
            # No image available — still trigger alert, mark for VLM on next frame
            self.sensor_fusion.mark_all_alerted(triggers)

    # ── SLAM feed ──────────────────────────────────────────────────────────

    def _feed_slam(self, pkt):
        """Extract odometry deltas from sensor packet and feed SLAM."""
        if not hasattr(pkt, "odom_dist_mm"):
            return

        # compute deltas from previous odometry
        curr_dist = pkt.odom_dist_mm
        curr_hdg = pkt.odom_hdg_cdeg
        prev_dist = self._prev_odom.get("dist_mm", 0)
        prev_hdg = self._prev_odom.get("heading_cdeg", 0)

        d_dist = curr_dist - prev_dist
        d_hdg = curr_hdg - prev_hdg
        # wrap heading delta
        if d_hdg > 18000:
            d_hdg -= 36000
        elif d_hdg < -18000:
            d_hdg += 36000

        self._prev_odom = {"dist_mm": curr_dist, "heading_cdeg": curr_hdg}

        # forward distance as dx (robot frame), no lateral movement for diff drive
        if self.patrol and d_dist != 0:
            # empty scan for now — real LiDAR packets would feed this
            self.patrol.feed_sensors(d_dist, 0, d_hdg, [])

    # ── Camera frame decoder ─────────────────────────────────────────────────

    # Camera payload format constants
    _CAM_HDR_SIZE = 5
    _CAM_FMT_GRAY8 = 0
    _CAM_FMT_JPEG = 1

    # OOM guard: reject attacker-controlled dimensions that would allocate an
    # enormous buffer.  The kernel-side camera drivers encode at 320×240 for
    # the robot's on-board camera, but we allow up to 1920×1080 so the brain
    # can also accept frames from higher-resolution USB cameras without changes.
    # A 65535×65535 Gray8 frame would be ~4 GB — reject it before PIL touches it.
    _MAX_CAM_WIDTH = 1920
    _MAX_CAM_HEIGHT = 1080
    _MAX_CAM_PIXELS = _MAX_CAM_WIDTH * _MAX_CAM_HEIGHT  # 2 073 600

    def _decode_camera(self, payload: bytes) -> bytes | None:
        """Decode a CAMERA_FRAME payload. Returns JPEG bytes for the VLM.

        Header: width(u16 LE) + height(u16 LE) + format(u8)
        If format is Gray8, converts raw grayscale to JPEG via Pillow.
        If format is JPEG, returns the bytes as-is.
        """
        if len(payload) < self._CAM_HDR_SIZE:
            return None
        width, height, fmt = struct.unpack_from("<HHB", payload)
        image_data = payload[self._CAM_HDR_SIZE :]
        if not image_data:
            return None

        # Guard against OOM: reject frames with out-of-range dimensions before
        # any buffer is allocated.  Malformed/attacker-controlled packets cannot
        # trigger a multi-GB allocation this way.
        if width > self._MAX_CAM_WIDTH or height > self._MAX_CAM_HEIGHT:
            print(
                f"[BRAIN] Camera: frame dimensions {width}×{height} exceed "
                f"cap {self._MAX_CAM_WIDTH}×{self._MAX_CAM_HEIGHT} — dropped"
            )
            return None
        if width * height > self._MAX_CAM_PIXELS:
            print(
                f"[BRAIN] Camera: {width}×{height}={width * height} px exceeds "
                f"max {self._MAX_CAM_PIXELS} px — dropped"
            )
            return None

        if fmt == self._CAM_FMT_JPEG:
            return image_data

        if fmt == self._CAM_FMT_GRAY8:
            expected = width * height
            if len(image_data) < expected:
                print(f"[BRAIN] Camera: Gray8 truncated ({len(image_data)}/{expected})")
                return None
            try:
                from PIL import Image
                import io

                img = Image.frombytes("L", (width, height), image_data[:expected])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                return buf.getvalue()
            except ImportError:
                print("[BRAIN] Camera: Pillow not installed, cannot convert Gray8")
                return None

        print(f"[BRAIN] Camera: unknown format {fmt}")
        return None

    # ── Safety ────────────────────────────────────────────────────────────────

    # Channel counts per actuator type for emergency stop
    _STOP_CHANNELS = {
        ROBOT_WHEELED: 2,
        ROBOT_DRONE: 4,
        ROBOT_HUMANOID: 12,
    }

    async def emergency_stop(self, reason: str = "emergency_stop") -> None:
        """Central emergency-stop entry point for every operator/error-
        triggered stop path (HTTP /stop, Telegram /stop, in-brain perception-
        error stop). Robot-type aware: a drone must HOVER in place, not
        receive a diff-drive zero-channel ActuatorCmd — wrong actuator_type
        and channel count, which the kernel would either reject or (worse)
        misinterpret as throttle/roll/pitch/yaw=0 (freefall, not a safe
        state) rather than a hold.
        """
        # getattr-defensive like the call sites this replaces (api.py/
        # telegram_bot.py used getattr(self.brain, "runner", None) before
        # routing through here): this method now runs against BrainServer
        # instances that may be partially constructed (tests build one via
        # BrainServer.__new__() + hand-picked attributes).
        runner = getattr(self, "runner", None)
        if runner:
            runner.interrupt(reason)
        if not self._writer:
            return
        robot_type = getattr(self, "robot_type", ROBOT_WHEELED)
        if robot_type == ROBOT_DRONE:
            cmd = self.policy.translate("HOVER")
        else:
            n_channels = self._STOP_CHANNELS.get(robot_type, 2)
            cmd = ActuatorCmd.stop(n_channels=n_channels)
        await protocol.send_packet(self._writer, ACTUATOR_CMD, cmd.to_bytes())

    async def _safety_check(self, pkt, writer: asyncio.StreamWriter):
        """Hard safety stops — always evaluated, regardless of planner state.

        Uses SafetyProfile to run robot-type-specific checks.
        On any violation: emergency stop + interrupt runner + notify + log.
        """
        violations = self.safety_profile.check(pkt)

        if not violations:
            return

        # Log every violation
        for v in violations:
            logger.warning("[SAFETY] %s", v)
            print(f"[SAFETY] {v}")

        # Interrupt any running skill
        reason = violations[0]
        if self.runner:
            self.runner.interrupt(reason)

        # Determine emergency response based on robot type
        if self.robot_type == ROBOT_DRONE:
            await self._drone_emergency(pkt, writer, violations)
        else:
            # Wheeled / Humanoid / Ackermann: full stop
            n_channels = self._STOP_CHANNELS.get(self.robot_type, 2)
            cmd = ActuatorCmd.stop(n_channels=n_channels)
            await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())

        # LED feedback for violations
        battery_violations = [v for v in violations if "battery" in v.lower()]
        if battery_violations:
            await self.led.set_state("low_battery", writer)
        else:
            await self.led.set_state("confirmed", writer)

        # Notify operator for battery violations
        for bv in battery_violations:
            asyncio.create_task(self.notifier.alert(bv, title="Battery Alert"))

    async def _drone_emergency(self, pkt, writer: asyncio.StreamWriter, violations: list[str]):
        """Handle drone-specific emergency actions.

        Priority order:
          1. Critical battery -> immediate LAND
          2. Low battery -> RTL (Return-To-Launch)
          3. Other violation -> HOVER (safest for drone)
        """
        action = self.safety_profile.drone_action(pkt)

        if action == "land":
            # Immediate landing — send LAND command via policy
            logger.critical("[SAFETY] Drone CRITICAL — forcing LAND")
            print("[SAFETY] Drone CRITICAL — forcing LAND")
            cmd = self.policy.translate("LAND")
            await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())
            asyncio.create_task(
                self.notifier.alert("DRONE CRITICAL BATTERY — LANDING NOW", title="DRONE EMERGENCY")
            )

        elif action == "rtl":
            # Return to launch
            logger.warning("[SAFETY] Drone low battery — RTL")
            print("[SAFETY] Drone low battery — RTL")
            cmd = self.policy.translate("RETURN_HOME")
            await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())
            asyncio.create_task(
                self.notifier.alert("Drone low battery — returning home", title="Drone RTL")
            )

        else:
            # Other violation — hover in place (safest for drone)
            logger.warning("[SAFETY] Drone violation — HOVER")
            print("[SAFETY] Drone violation — HOVER")
            cmd = self.policy.translate("HOVER")
            await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())

    # ── Perception cycle ──────────────────────────────────────────────────────

    async def _handle_camera_frame(self, payload: bytes, writer: asyncio.StreamWriter) -> None:
        """A CAMERA_FRAME, from either connection; the cycle's commands go on
        `writer`, the control connection."""
        _metrics_mod.M.pkt_dispatch_handled_total.labels(type=f"0x{CAMERA_FRAME:02x}").inc()
        image_data = self._decode_camera(payload)
        if image_data:
            self.state.update_image(image_data)
            await self._perception_cycle(image_data, writer)

    async def _on_camera_frame(self, payload: bytes) -> None:
        """A CAMERA_FRAME from the camera connection (camera_link.py). The
        cycle commands the control connection live when the frame arrived;
        with none, the frame is dropped. If control drops during the cycle,
        the send on its writer fails and the cycle's error path stops."""
        writer = self._writer
        if writer is None:
            return
        await self._handle_camera_frame(payload, writer)

    async def _perception_cycle(self, image_data: bytes, writer: asyncio.StreamWriter):
        """Run VLM → LLM → policy → send command. Skipped in ECO mode."""
        # In ECO mode, camera is OFF — skip VLM processing
        if self.power.is_eco:
            return

        # If actively recording evidence, save frame
        if self.alert_pipeline.active_evidence:
            await self.alert_pipeline.save_evidence_frame(image_data)

        # B-C5: refuse to command movement from stale sensor data. Camera
        # frames (which drive this cycle) and SENSOR_PACKETs are separate
        # pipelines — if the kernel stops sending sensors but keeps sending
        # frames, the LLM would otherwise decide against a frozen
        # self.state.sensors/odom snapshot (or, right after a reconnect
        # before the first sensor packet arrives, an all-default one).
        # getattr-defensive like emergency_stop(): this method also runs
        # against partially-constructed BrainServer instances in tests.
        last_sensor_time = getattr(self.state, "last_sensor_time", None)
        comms_timeout_s = getattr(getattr(self, "safety_profile", None), "comms_timeout_s", None)
        if last_sensor_time is not None and comms_timeout_s is not None:
            comms_age = time.time() - last_sensor_time
            if comms_age > comms_timeout_s:
                logger.warning(
                    "[SAFETY] Sensor data stale (%.1fs > %.1fs) — refusing to act, STOP",
                    comms_age, comms_timeout_s,
                )
                await self.emergency_stop("stale sensor data")
                return

        if self.mode_manager.uses_llm():
            # LLM reactive mode: VLM describes scene → LLM picks single action
            try:
                # Both calls below hit LM Studio via the synchronous OpenAI
                # client — blocking the event loop here would stall every
                # other connection (sensor packets, HTTP API, Telegram) for
                # the duration of the HTTP round-trip. Run off-thread with a
                # hard timeout so a stuck/dead backend degrades to "skip
                # this cycle" (caught below, falls through to STOP) instead
                # of freezing the whole brain.
                scene = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vision.describe,
                        image_data,
                        context=self.mode_manager.current_name,
                    ),
                    timeout=VLM_TIMEOUT_S,
                )
                print(f"[VLM] {scene}")

                action = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.planner.decide,
                        scene=scene,
                        sensors=self.state.sensors,
                        task=self.mode_manager.current_name,
                        odom=self.state.odom,
                    ),
                    timeout=LLM_TIMEOUT_S,
                )
                print(f"[LLM] {action}")

                cmd = self.policy.from_text(action)
                # RFC-0035: reactive-LLM actions are lower-confidence than
                # deterministic plan/scripted steps → mark them so the kernel
                # tightens the motor envelope (acts, but cautiously). Never on an
                # emergency stop (that must not be slowed).
                if not (cmd.flags & protocol.FLAG_EMERGENCY):
                    cmd.flags |= protocol.FLAG_LOW_CONFIDENCE
                await protocol.send_packet(writer, ACTUATOR_CMD, cmd.to_bytes())

                # RFC-0036: perception succeeded — reset the blind streak and, if
                # we had armed degraded mode, clear it now that the brain can see.
                self._perception_fail_streak = 0
                if self._degraded_sent:
                    await self._send_degrade_cmd(protocol.DEGRADE_CLEAR)
                    self._degraded_sent = False
                    print("[BRAIN] perception recovered — degraded mode cleared")

            except Exception as e:
                print(f"[BRAIN] AI error: {e} — STOP")
                # Robot-type-aware stop (a drone must HOVER, not zero out
                # diff-drive channels it doesn't have) — see emergency_stop().
                await self.emergency_stop("perception/LLM error")
                # RFC-0036: count consecutive blind cycles. Once persistently
                # blind (not a single transient hiccup), arm degraded mode once
                # so the kernel contains userspace until the brain recovers.
                self._perception_fail_streak += 1
                if (
                    self._perception_fail_streak >= DEGRADE_PERCEPTION_FAIL_THRESHOLD
                    and not self._degraded_sent
                ):
                    await self._send_degrade_cmd(protocol.DEGRADE_REASON_PERCEPTION_BLIND)
                    self._degraded_sent = True
                    print(
                        f"[BRAIN] perception blind {self._perception_fail_streak}× "
                        "— degraded mode armed"
                    )

    # ── Status / type negotiation ─────────────────────────────────────────────

    async def _on_status(self, pkt: StatusPacket):
        # B-A14: this is a VALIDATION, not a negotiation. It used to rebuild
        # self.policy / self.safety_profile from the robot_type byte on the
        # wire, which meant one unauthenticated STATUS packet re-armed the
        # whole safety stack for a different chassis: _safety_check routed to
        # _drone_emergency and emergency_stop sent HOVER to a diff-drive base.
        # The robot type is a deployment fact the operator declares in
        # config.yaml; a peer claiming otherwise is either misconfigured or
        # hostile, and either way the answer is to say so loudly and keep the
        # configured profile — never to adopt the claim.
        if pkt.robot_type != self._configured_robot_type:
            claimed = ROBOT_TYPE_NAME_BY_ID.get(pkt.robot_type, f"unknown({pkt.robot_type})")
            configured = ROBOT_TYPE_NAME_BY_ID.get(
                self._configured_robot_type, ROBOT_TYPE_DEFAULT_NAME
            )
            logger.warning(
                "[BRAIN] STATUS claims robot_type=%s but config declares %s — "
                "REJECTED, keeping the configured safety profile. Fix "
                "robot.type in config.yaml if the chassis really changed.",
                claimed,
                configured,
            )

    # ── Experience loop (Hyperagent) ────────────────────────────────────────

    def _on_plan_done(
        self,
        plan: list,
        outcome: str,
        steps_executed: int,
        error: str,
        interrupt_reason: str,
        duration_s: float,
    ):
        """Called by SkillRunner when a plan finishes — records to experience."""
        task_desc = self._current_task_desc or "unknown"
        context = self._build_experience_context()
        self.experience.record(
            task=task_desc,
            plan=plan,
            outcome=outcome,
            context=context,
            steps_executed=steps_executed,
            error=error,
            interrupt_reason=interrupt_reason,
            duration_s=duration_s,
        )
        # Trigger meta-review if due (runs in background)
        if self.meta_reviewer.should_review():
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(self._run_meta_review())
            )

    def _build_experience_context(self) -> str:
        """Build a context string from current robot state for experience."""
        parts = []
        if self.state.sensors.get("battery_mv"):
            parts.append(f"battery={self.state.sensors['battery_mv']}mV")
        if self.mode_manager.current_name:
            parts.append(f"mode={self.mode_manager.current_name}")
        if self.state.sensors.get("range_front_mm"):
            parts.append(f"range_front={self.state.sensors['range_front_mm']}mm")
        return ", ".join(parts) if parts else ""

    async def _run_meta_review(self):
        """Run meta-review in background (LLM call)."""
        try:
            loop = asyncio.get_event_loop()
            rules = await loop.run_in_executor(None, self.meta_reviewer.review)
            if rules:
                logger.info("[Meta] Updated %d heuristic rules", len(rules))
        except Exception as e:
            logger.error("[Meta] Review failed: %s", e)

    # ── Task queue worker ─────────────────────────────────────────────────────

    async def _task_worker(self):
        """Background coroutine that processes queued tasks via TaskPlanner."""
        while True:
            task_desc = await self.task_queue.get()
            print(f"[BRAIN] Task: {task_desc!r}")
            try:
                # Handle patrol/mapping commands directly
                if task_desc.strip().upper() == "MAP":
                    if self.patrol:
                        await self.led.set_state("mapping", self._writer)
                        await self.patrol.run_mapping()
                        await self.led.set_state("monitoring", self._writer)
                elif task_desc.strip().upper() == "PATROL":
                    if self.patrol:
                        await self.led.set_state("monitoring", self._writer)
                        await self.patrol.run_patrol(loop=True)
                elif task_desc.strip().upper() == "PATROL_ONCE":
                    if self.patrol:
                        await self.led.set_state("monitoring", self._writer)
                        await self.patrol.run_patrol(loop=False)
                        await self.led.set_state("monitoring", self._writer)
                elif task_desc.strip().upper() == "STOP_PATROL":
                    if self.patrol:
                        self.patrol.stop()
                        await self.led.set_state("monitoring", self._writer)
                elif task_desc.strip().upper() == "DETERRENT":
                    await self.deterrent.start(self._writer)
                    await self.led.set_state("panic", self._writer)
                elif task_desc.strip().upper() == "STAND_DOWN":
                    await self.deterrent.stand_down(self._writer)
                    await self.led.set_state("monitoring", self._writer)
                elif task_desc.strip().upper() == "SILENCE":
                    await self.deterrent.silence(self._writer)
                elif task_desc.strip().upper() == "PANIC":
                    await self.deterrent.start(self._writer)
                    # Fast-escalate to max level
                    for _ in range(DeterrentLevel.AGGRESSIVE):
                        await self.deterrent.escalate(self._writer)
                    await self.led.set_state("panic", self._writer)
                elif task_desc.strip().upper() == "RETURN_TO_DOCK":
                    if self.patrol:
                        await self.led.set_state("docking", self._writer)
                        dock = self.dock_manager.dock_info
                        from planner.mapper import Waypoint

                        wp = Waypoint(
                            x_mm=dock.x_mm,
                            y_mm=dock.y_mm,
                            heading_cdeg=dock.heading_cdeg,
                            label="dock",
                        )
                        self.dock_manager.state = DockState.NAVIGATING
                        try:
                            reached = await asyncio.wait_for(
                                self.patrol._navigate_to(wp),
                                timeout=NAVIGATION_TIMEOUT_S,
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "[Task] Navigation to dock timed out after %.0fs",
                                NAVIGATION_TIMEOUT_S,
                            )
                            reached = False
                        if reached:
                            self.dock_manager.report_arrived_at_dock()
                            # IR homing would happen here in real hardware
                            self.dock_manager.report_docked()
                            self.dock_manager.report_charging()
                            await self.led.set_state("docking", self._writer)
                        else:
                            self.dock_manager.abort()
                            await self.led.set_state("monitoring", self._writer)
                elif task_desc.strip().upper() == "UNDOCK":
                    self.dock_manager.start_undock_sequence()
                    if self.runner:
                        await self.runner.execute_one("UNDOCK")
                    self.dock_manager.report_undocked()
                    await self.led.set_state("monitoring", self._writer)
                    # Resume patrol
                    await self.task_queue.put("PATROL")
                elif task_desc.strip().upper() == "DOCK":
                    # Manual dock command (Telegram /dock)
                    self.dock_manager.start_dock_sequence()
                    await self.task_queue.put("RETURN_TO_DOCK")
                elif task_desc.strip().upper().startswith("TRACK_INTRUDER"):
                    parts = task_desc.strip().split(maxsplit=1)
                    target = parts[1] if len(parts) > 1 else "person"
                    self.tracker.start(target)
                    await self.led.set_state("tracking", self._writer)
                elif task_desc.strip().upper().startswith("INVESTIGATE_ZONE"):
                    parts = task_desc.strip().split(maxsplit=1)
                    zone = parts[1] if len(parts) > 1 else ""
                    if self.patrol and zone:
                        await self.led.set_state("detecting", self._writer)
                        # Navigate to zone waypoint, scan, report
                        wp = self.mapper.get_waypoint_by_label(zone)
                        if wp:
                            from planner.mapper import Waypoint

                            try:
                                reached = await asyncio.wait_for(
                                    self.patrol._navigate_to(wp),
                                    timeout=NAVIGATION_TIMEOUT_S,
                                )
                            except asyncio.TimeoutError:
                                logger.error(
                                    "[Task] Navigation to zone '%s' timed " "out after %.0fs",
                                    zone,
                                    NAVIGATION_TIMEOUT_S,
                                )
                                reached = False
                            if reached:
                                await self.patrol._rotate_360()
                                if self._on_waypoint_reached:
                                    await self._on_waypoint_reached(wp, 0)
                        else:
                            logger.warning(
                                "[Task] Zone '%s' not found in map",
                                zone,
                            )
                        await self.led.set_state("monitoring", self._writer)
                else:
                    self._current_task_desc = task_desc
                    context = self._build_experience_context()
                    # Blocking OpenAI-client call — see _perception_cycle for
                    # why this must be off-thread + timeout-bounded rather
                    # than called inline from this coroutine.
                    plan = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.task_planner.plan,
                            task_desc,
                            context=context,
                        ),
                        timeout=LLM_TIMEOUT_S,
                    )
                    print(f"[BRAIN] Plan: {plan}")
                    if self.runner:
                        await self.runner.execute_plan(plan)
            except Exception as e:
                print(f"[BRAIN] Task error: {e}")
            finally:
                self.task_queue.task_done()

    # ── Server entry point ────────────────────────────────────────────────────

    async def run(self):
        port = self.config["robot"]["listen_port"]
        # The robot link is a Noise session and has no other mode (the kernel
        # speaks nothing else), so the brain cannot run without the link key:
        # any peer that reached the port would otherwise be accepted as THE
        # robot and drive its motors. Both HTTP tiers refuse to start without
        # a key or an explicit opt-in as well (api.py APIServer.run,
        # control_plane.auth.BearerAuth.__init__).
        if not protocol.enable_encrypt_link():
            raise RuntimeError(
                "[BRAIN] Cannot start: the robot link needs a key. Set "
                "AZOS_BRAIN_LINK_KEY to the shared secret provisioned as the "
                "kernel's link key (64 hex digits)."
            )
        print(
            "[BRAIN] secure_channel: Noise session per connection "
            "(AZOS_BRAIN_LINK_KEY; handshake required)"
        )
        bind_host = LINK_BIND_ANY
        server = await asyncio.start_server(self.handle_robot, bind_host, port)
        print(f"[BRAIN] Listening on {bind_host}:{port}")
        # C1: the camera connection's port, on the same bind address. Absent
        # or 0: no camera listener; frames arrive on the control connection.
        camera_port = int(self.config["robot"].get("camera_port", 0))
        camera_server = None
        if camera_port:
            camera_server = await asyncio.start_server(self.camera.handle, bind_host, camera_port)
            print(f"[BRAIN] Camera connection on {bind_host}:{camera_port}")
        print(f"[BRAIN] Robot type: {self.config['robot'].get('type', 'wheeled')}")
        print(f"[BRAIN] Mode: {self.mode_manager.current_name}")
        print(f"[BRAIN] VLM: {self.config['lmstudio']['vlm_model']}")
        print(f"[BRAIN] LLM: {self.config['lmstudio']['llm_model']}")

        tasks = [server.serve_forever(), self._task_worker()]
        if camera_server is not None:
            tasks.append(camera_server.serve_forever())

        if self.tg_bot:
            tasks.append(self.tg_bot.run())
            print("[BRAIN] Telegram bot enabled")

        if self.api:
            tasks.append(self.api.run())

        if self.rtsp_monitor and self.rtsp_monitor.camera_count > 0:
            tasks.append(self.rtsp_monitor.start())
            print(f"[BRAIN] RTSP monitoring: {self.rtsp_monitor.camera_count} cameras")

        try:
            async with server:
                await asyncio.gather(*tasks)
        finally:
            if camera_server is not None:
                camera_server.close()


def main():
    brain = BrainServer()
    asyncio.run(brain.run())


if __name__ == "__main__":
    main()
