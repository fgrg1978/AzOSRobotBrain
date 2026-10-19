# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/alert.py — alert pipeline (buzzer, evidence, notifications)."""

import asyncio
import json
import os
import tempfile
import time
import pytest

from planner.alert import (
    AlertPipeline,
    AlertEvent,
    ALERT_COOLDOWN_S,
    EVIDENCE_FRAMES,
    EVIDENCE_RETENTION_DAYS,
    BUZZER_BEEP,
    BUZZER_SIREN,
    BUZZER_OFF,
)
from protocol import ConfigCmd, CONFIG_CMD, BUZZER_CONFIG_KEY


class TestAlertPipeline:

    def _make_pipeline(self, cooldown_s=0.0, evidence_frames=10):
        packets_sent = []
        notifications = []

        async def mock_send(writer, pkt_type, payload):
            cmd = ConfigCmd.from_bytes(payload)
            packets_sent.append((pkt_type, cmd.config_key, cmd.value))

        class MockNotifier:
            async def alert(self, message, title="", image=None):
                notifications.append(
                    {"message": message, "title": title, "has_image": image is not None}
                )
                return {"mock": True}

        tmpdir = tempfile.mkdtemp()
        pipeline = AlertPipeline(
            send_packet=mock_send,
            notifier=MockNotifier(),
            evidence_dir=tmpdir,
            cooldown_s=cooldown_s,
            evidence_frames=evidence_frames,
        )
        return pipeline, packets_sent, notifications, tmpdir

    def test_raise_alert_basic(self):
        p, packets, notifs, _ = self._make_pipeline()
        writer = object()

        event = asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="A person near the door",
                image_data=b"\xff\xd8\xff\xe0test",
                writer=writer,
                actions=["notify", "alert"],
            )
        )

        assert event is not None
        assert event.trigger_label == "pir_motion"
        assert event.detection_label == "person"
        assert event.vlm_description == "A person near the door"
        assert event.frames_saved == 1
        assert event.notified is True

    def test_raise_alert_sends_buzzer(self):
        p, packets, _, _ = self._make_pipeline()
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="test",
                writer=writer,
                actions=["buzzer_alert"],
            )
        )

        # should have sent buzzer beep
        buzzer_pkts = [pk for pk in packets if pk[1] == BUZZER_CONFIG_KEY]
        assert len(buzzer_pkts) == 1
        assert buzzer_pkts[0][2] == BUZZER_BEEP

    def test_raise_alert_no_buzzer_without_action(self):
        p, packets, _, _ = self._make_pipeline()
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="test",
                writer=writer,
                actions=["notify"],  # no buzzer action
            )
        )

        buzzer_pkts = [pk for pk in packets if pk[1] == BUZZER_CONFIG_KEY]
        assert len(buzzer_pkts) == 0

    def test_cooldown_blocks_repeat_alert(self):
        p, _, _, _ = self._make_pipeline(cooldown_s=10.0)
        writer = object()

        e1 = asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="test",
                writer=writer,
                actions=[],
            )
        )
        assert e1 is not None

        # same trigger+detection should be cooled down
        e2 = asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="test again",
                writer=writer,
                actions=[],
            )
        )
        assert e2 is None

    def test_cooldown_allows_different_label(self):
        p, _, _, _ = self._make_pipeline(cooldown_s=10.0)
        writer = object()

        e1 = asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="person",
                vlm_description="test",
                writer=writer,
                actions=[],
            )
        )
        assert e1 is not None

        # different detection label should not be cooled down
        e2 = asyncio.run(
            p.raise_alert(
                trigger_label="pir_motion",
                detection_label="fire",
                vlm_description="fire detected",
                writer=writer,
                actions=[],
            )
        )
        assert e2 is not None

    def test_evidence_saves_frame(self):
        p, _, _, tmpdir = self._make_pipeline()
        writer = object()

        event = asyncio.run(
            p.raise_alert(
                trigger_label="sound_event",
                detection_label="glass_break",
                vlm_description="broken glass",
                image_data=b"\xff\xd8\xff\xe0jpeg_data",
                writer=writer,
                actions=[],
            )
        )

        assert event.frames_saved == 1
        assert event.evidence_dir != ""
        assert os.path.exists(event.evidence_dir)

        # check frame file
        frame_path = os.path.join(event.evidence_dir, "frame_000.jpg")
        assert os.path.exists(frame_path)

        # check metadata
        meta_path = os.path.join(event.evidence_dir, "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["trigger"] == "sound_event"
        assert meta["detection"] == "glass_break"

    def test_evidence_additional_frames(self):
        p, _, _, tmpdir = self._make_pipeline(evidence_frames=3)
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="test",
                image_data=b"\xff\xd8frame1",
                writer=writer,
                actions=[],
            )
        )

        # save additional frames
        asyncio.run(p.save_evidence_frame(b"\xff\xd8frame2"))
        asyncio.run(p.save_evidence_frame(b"\xff\xd8frame3"))

        assert p.active_evidence.frames_saved == 3

        # should not save beyond limit
        asyncio.run(p.save_evidence_frame(b"\xff\xd8frame4"))
        assert p.active_evidence.frames_saved == 3

    def test_finish_evidence(self):
        p, _, _, _ = self._make_pipeline()
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="test",
                image_data=b"\xff\xd8data",
                writer=writer,
                actions=[],
            )
        )

        assert p.active_evidence is not None
        p.finish_evidence()
        assert p.active_evidence is None

    def test_no_evidence_without_image(self):
        p, _, _, _ = self._make_pipeline()
        writer = object()

        event = asyncio.run(
            p.raise_alert(
                trigger_label="ir",
                detection_label="proximity",
                vlm_description="test",
                writer=writer,
                actions=[],
            )
        )

        assert event.frames_saved == 0
        assert event.evidence_dir == ""

    def test_notification_sent(self):
        p, _, notifs, _ = self._make_pipeline()
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="Person at door",
                image_data=b"\xff\xd8photo",
                writer=writer,
                actions=["notify"],
            )
        )

        assert len(notifs) == 1
        assert "PERSON" in notifs[0]["message"]
        assert notifs[0]["has_image"] is True

    def test_no_notification_without_action(self):
        p, _, notifs, _ = self._make_pipeline()
        writer = object()

        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="test",
                writer=writer,
                actions=["alert"],  # no notify
            )
        )

        assert len(notifs) == 0

    def test_alert_count_tracks_events(self):
        p, _, _, _ = self._make_pipeline()
        writer = object()

        assert p.alert_count == 0

        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="test 1",
                writer=writer,
                actions=[],
            )
        )
        assert p.alert_count == 1

        asyncio.run(
            p.raise_alert(
                trigger_label="sound",
                detection_label="fire",
                vlm_description="test 2",
                writer=writer,
                actions=[],
            )
        )
        assert p.alert_count == 2

    def test_alerts_list_returns_copy(self):
        p, _, _, _ = self._make_pipeline()
        alerts = p.alerts
        assert alerts == []
        # modifying returned list shouldn't affect internal state
        alerts.append(None)
        assert p.alert_count == 0

    def test_buzzer_on_off(self):
        p, packets, _, _ = self._make_pipeline()
        writer = object()

        asyncio.run(p.buzzer_on(BUZZER_SIREN, writer))
        asyncio.run(p.buzzer_off(writer))

        assert len(packets) == 2
        assert packets[0][2] == BUZZER_SIREN
        assert packets[1][2] == BUZZER_OFF

    def test_no_writer_no_buzzer_send(self):
        p, packets, _, _ = self._make_pipeline()
        asyncio.run(p.buzzer_on(BUZZER_BEEP))  # no writer
        assert len(packets) == 0

    def test_repr(self):
        p, _, _, _ = self._make_pipeline()
        r = repr(p)
        assert "AlertPipeline" in r
        assert "idle" in r

    def test_repr_recording(self):
        p, _, _, _ = self._make_pipeline()
        writer = object()
        asyncio.run(
            p.raise_alert(
                trigger_label="pir",
                detection_label="person",
                vlm_description="test",
                image_data=b"\xff\xd8data",
                writer=writer,
                actions=[],
            )
        )
        r = repr(p)
        assert "recording" in r

    def test_is_cooled_down_initially(self):
        p, _, _, _ = self._make_pipeline(cooldown_s=10.0)
        assert p.is_cooled_down("pir_motion:person") is True

    def test_default_constants(self):
        assert ALERT_COOLDOWN_S > 0
        assert EVIDENCE_FRAMES > 0
        assert EVIDENCE_RETENTION_DAYS > 0


class TestProtocolBuzzer:

    def test_buzzer_factory(self):
        cmd = ConfigCmd.buzzer(BUZZER_BEEP)
        assert cmd.config_key == BUZZER_CONFIG_KEY
        assert cmd.value == BUZZER_BEEP

    def test_buzzer_roundtrip(self):
        cmd = ConfigCmd.buzzer(BUZZER_SIREN)
        data = cmd.to_bytes()
        cmd2 = ConfigCmd.from_bytes(data)
        assert cmd2.config_key == BUZZER_CONFIG_KEY
        assert cmd2.value == BUZZER_SIREN

    def test_buzzer_codes_unique(self):
        codes = [BUZZER_OFF, BUZZER_BEEP, BUZZER_SIREN]
        assert len(codes) == len(set(codes))


class TestSensorFlagsProtocol:

    def test_sensor_packet_with_flags(self):
        from protocol import SensorPacket, SENSOR_FLAG_PIR, SENSOR_FLAG_SOUND, SENSOR_FLAG_IR

        flags = SENSOR_FLAG_PIR | SENSOR_FLAG_SOUND
        pkt = SensorPacket(
            timestamp_ms=1000,
            battery_mv=7400,
            accel_mg=(0, 0, 1000),
            gyro_mdps=(0, 0, 0),
            odom_dist_mm=0,
            odom_hdg_cdeg=0,
            encoder_l=0,
            encoder_r=0,
            range_front_mm=500,
            range_right_mm=300,
            sensor_flags=flags,
        )
        data = pkt.to_bytes()
        pkt2 = SensorPacket.from_bytes(data)
        assert pkt2.sensor_flags == flags
        assert pkt2.sensor_flags & SENSOR_FLAG_PIR
        assert pkt2.sensor_flags & SENSOR_FLAG_SOUND
        assert not (pkt2.sensor_flags & SENSOR_FLAG_IR)

    def test_sensor_packet_legacy_no_flags(self):
        """Legacy 62-byte packet should parse with sensor_flags=0."""
        from protocol import SensorPacket
        import struct

        # Build a legacy packet (no flags field)
        hdr = struct.pack("<Q3i3iH", 1000, 0, 0, 1000, 0, 0, 0, 7400)
        whl = struct.pack("<2i2q2H", 0, 0, 0, 0, 500, 300)
        data = hdr + whl
        pkt = SensorPacket.from_bytes(data)
        assert pkt.sensor_flags == 0
        assert pkt.battery_mv == 7400

    def test_sensor_packet_with_flags_roundtrip(self):
        from protocol import SensorPacket, SENSOR_FLAG_IR

        pkt = SensorPacket(
            timestamp_ms=2000,
            battery_mv=7200,
            accel_mg=(100, -50, 980),
            gyro_mdps=(10, 20, 30),
            odom_dist_mm=1000,
            odom_hdg_cdeg=9000,
            encoder_l=500,
            encoder_r=500,
            range_front_mm=200,
            range_right_mm=400,
            sensor_flags=SENSOR_FLAG_IR,
        )
        data = pkt.to_bytes()
        pkt2 = SensorPacket.from_bytes(data)
        assert pkt2.sensor_flags == SENSOR_FLAG_IR
        assert pkt2.odom_dist_mm == 1000
        assert pkt2.battery_mv == 7200
