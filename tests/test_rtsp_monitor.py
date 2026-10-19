# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception/rtsp_monitor.py — RTSP camera monitoring."""

import asyncio
import io
import time

from perception.rtsp_monitor import (
    RtspMonitor,
    RtspCamera,
    RtspEvent,
    cameras_from_config,
    DEFAULT_SCAN_INTERVAL_S,
    RTSP_CAPTURE_TIMEOUT_S,
    RTSP_RECONNECT_DELAY_S,
    RTSP_MAX_CONSECUTIVE_ERRORS,
)
from perception.motion_detect import MOTION_THRESHOLD_PCT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeVision:
    """Fake VisionPerception for testing."""

    def __init__(self, response: str = "CLEAR"):
        self.response = response
        self.calls = []

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        self.calls.append((len(image_bytes), context))
        return self.response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_scan_interval_positive(self):
        assert DEFAULT_SCAN_INTERVAL_S > 0

    def test_capture_timeout_positive(self):
        assert RTSP_CAPTURE_TIMEOUT_S > 0

    def test_reconnect_delay_positive(self):
        assert RTSP_RECONNECT_DELAY_S > 0

    def test_max_errors_positive(self):
        assert RTSP_MAX_CONSECUTIVE_ERRORS > 0


# ---------------------------------------------------------------------------
# RtspCamera dataclass
# ---------------------------------------------------------------------------


class TestRtspCamera:
    def test_default_values(self):
        cam = RtspCamera(name="test", url="rtsp://1.2.3.4/stream")
        assert cam.name == "test"
        assert cam.url == "rtsp://1.2.3.4/stream"
        assert cam.zone_waypoint == ""
        assert cam.scan_interval_s == DEFAULT_SCAN_INTERVAL_S
        assert cam.motion_threshold_pct == MOTION_THRESHOLD_PCT
        assert cam.enabled is True

    def test_custom_values(self):
        cam = RtspCamera(
            name="garden",
            url="rtsp://cam/stream",
            zone_waypoint="jardin",
            scan_interval_s=5.0,
            motion_threshold_pct=20,
            enabled=False,
        )
        assert cam.name == "garden"
        assert cam.zone_waypoint == "jardin"
        assert cam.scan_interval_s == 5.0
        assert cam.motion_threshold_pct == 20
        assert cam.enabled is False


# ---------------------------------------------------------------------------
# RtspEvent dataclass
# ---------------------------------------------------------------------------


class TestRtspEvent:
    def test_creation(self):
        event = RtspEvent(
            camera_name="cam1",
            zone_waypoint="zona_norte",
            motion_score=45.0,
            vlm_description="person detected near fence",
            detection_label="person",
            image_data=b"jpeg_data",
        )
        assert event.camera_name == "cam1"
        assert event.zone_waypoint == "zona_norte"
        assert event.motion_score == 45.0
        assert event.detection_label == "person"
        assert event.timestamp > 0

    def test_timestamp_auto(self):
        before = time.time()
        event = RtspEvent(
            camera_name="c",
            zone_waypoint="z",
            motion_score=0,
            vlm_description="",
            detection_label="",
            image_data=b"",
        )
        assert event.timestamp >= before


# ---------------------------------------------------------------------------
# cameras_from_config
# ---------------------------------------------------------------------------


class TestCamerasFromConfig:
    def test_empty_config(self):
        cameras = cameras_from_config({})
        assert cameras == []

    def test_empty_list(self):
        cameras = cameras_from_config({"rtsp_cameras": []})
        assert cameras == []

    def test_single_camera(self):
        config = {
            "rtsp_cameras": [
                {
                    "name": "jardin",
                    "url": "rtsp://192.168.1.100:554/stream1",
                    "zone_waypoint": "jardin_centro",
                    "scan_interval_s": 10,
                    "motion_threshold_pct": 15,
                }
            ],
        }
        cameras = cameras_from_config(config)
        assert len(cameras) == 1
        assert cameras[0].name == "jardin"
        assert cameras[0].url == "rtsp://192.168.1.100:554/stream1"
        assert cameras[0].zone_waypoint == "jardin_centro"
        assert cameras[0].scan_interval_s == 10.0
        assert cameras[0].motion_threshold_pct == 15

    def test_multiple_cameras(self):
        config = {
            "rtsp_cameras": [
                {"name": "cam1", "url": "rtsp://1/s"},
                {"name": "cam2", "url": "rtsp://2/s"},
                {"name": "cam3", "url": "rtsp://3/s"},
                {"name": "cam4", "url": "rtsp://4/s"},
            ],
        }
        cameras = cameras_from_config(config)
        assert len(cameras) == 4
        assert cameras[2].name == "cam3"

    def test_defaults_applied(self):
        config = {"rtsp_cameras": [{"name": "x", "url": "rtsp://x/s"}]}
        cams = cameras_from_config(config)
        assert cams[0].scan_interval_s == DEFAULT_SCAN_INTERVAL_S
        assert cams[0].motion_threshold_pct == MOTION_THRESHOLD_PCT
        assert cams[0].enabled is True

    def test_disabled_camera(self):
        config = {
            "rtsp_cameras": [
                {"name": "x", "url": "rtsp://x/s", "enabled": False},
            ]
        }
        cams = cameras_from_config(config)
        assert cams[0].enabled is False

    def test_auto_name_when_missing(self):
        config = {"rtsp_cameras": [{"url": "rtsp://x/s"}]}
        cams = cameras_from_config(config)
        assert cams[0].name == "cam_0"


# ---------------------------------------------------------------------------
# RtspMonitor — sync tests
# ---------------------------------------------------------------------------


class TestRtspMonitor:
    def test_init_no_cameras(self):
        mon = RtspMonitor(cameras=[])
        assert mon.camera_count == 0
        assert not mon.running

    def test_init_with_cameras(self):
        cams = [
            RtspCamera(name="a", url="rtsp://a/s"),
            RtspCamera(name="b", url="rtsp://b/s"),
        ]
        mon = RtspMonitor(cameras=cams)
        assert mon.camera_count == 2

    def test_get_camera_found(self):
        cams = [RtspCamera(name="garden", url="rtsp://g/s")]
        mon = RtspMonitor(cameras=cams)
        assert mon.get_camera("garden") is not None
        assert mon.get_camera("garden").url == "rtsp://g/s"

    def test_get_camera_not_found(self):
        cams = [RtspCamera(name="garden", url="rtsp://g/s")]
        mon = RtspMonitor(cameras=cams)
        assert mon.get_camera("garage") is None

    def test_stats_initialized(self):
        cams = [RtspCamera(name="cam1", url="rtsp://1/s")]
        mon = RtspMonitor(cameras=cams)
        stats = mon.get_stats()
        assert "cam1" in stats
        assert stats["cam1"]["frames"] == 0
        assert stats["cam1"]["motions"] == 0
        assert stats["cam1"]["detections"] == 0
        assert stats["cam1"]["errors"] == 0

    def test_detect_labels_default(self):
        mon = RtspMonitor(cameras=[])
        assert "person" in mon._detect_labels
        assert "vehicle" in mon._detect_labels

    def test_detect_labels_custom(self):
        mon = RtspMonitor(cameras=[], detect_labels=["dog", "cat"])
        assert mon._detect_labels == ["dog", "cat"]


# ---------------------------------------------------------------------------
# RtspMonitor — async tests (VLM analysis)
# ---------------------------------------------------------------------------


class TestRtspMonitorVLM:
    def test_analyze_no_vlm_returns_motion_event(self):
        async def _run():
            cam = RtspCamera(name="test", url="rtsp://x/s", zone_waypoint="zone1")
            mon = RtspMonitor(cameras=[cam], vision=None)
            event = await mon._analyze_with_vlm(cam, b"jpeg", 55.0)
            assert event is not None
            assert event.camera_name == "test"
            assert event.detection_label == "motion"
            assert event.motion_score == 55.0
            assert "no VLM" in event.vlm_description

        asyncio.run(_run())

    def test_analyze_vlm_detects_threat(self):
        async def _run():
            vision = FakeVision(response="A person near the gate")
            cam = RtspCamera(name="gate", url="rtsp://x/s", zone_waypoint="gate")
            mon = RtspMonitor(
                cameras=[cam],
                vision=vision,
                detect_labels=["person", "vehicle"],
            )
            event = await mon._analyze_with_vlm(cam, b"fake_jpeg", 30.0)
            assert event is not None
            assert event.detection_label == "person"
            assert event.vlm_description == "A person near the gate"
            assert len(vision.calls) == 1

        asyncio.run(_run())

    def test_analyze_vlm_clear_returns_none(self):
        async def _run():
            vision = FakeVision(response="Empty garden, no movement")
            cam = RtspCamera(name="garden", url="rtsp://x/s")
            mon = RtspMonitor(
                cameras=[cam],
                vision=vision,
                detect_labels=["person", "vehicle"],
            )
            event = await mon._analyze_with_vlm(cam, b"fake_jpeg", 20.0)
            assert event is None

        asyncio.run(_run())

    def test_analyze_vlm_error_returns_none(self):
        async def _run():
            class FailVision:
                def describe(self, *a, **kw):
                    raise RuntimeError("VLM down")

            cam = RtspCamera(name="cam", url="rtsp://x/s")
            mon = RtspMonitor(cameras=[cam], vision=FailVision())
            event = await mon._analyze_with_vlm(cam, b"jpeg", 50.0)
            assert event is None

        asyncio.run(_run())

    def test_analyze_vlm_multiple_labels(self):
        async def _run():
            vision = FakeVision(response="A vehicle parked by the gate")
            cam = RtspCamera(name="gate", url="rtsp://x/s")
            mon = RtspMonitor(
                cameras=[cam],
                vision=vision,
                detect_labels=["person", "vehicle", "fire"],
            )
            event = await mon._analyze_with_vlm(cam, b"data", 25.0)
            assert event is not None
            assert event.detection_label == "vehicle"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# RtspMonitor — lifecycle
# ---------------------------------------------------------------------------


class TestRtspMonitorLifecycle:
    def test_start_stop_empty(self):
        async def _run():
            mon = RtspMonitor(cameras=[])
            await mon.start()
            assert mon.running
            await mon.stop()
            assert not mon.running

        asyncio.run(_run())

    def test_start_idempotent(self):
        async def _run():
            mon = RtspMonitor(cameras=[])
            await mon.start()
            await mon.start()  # should not error
            assert mon.running
            await mon.stop()

        asyncio.run(_run())

    def test_stop_cancels_tasks(self):
        async def _run():
            cam = RtspCamera(name="cam", url="rtsp://x/s", enabled=True)
            mon = RtspMonitor(cameras=[cam])
            await mon.start()
            assert len(mon._tasks) == 1
            await mon.stop()
            assert len(mon._tasks) == 0

        asyncio.run(_run())

    def test_disabled_camera_not_started(self):
        async def _run():
            cam = RtspCamera(name="cam", url="rtsp://x/s", enabled=False)
            mon = RtspMonitor(cameras=[cam])
            await mon.start()
            assert len(mon._tasks) == 0
            await mon.stop()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# RtspMonitor — threat callback
# ---------------------------------------------------------------------------


class TestRtspMonitorCallback:
    def test_on_threat_called(self):
        async def _run():
            events_received = []

            async def on_threat(event: RtspEvent):
                events_received.append(event)

            cam = RtspCamera(name="cam", url="rtsp://x/s", zone_waypoint="zone1")
            mon = RtspMonitor(
                cameras=[cam],
                vision=None,
                on_threat=on_threat,
            )

            event = await mon._analyze_with_vlm(cam, b"data", 60.0)
            assert event is not None
            await on_threat(event)
            assert len(events_received) == 1
            assert events_received[0].camera_name == "cam"

        asyncio.run(_run())

    def test_on_threat_with_vlm_detection(self):
        async def _run():
            events_received = []

            async def on_threat(event: RtspEvent):
                events_received.append(event)

            vision = FakeVision(response="person walking in driveway")
            cam = RtspCamera(name="driveway", url="rtsp://x/s", zone_waypoint="z")
            mon = RtspMonitor(
                cameras=[cam],
                vision=vision,
                on_threat=on_threat,
                detect_labels=["person"],
            )
            event = await mon._analyze_with_vlm(cam, b"data", 40.0)
            assert event is not None
            assert event.detection_label == "person"
            await on_threat(event)
            assert len(events_received) == 1

        asyncio.run(_run())
