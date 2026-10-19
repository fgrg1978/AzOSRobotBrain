# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RTSP camera monitor — watches fixed surveillance cameras for motion.

Captures frames from N RTSP cameras on a configurable interval, runs motion
detection as a cheap pre-filter, and escalates to VLM analysis on motion.
When a threat is confirmed, dispatches the robot to investigate.

Architecture:
    RtspMonitor ──per camera──→ MotionDetector → VLM (if motion) → dispatch

Each camera runs as an independent async task.

Usage:
    monitor = RtspMonitor(config, vision, on_threat_confirmed)
    await monitor.start()   # spawns N tasks
    await monitor.stop()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

from perception.motion_detect import MotionDetector, MOTION_THRESHOLD_PCT

logger = logging.getLogger("brain.rtsp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SCAN_INTERVAL_S = 10.0  # seconds between RTSP frame grabs
RTSP_CAPTURE_TIMEOUT_S = 5.0  # max time to grab one frame
RTSP_RECONNECT_DELAY_S = 10.0  # delay before retrying failed camera
RTSP_MAX_CONSECUTIVE_ERRORS = 5  # stop retrying after this many failures


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class RtspCamera:
    """Configuration for one RTSP camera."""

    name: str
    url: str
    zone_waypoint: str = ""
    scan_interval_s: float = DEFAULT_SCAN_INTERVAL_S
    motion_threshold_pct: int = MOTION_THRESHOLD_PCT
    enabled: bool = True


@dataclass
class RtspEvent:
    """A confirmed detection from an RTSP camera."""

    camera_name: str
    zone_waypoint: str
    motion_score: float
    vlm_description: str
    detection_label: str
    image_data: bytes
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# RtspMonitor
# ---------------------------------------------------------------------------
class RtspMonitor:
    """Monitors N RTSP cameras for motion, escalates to VLM, dispatches robot."""

    def __init__(
        self,
        cameras: list[RtspCamera],
        vision=None,  # VisionPerception (optional, for VLM)
        on_threat: Optional[Callable[[RtspEvent], Awaitable[None]]] = None,
        detect_labels: list[str] | None = None,
    ):
        self._cameras = cameras
        self._vision = vision
        self._on_threat = on_threat
        self._detect_labels = detect_labels or [
            "person",
            "vehicle",
            "fire",
            "smoke",
        ]
        self._detectors: dict[str, MotionDetector] = {}
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._stats: dict[str, dict] = {}

        for cam in cameras:
            self._detectors[cam.name] = MotionDetector(
                threshold_pct=cam.motion_threshold_pct,
            )
            self._stats[cam.name] = {
                "frames": 0,
                "motions": 0,
                "detections": 0,
                "errors": 0,
                "last_motion": 0.0,
                "last_frame": 0.0,
            }

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self):
        """Spawn one monitoring task per camera."""
        if self._running:
            return
        self._running = True
        for cam in self._cameras:
            if cam.enabled:
                task = asyncio.create_task(
                    self._monitor_camera(cam),
                    name=f"rtsp_{cam.name}",
                )
                self._tasks.append(task)
        logger.info("[RTSP] Started monitoring %d cameras", len(self._tasks))

    async def stop(self):
        """Cancel all monitoring tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[RTSP] Stopped monitoring")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    def get_stats(self) -> dict[str, dict]:
        """Return per-camera statistics."""
        return dict(self._stats)

    def get_camera(self, name: str) -> RtspCamera | None:
        """Get camera config by name."""
        for cam in self._cameras:
            if cam.name == name:
                return cam
        return None

    # ── Per-camera monitoring loop ────────────────────────────────────────

    async def _monitor_camera(self, cam: RtspCamera):
        """Main loop for one camera: capture → motion → VLM → dispatch."""
        consecutive_errors = 0
        logger.info("[RTSP] Monitoring camera '%s' every %.1fs", cam.name, cam.scan_interval_s)

        while self._running:
            try:
                # grab frame
                frame = await self._capture_frame(cam.url)
                if frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= RTSP_MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            "[RTSP] Camera '%s': %d consecutive errors, pausing",
                            cam.name,
                            consecutive_errors,
                        )
                        await asyncio.sleep(RTSP_RECONNECT_DELAY_S)
                        consecutive_errors = 0
                    continue

                consecutive_errors = 0
                self._stats[cam.name]["frames"] += 1
                self._stats[cam.name]["last_frame"] = time.time()

                # motion detection (cheap pre-filter)
                detector = self._detectors[cam.name]
                score = detector.feed(frame)

                if score >= cam.motion_threshold_pct:
                    self._stats[cam.name]["motions"] += 1
                    self._stats[cam.name]["last_motion"] = time.time()
                    logger.info(
                        "[RTSP] Motion on '%s': %.1f%% (threshold %d%%)",
                        cam.name,
                        score,
                        cam.motion_threshold_pct,
                    )

                    # VLM analysis
                    event = await self._analyze_with_vlm(cam, frame, score)
                    if event and self._on_threat:
                        self._stats[cam.name]["detections"] += 1
                        await self._on_threat(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats[cam.name]["errors"] += 1
                logger.error("[RTSP] Camera '%s' error: %s", cam.name, e)
                consecutive_errors += 1

            await asyncio.sleep(cam.scan_interval_s)

        logger.info("[RTSP] Camera '%s' monitoring stopped", cam.name)

    # ── Frame capture ─────────────────────────────────────────────────────

    @staticmethod
    async def _capture_frame(url: str) -> bytes | None:
        """Capture a single JPEG frame from an RTSP URL.

        Uses OpenCV (cv2) if available, falls back to ffmpeg subprocess.
        Returns JPEG bytes or None on failure.
        """
        # Try OpenCV first
        try:
            return await _capture_cv2(url)
        except ImportError:
            pass

        # Fallback: ffmpeg subprocess
        try:
            return await _capture_ffmpeg(url)
        except Exception as e:
            logger.debug("[RTSP] Capture failed for %s: %s", url, e)
            return None

    # ── VLM analysis ──────────────────────────────────────────────────────

    async def _analyze_with_vlm(
        self,
        cam: RtspCamera,
        frame: bytes,
        motion_score: float,
    ) -> RtspEvent | None:
        """Run VLM on a frame with motion detected. Returns event if threat found."""
        if not self._vision:
            # No VLM available — report raw motion as event
            return RtspEvent(
                camera_name=cam.name,
                zone_waypoint=cam.zone_waypoint,
                motion_score=motion_score,
                vlm_description="motion detected (no VLM)",
                detection_label="motion",
                image_data=frame,
            )

        try:
            context = f"RTSP camera '{cam.name}', motion score {motion_score:.0f}%"
            description = await asyncio.to_thread(
                self._vision.describe,
                frame,
                context,
            )
            logger.info("[RTSP] VLM@%s: %s", cam.name, description)

            # Check for threat labels
            desc_lower = description.lower()
            for label in self._detect_labels:
                if label.lower() in desc_lower:
                    return RtspEvent(
                        camera_name=cam.name,
                        zone_waypoint=cam.zone_waypoint,
                        motion_score=motion_score,
                        vlm_description=description,
                        detection_label=label,
                        image_data=frame,
                    )

            # VLM says nothing interesting
            return None

        except Exception as e:
            logger.error("[RTSP] VLM error for '%s': %s", cam.name, e)
            return None


# ---------------------------------------------------------------------------
# Frame capture backends
# ---------------------------------------------------------------------------


async def _capture_cv2(url: str) -> bytes | None:
    """Capture using OpenCV (runs in thread to avoid blocking)."""
    import cv2  # import here to fail fast if not installed

    def _grab():
        cap = cv2.VideoCapture(url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        finally:
            cap.release()

    return await asyncio.wait_for(
        asyncio.to_thread(_grab),
        timeout=RTSP_CAPTURE_TIMEOUT_S,
    )


async def _capture_ffmpeg(url: str) -> bytes | None:
    """Capture a single frame using ffmpeg subprocess."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "5",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=RTSP_CAPTURE_TIMEOUT_S,
        )
        if proc.returncode == 0 and stdout:
            return stdout
        return None
    except asyncio.TimeoutError:
        proc.kill()
        return None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def cameras_from_config(config: dict) -> list[RtspCamera]:
    """Parse rtsp_cameras section from config.yaml into RtspCamera list."""
    raw = config.get("rtsp_cameras") or []
    cameras = []
    for entry in raw:
        cameras.append(
            RtspCamera(
                name=entry.get("name", f"cam_{len(cameras)}"),
                url=entry.get("url", ""),
                zone_waypoint=entry.get("zone_waypoint", ""),
                scan_interval_s=float(entry.get("scan_interval_s", DEFAULT_SCAN_INTERVAL_S)),
                motion_threshold_pct=int(entry.get("motion_threshold_pct", MOTION_THRESHOLD_PCT)),
                enabled=entry.get("enabled", True),
            )
        )
    return cameras
