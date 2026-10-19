# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Alert pipeline — manages detection response: buzzer, evidence, notifications.

Flow:
  Sensor trigger → VLM confirm → AlertPipeline.raise_alert() →
    1. Buzzer beep on robot
    2. Save evidence frames to data/evidence/{timestamp}/
    3. Telegram/Pushover notification with image
    4. Cooldown to prevent alert spam
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

import protocol
from protocol import (
    ConfigCmd,
    CONFIG_CMD,
    BUZZER_CONFIG_KEY,
    BUZZER_OFF,
    BUZZER_BEEP,
    BUZZER_SIREN,
    BUZZER_CHIRP,
)

logger = logging.getLogger("brain.alert")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_S = 30  # don't repeat alerts for same zone within this window
EVIDENCE_FRAMES = 10  # frames to save per detection event
EVIDENCE_RETENTION_DAYS = 30  # cleanup old evidence after this many days
EVIDENCE_DIR = "data/evidence"  # base directory for evidence storage
EVIDENCE_CAPTURE_INTERVAL_S = 0.5  # interval between evidence frames


@dataclass
class AlertEvent:
    """A single alert event with metadata."""

    alert_id: str  # unique ID (timestamp-based)
    trigger_label: str  # what triggered it (pir_motion, sound_event, etc.)
    vlm_description: str  # VLM scene description
    detection_label: str  # what was detected (person, fire, etc.)
    timestamp: float  # monotonic time of alert
    wall_time: str  # ISO format wall clock time
    evidence_dir: str = ""  # path to evidence directory
    frames_saved: int = 0  # number of evidence frames saved
    notified: bool = False  # whether notifications were sent


class AlertPipeline:
    """Manages the alert response sequence: buzzer, evidence, notifications."""

    def __init__(
        self,
        send_packet: Optional[Callable] = None,
        notifier=None,
        evidence_dir: str = EVIDENCE_DIR,
        cooldown_s: float = ALERT_COOLDOWN_S,
        evidence_frames: int = EVIDENCE_FRAMES,
    ):
        self._send_packet = send_packet
        self._notifier = notifier
        self._evidence_dir = evidence_dir
        self._cooldown_s = cooldown_s
        self._evidence_frames = evidence_frames

        self._last_alert_time: dict[str, float] = {}  # zone/label → time
        self._alerts: list[AlertEvent] = []
        self._active_evidence: Optional[AlertEvent] = None

    @property
    def alerts(self) -> list[AlertEvent]:
        """All alert events (most recent last)."""
        return list(self._alerts)

    @property
    def alert_count(self) -> int:
        return len(self._alerts)

    @property
    def active_evidence(self) -> Optional[AlertEvent]:
        return self._active_evidence

    def is_cooled_down(self, label: str) -> bool:
        """Check if enough time has passed since last alert for this label."""
        last = self._last_alert_time.get(label, 0.0)
        return (time.monotonic() - last) >= self._cooldown_s

    async def raise_alert(
        self,
        trigger_label: str,
        detection_label: str,
        vlm_description: str,
        image_data: Optional[bytes] = None,
        writer=None,
        actions: Optional[list[str]] = None,
    ) -> Optional[AlertEvent]:
        """Execute the full alert sequence.

        Args:
            trigger_label: sensor that triggered (pir_motion, etc.)
            detection_label: what VLM detected (person, fire, etc.)
            vlm_description: VLM scene description
            image_data: camera frame (JPEG) for evidence + notification
            writer: asyncio.StreamWriter to send buzzer command
            actions: list of actions from mode config (notify, alert, etc.)

        Returns:
            AlertEvent if alert was raised, None if cooled down.
        """
        if actions is None:
            actions = []

        # Check cooldown
        cooldown_key = f"{trigger_label}:{detection_label}"
        if not self.is_cooled_down(cooldown_key):
            logger.info("[Alert] Cooldown active for %s, skipping", cooldown_key)
            return None

        # Create alert event
        now = time.monotonic()
        wall_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        alert_id = time.strftime("%Y%m%d_%H%M%S")
        event = AlertEvent(
            alert_id=alert_id,
            trigger_label=trigger_label,
            vlm_description=vlm_description,
            detection_label=detection_label,
            timestamp=now,
            wall_time=wall_time,
        )

        self._last_alert_time[cooldown_key] = now
        self._alerts.append(event)

        logger.info(
            "[Alert] RAISED: %s detected by %s — %s",
            detection_label,
            trigger_label,
            vlm_description,
        )

        # 1. Buzzer beep
        if "alert" in actions or "buzzer_alert" in actions:
            await self._buzzer(BUZZER_BEEP, writer)

        # 2. Save evidence (first frame)
        if image_data:
            self._save_evidence_frame(event, image_data)
            self._active_evidence = event

        # 3. Send notification
        if "notify" in actions or "notify_telegram_photo" in actions:
            await self._notify(event, image_data)

        return event

    async def save_evidence_frame(self, image_data: bytes):
        """Save additional evidence frame for the active alert event."""
        if self._active_evidence is None:
            return
        if self._active_evidence.frames_saved >= self._evidence_frames:
            return
        self._save_evidence_frame(self._active_evidence, image_data)

    def finish_evidence(self):
        """Stop collecting evidence for the current alert."""
        if self._active_evidence:
            logger.info(
                "[Alert] Evidence collection done: %d frames in %s",
                self._active_evidence.frames_saved,
                self._active_evidence.evidence_dir,
            )
            self._active_evidence = None

    # ------------------------------------------------------------------
    # Buzzer control
    # ------------------------------------------------------------------

    async def _buzzer(self, pattern: int, writer=None):
        """Send buzzer command to robot."""
        if writer is not None and self._send_packet is not None:
            cmd = ConfigCmd.buzzer(pattern)
            await self._send_packet(writer, CONFIG_CMD, cmd.to_bytes())
            logger.info("[Alert] Buzzer: 0x%02x", pattern)

    async def buzzer_on(self, pattern: int = BUZZER_BEEP, writer=None):
        """Public method to activate buzzer."""
        await self._buzzer(pattern, writer)

    async def buzzer_off(self, writer=None):
        """Turn buzzer off."""
        await self._buzzer(BUZZER_OFF, writer)

    # ------------------------------------------------------------------
    # Evidence storage
    # ------------------------------------------------------------------

    def _save_evidence_frame(self, event: AlertEvent, image_data: bytes):
        """Save a single evidence frame to disk."""
        if not event.evidence_dir:
            event.evidence_dir = os.path.join(self._evidence_dir, event.alert_id)

        os.makedirs(event.evidence_dir, exist_ok=True)

        frame_path = os.path.join(
            event.evidence_dir,
            f"frame_{event.frames_saved:03d}.jpg",
        )
        try:
            with open(frame_path, "wb") as f:
                f.write(image_data)
            event.frames_saved += 1

            # Save metadata on first frame
            if event.frames_saved == 1:
                self._save_metadata(event)

            logger.debug("[Alert] Saved evidence frame %d: %s", event.frames_saved, frame_path)
        except OSError as e:
            logger.error("[Alert] Failed to save evidence: %s", e)

    def _save_metadata(self, event: AlertEvent):
        """Save alert metadata JSON alongside evidence frames."""
        meta_path = os.path.join(event.evidence_dir, "metadata.json")
        meta = {
            "alert_id": event.alert_id,
            "trigger": event.trigger_label,
            "detection": event.detection_label,
            "vlm_description": event.vlm_description,
            "wall_time": event.wall_time,
        }
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            logger.error("[Alert] Failed to save metadata: %s", e)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def _notify(self, event: AlertEvent, image_data: Optional[bytes]):
        """Send notification via all enabled backends."""
        if self._notifier is None:
            return

        message = (
            f"[{event.detection_label.upper()}] {event.vlm_description}\n"
            f"Trigger: {event.trigger_label}\n"
            f"Time: {event.wall_time}"
        )
        try:
            results = await self._notifier.alert(
                message,
                title=f"Detection: {event.detection_label}",
                image=image_data,
            )
            event.notified = True
            logger.info("[Alert] Notifications sent: %s", results)
        except Exception as e:
            logger.error("[Alert] Notification error: %s", e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_evidence(self, retention_days: int = EVIDENCE_RETENTION_DAYS):
        """Remove evidence directories older than retention_days."""
        if not os.path.exists(self._evidence_dir):
            return 0

        cutoff = time.time() - (retention_days * 86400)  # seconds per day
        removed = 0

        try:
            for entry in os.listdir(self._evidence_dir):
                entry_path = os.path.join(self._evidence_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                mtime = os.path.getmtime(entry_path)
                if mtime < cutoff:
                    import shutil

                    shutil.rmtree(entry_path)
                    removed += 1
                    logger.info("[Alert] Cleaned old evidence: %s", entry)
        except OSError as e:
            logger.error("[Alert] Cleanup error: %s", e)

        return removed

    def __repr__(self) -> str:
        active = "recording" if self._active_evidence else "idle"
        return f"AlertPipeline(alerts={self.alert_count}, " f"evidence={active})"
