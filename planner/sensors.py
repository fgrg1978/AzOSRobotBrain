# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Multi-sensor fusion — PIR, sound, IR trigger detection.

Processes digital sensor flags from the wheeled sensor packet.
Any trigger → escalate to ALERT mode for VLM confirmation.

Sensors:
  PIR   — passive infrared, detects motion (warm bodies)
  Sound — digital out, detects glass break / loud impact / scream
  IR    — close proximity, backup obstacle detection
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from protocol import SENSOR_FLAG_PIR, SENSOR_FLAG_SOUND, SENSOR_FLAG_IR

logger = logging.getLogger("brain.sensors")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENSOR_DEBOUNCE_S = 1.0  # ignore repeat triggers within this window
SENSOR_TRIGGER_COOLDOWN_S = 30.0  # cooldown per sensor type after alert
PIR_LABEL = "pir_motion"
SOUND_LABEL = "sound_event"
IR_LABEL = "ir_proximity"

# Sensor flag to label mapping
SENSOR_FLAG_MAP = {
    SENSOR_FLAG_PIR: PIR_LABEL,
    SENSOR_FLAG_SOUND: SOUND_LABEL,
    SENSOR_FLAG_IR: IR_LABEL,
}


@dataclass
class SensorTrigger:
    """Represents a single sensor trigger event."""

    label: str
    timestamp: float
    flag: int


@dataclass
class SensorState:
    """Tracks per-sensor trigger timing for debounce and cooldown."""

    last_trigger_time: dict[int, float] = field(default_factory=dict)
    last_alert_time: dict[int, float] = field(default_factory=dict)


class SensorFusion:
    """Processes digital sensor flags and produces trigger events.

    Applies debounce (ignore rapid re-triggers) and cooldown
    (don't re-alert for same sensor type within cooldown window).
    """

    def __init__(
        self, debounce_s: float = SENSOR_DEBOUNCE_S, cooldown_s: float = SENSOR_TRIGGER_COOLDOWN_S
    ):
        self._debounce_s = debounce_s
        self._cooldown_s = cooldown_s
        self._state = SensorState()
        self._enabled_flags = SENSOR_FLAG_PIR | SENSOR_FLAG_SOUND | SENSOR_FLAG_IR

    @property
    def enabled_flags(self) -> int:
        return self._enabled_flags

    def set_enabled(self, pir: bool = True, sound: bool = True, ir: bool = True):
        """Enable/disable individual sensor types."""
        self._enabled_flags = 0
        if pir:
            self._enabled_flags |= SENSOR_FLAG_PIR
        if sound:
            self._enabled_flags |= SENSOR_FLAG_SOUND
        if ir:
            self._enabled_flags |= SENSOR_FLAG_IR

    def process_flags(self, sensor_flags: int) -> list[SensorTrigger]:
        """Process sensor_flags from a SensorPacket.

        Returns list of new trigger events (after debounce + cooldown filtering).
        Empty list = no new triggers.
        """
        if sensor_flags == 0:
            return []

        now = time.monotonic()
        triggers = []

        for flag, label in SENSOR_FLAG_MAP.items():
            if not (sensor_flags & flag):
                continue
            if not (self._enabled_flags & flag):
                continue

            # debounce: ignore if last trigger was too recent
            last = self._state.last_trigger_time.get(flag, 0.0)
            if now - last < self._debounce_s:
                continue

            # cooldown: ignore if we already alerted for this sensor recently
            last_alert = self._state.last_alert_time.get(flag, 0.0)
            if now - last_alert < self._cooldown_s:
                continue

            self._state.last_trigger_time[flag] = now
            triggers.append(SensorTrigger(label=label, timestamp=now, flag=flag))
            logger.info("[Sensor] Trigger: %s (flag=0x%04x)", label, flag)

        return triggers

    def mark_alerted(self, trigger: SensorTrigger):
        """Mark a sensor as having generated an alert (starts cooldown)."""
        self._state.last_alert_time[trigger.flag] = trigger.timestamp

    def mark_all_alerted(self, triggers: list[SensorTrigger]):
        """Mark all triggers as alerted."""
        for t in triggers:
            self.mark_alerted(t)

    def reset(self):
        """Reset all debounce and cooldown state."""
        self._state = SensorState()

    def __repr__(self) -> str:
        enabled = []
        if self._enabled_flags & SENSOR_FLAG_PIR:
            enabled.append("PIR")
        if self._enabled_flags & SENSOR_FLAG_SOUND:
            enabled.append("Sound")
        if self._enabled_flags & SENSOR_FLAG_IR:
            enabled.append("IR")
        return f"SensorFusion(enabled=[{', '.join(enabled)}])"
