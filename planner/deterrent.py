# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Deterrent system — escalating response on confirmed intrusion.

Escalation sequence:
  T+0s:   LED → RED solid (recording)
  T+1s:   Photo → Telegram notification
  T+2s:   Spotlight ON, strobing
  T+3s:   Siren ON
  T+4s:   Speaker: warning message
  T+5s:   Robot advances slowly toward intruder
  T+10s:  If still present → dog bark audio loop + continuous siren
  T+15s:  Second notification with evidence
  T+30s:  If intruder gone → de-escalate
  T+60s:  Full de-escalation → resume patrol

Hardware tiers:
  Tier 1 (HAVE):  Buzzer + RED LED strobe
  Tier 2 (~3€):   Siren module 12V via MOSFET
  Tier 3 (~4€):   LED 10W COB spotlight via MOSFET
  Tier 4 (~5€):   PAM8403 amplifier + speaker
  Tier 5 (~3€):   Green laser 532nm via MOSFET
"""

import enum
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

from protocol import (
    ConfigCmd,
    CONFIG_CMD,
    LED_RED,
    LED_RED_STROBE,
    LED_YELLOW,
    LED_GREEN,
    BUZZER_CONFIG_KEY,
    BUZZER_BEEP,
    BUZZER_SIREN,
    BUZZER_OFF,
    SIREN_CONFIG_KEY,
    SPOTLIGHT_CONFIG_KEY,
    LASER_CONFIG_KEY,
    SERVO_PAN_KEY,
    SERVO_TILT_KEY,
    SPEAKER_CONFIG_KEY,
    DEVICE_OFF,
    DEVICE_ON,
    SPOTLIGHT_STROBE,
    SPEAKER_STOP,
    SPEAKER_WARNING,
    SPEAKER_DOG_BARK,
)

logger = logging.getLogger("brain.deterrent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESCALATION_STEP_S = 1.0  # seconds between escalation steps
DEESCALATION_TIMEOUT_S = 30.0  # auto de-escalate if no threat
DEESCALATION_CLEAR_COUNT = 3  # VLM must say CLEAR 3 times
SIREN_MAX_DURATION_S = 120.0  # auto-off safety
SPOTLIGHT_STROBE_HZ = 2  # strobe frequency
ADVANCE_SPEED_PCT = 15  # slow approach toward intruder
SECOND_NOTIFY_DELAY_S = 15.0  # delay before second notification

# Servo constants
SERVO_PAN_NEUTRAL_DEG = 90  # forward-facing
SERVO_TILT_NEUTRAL_DEG = 90  # level (remapped from 0 to servo range)
SERVO_TILT_MIN_DEG = 60  # look down (-30° from neutral)
SERVO_TILT_MAX_DEG = 120  # look up (+30° from neutral)


class DeterrentLevel(enum.IntEnum):
    """Escalation levels — each includes all previous levels."""

    NONE = 0
    RECORD = 1  # LED red, start recording
    NOTIFY = 2  # send notification with photo
    SPOTLIGHT = 3  # spotlight strobe ON
    SIREN = 4  # siren ON
    WARNING = 5  # speaker: warning message
    ADVANCE = 6  # robot moves toward intruder
    AGGRESSIVE = 7  # dog bark + continuous siren


@dataclass
class DeterrentState:
    """Current deterrent state."""

    level: DeterrentLevel = DeterrentLevel.NONE
    start_time: float = 0.0
    last_escalation_time: float = 0.0
    clear_count: int = 0
    active: bool = False
    siren_on: bool = False
    spotlight_on: bool = False
    laser_on: bool = False
    speaker_on: bool = False


class DeterrentManager:
    """Manages escalating deterrent response.

    Integrates with LED controller, alert pipeline, and power manager.
    Sends CONFIG_CMD packets for siren, spotlight, laser, speaker, servos.
    """

    def __init__(self, send_packet: Optional[Callable] = None):
        self._send_packet = send_packet
        self.state = DeterrentState()

    @property
    def active(self) -> bool:
        return self.state.active

    @property
    def level(self) -> DeterrentLevel:
        return self.state.level

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    async def start(self, writer=None):
        """Begin escalation sequence from level 0."""
        if self.state.active:
            return  # already active

        now = time.monotonic()
        self.state = DeterrentState(
            level=DeterrentLevel.RECORD,
            start_time=now,
            last_escalation_time=now,
            active=True,
        )
        logger.info("[Deterrent] STARTED — level RECORD")

        # Level 1: LED red
        await self._send_config(BUZZER_CONFIG_KEY, BUZZER_BEEP, writer)

    async def escalate(self, writer=None) -> DeterrentLevel:
        """Escalate to next level. Returns new level.

        Call this periodically (every ESCALATION_STEP_S) while threat persists.
        """
        if not self.state.active:
            return DeterrentLevel.NONE

        current = self.state.level
        if current >= DeterrentLevel.AGGRESSIVE:
            return current  # already at max

        now = time.monotonic()
        elapsed = now - self.state.last_escalation_time
        if elapsed < ESCALATION_STEP_S:
            return current  # too soon

        new_level = DeterrentLevel(current + 1)
        self.state.level = new_level
        self.state.last_escalation_time = now

        await self._apply_level(new_level, writer)
        logger.info("[Deterrent] Escalated to %s", new_level.name)
        return new_level

    async def _apply_level(self, level: DeterrentLevel, writer=None):
        """Apply hardware actions for a given level."""
        if level == DeterrentLevel.NOTIFY:
            # Notification handled externally by alert pipeline
            pass

        elif level == DeterrentLevel.SPOTLIGHT:
            await self._spotlight_on(writer)

        elif level == DeterrentLevel.SIREN:
            await self._siren_on(writer)

        elif level == DeterrentLevel.WARNING:
            await self._speaker_play(SPEAKER_WARNING, writer)

        elif level == DeterrentLevel.ADVANCE:
            # Advance handled externally by patrol controller
            pass

        elif level == DeterrentLevel.AGGRESSIVE:
            await self._speaker_play(SPEAKER_DOG_BARK, writer)
            # Ensure siren still on
            if not self.state.siren_on:
                await self._siren_on(writer)

    # ------------------------------------------------------------------
    # De-escalation
    # ------------------------------------------------------------------

    async def report_clear(self, writer=None) -> bool:
        """Report VLM says CLEAR. Returns True if fully de-escalated."""
        if not self.state.active:
            return False

        self.state.clear_count += 1
        logger.info("[Deterrent] CLEAR %d/%d", self.state.clear_count, DEESCALATION_CLEAR_COUNT)

        if self.state.clear_count >= DEESCALATION_CLEAR_COUNT:
            await self.stand_down(writer)
            return True
        return False

    def report_threat(self):
        """Report VLM confirms threat — reset clear count."""
        self.state.clear_count = 0

    async def check_timeout(self, writer=None) -> bool:
        """Check if deterrent should auto-deescalate. Returns True if stood down."""
        if not self.state.active:
            return False

        elapsed = time.monotonic() - self.state.start_time
        if elapsed >= SIREN_MAX_DURATION_S:
            logger.info("[Deterrent] Safety timeout after %ds — standing down", int(elapsed))
            await self.stand_down(writer)
            return True
        return False

    async def stand_down(self, writer=None):
        """Full de-escalation — turn off all deterrent hardware."""
        if not self.state.active:
            return

        logger.info("[Deterrent] STAND DOWN (was level %s)", self.state.level.name)

        # Turn off all hardware
        await self._siren_off(writer)
        await self._spotlight_off(writer)
        await self._laser_off(writer)
        await self._speaker_stop(writer)
        await self._send_config(BUZZER_CONFIG_KEY, BUZZER_OFF, writer)

        # Reset servos to neutral
        await self._servo_neutral(writer)

        self.state = DeterrentState()  # reset to defaults

    async def silence(self, writer=None):
        """Silence deterrent (siren/speaker off) but keep recording + LED red."""
        if not self.state.active:
            return

        await self._siren_off(writer)
        await self._speaker_stop(writer)
        await self._send_config(BUZZER_CONFIG_KEY, BUZZER_OFF, writer)
        logger.info("[Deterrent] Silenced (still recording)")

    # ------------------------------------------------------------------
    # Turret aiming (pan-tilt servo control)
    # ------------------------------------------------------------------

    async def aim_at(self, pan_deg: int, tilt_deg: int, writer=None):
        """Point turret (spotlight + laser) at target.

        Args:
            pan_deg: horizontal angle 0-180 (90 = forward)
            tilt_deg: vertical angle 60-120 (90 = level)
        """
        pan_deg = max(0, min(180, pan_deg))
        tilt_deg = max(SERVO_TILT_MIN_DEG, min(SERVO_TILT_MAX_DEG, tilt_deg))

        await self._send_config(SERVO_PAN_KEY, pan_deg, writer)
        await self._send_config(SERVO_TILT_KEY, tilt_deg, writer)

    async def aim_from_frame(self, px_x: int, px_y: int, frame_w: int, frame_h: int, writer=None):
        """Convert pixel position to servo angles and aim turret.

        Maps frame coordinates to servo angles:
          px_x=0 → pan=0°, px_x=frame_w → pan=180°
          px_y=0 → tilt=120° (up), px_y=frame_h → tilt=60° (down)
        """
        if frame_w <= 0 or frame_h <= 0:
            return

        pan_deg = int((px_x / frame_w) * 180)
        # Y is inverted: top of frame = look up (higher tilt), bottom = look down
        tilt_range = SERVO_TILT_MAX_DEG - SERVO_TILT_MIN_DEG
        tilt_deg = SERVO_TILT_MAX_DEG - int((px_y / frame_h) * tilt_range)

        await self.aim_at(pan_deg, tilt_deg, writer)

    async def laser_on(self, writer=None):
        """Turn laser on (for aiming)."""
        self.state.laser_on = True
        await self._send_config(LASER_CONFIG_KEY, DEVICE_ON, writer)

    async def laser_off(self, writer=None):
        """Turn laser off."""
        await self._laser_off(writer)

    # ------------------------------------------------------------------
    # Hardware control (internal)
    # ------------------------------------------------------------------

    async def _siren_on(self, writer=None):
        self.state.siren_on = True
        await self._send_config(SIREN_CONFIG_KEY, DEVICE_ON, writer)
        logger.info("[Deterrent] Siren ON")

    async def _siren_off(self, writer=None):
        if self.state.siren_on:
            self.state.siren_on = False
            await self._send_config(SIREN_CONFIG_KEY, DEVICE_OFF, writer)
            logger.info("[Deterrent] Siren OFF")

    async def _spotlight_on(self, writer=None):
        self.state.spotlight_on = True
        await self._send_config(SPOTLIGHT_CONFIG_KEY, SPOTLIGHT_STROBE, writer)
        logger.info("[Deterrent] Spotlight STROBE ON")

    async def _spotlight_off(self, writer=None):
        if self.state.spotlight_on:
            self.state.spotlight_on = False
            await self._send_config(SPOTLIGHT_CONFIG_KEY, DEVICE_OFF, writer)
            logger.info("[Deterrent] Spotlight OFF")

    async def _laser_off(self, writer=None):
        if self.state.laser_on:
            self.state.laser_on = False
            await self._send_config(LASER_CONFIG_KEY, DEVICE_OFF, writer)

    async def _speaker_play(self, audio_id: int, writer=None):
        self.state.speaker_on = True
        await self._send_config(SPEAKER_CONFIG_KEY, audio_id, writer)
        logger.info("[Deterrent] Speaker: audio 0x%02x", audio_id)

    async def _speaker_stop(self, writer=None):
        if self.state.speaker_on:
            self.state.speaker_on = False
            await self._send_config(SPEAKER_CONFIG_KEY, SPEAKER_STOP, writer)

    async def _servo_neutral(self, writer=None):
        await self._send_config(SERVO_PAN_KEY, SERVO_PAN_NEUTRAL_DEG, writer)
        await self._send_config(SERVO_TILT_KEY, SERVO_TILT_NEUTRAL_DEG, writer)

    async def _send_config(self, key: int, value: int, writer=None):
        if writer is not None and self._send_packet is not None:
            cmd = ConfigCmd(config_key=key, value=value)
            await self._send_packet(writer, CONFIG_CMD, cmd.to_bytes())

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.state.active:
            return "DeterrentManager(inactive)"
        hw = []
        if self.state.siren_on:
            hw.append("siren")
        if self.state.spotlight_on:
            hw.append("spotlight")
        if self.state.laser_on:
            hw.append("laser")
        if self.state.speaker_on:
            hw.append("speaker")
        return (
            f"DeterrentManager(level={self.state.level.name}, "
            f"hw=[{', '.join(hw)}], "
            f"clears={self.state.clear_count}/{DEESCALATION_CLEAR_COUNT})"
        )
