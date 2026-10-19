# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Power mode manager — ECO/ALERT modes to maximize battery life.

ECO mode: camera OFF, LiDAR reduced, slow patrol, passive sensors only.
ALERT mode: camera ON, full LiDAR, VLM active, fast response.

Transition: ECO --trigger--> ALERT --clear×N--> ECO
"""

import enum
import logging
import time
from dataclasses import dataclass
from typing import Optional

import protocol
from protocol import ConfigCmd, CONFIG_CMD

logger = logging.getLogger("brain.power")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATROL_SPEED_ECO_PCT = 25  # slower = less motor current
PATROL_SPEED_ALERT_PCT = 50  # normal speed in alert
LIDAR_HZ_ECO = 5  # reduced scan rate
LIDAR_HZ_ALERT = 10  # full scan rate
CAMERA_WARMUP_MS = 200  # CSI re-init time after power-on
WIFI_BATCH_INTERVAL_S = 2  # ESP32 batch send in ECO
ALERT_TIMEOUT_S = 120  # auto-return to ECO after no threat
ALERT_CLEAR_COUNT = 3  # VLM must say CLEAR this many times
LED_ECO_DUTY_PCT = 10  # dim green in ECO
LED_ALERT_DUTY_PCT = 100  # full brightness in ALERT

# CONFIG_CMD keys for power-related commands
POWER_CONFIG_KEY = 0x11  # config key for power mode
CAMERA_POWER_KEY = 0x12  # config key for camera GPIO
WIFI_MODE_KEY = 0x13  # config key for ESP32 sleep mode
LIDAR_HZ_KEY = 0x14  # config key for LiDAR scan rate

# Power mode values
POWER_ECO = 0x00
POWER_ALERT = 0x01

# Camera power values
CAMERA_OFF = 0x00
CAMERA_ON = 0x01

# WiFi mode values
WIFI_BATCH = 0x00  # ESP32 light sleep between sends
WIFI_CONTINUOUS = 0x01  # always awake


class PowerMode(enum.Enum):
    ECO = "eco"
    ALERT = "alert"


@dataclass
class PowerState:
    """Current power mode state with transition tracking."""

    mode: PowerMode = PowerMode.ECO
    alert_trigger: str = ""
    alert_start_time: float = 0.0
    clear_count: int = 0  # consecutive VLM CLEAR results
    camera_on: bool = False


class PowerManager:
    """Manages ECO/ALERT power mode transitions."""

    def __init__(self, send_packet=None):
        """Args:
        send_packet: async callable(writer, pkt_type, payload) for sending config
        """
        self._send_packet = send_packet
        self.state = PowerState()

    @property
    def mode(self) -> PowerMode:
        return self.state.mode

    @property
    def is_eco(self) -> bool:
        return self.state.mode == PowerMode.ECO

    @property
    def is_alert(self) -> bool:
        return self.state.mode == PowerMode.ALERT

    @property
    def patrol_speed(self) -> int:
        """Return appropriate patrol speed for current mode."""
        return PATROL_SPEED_ALERT_PCT if self.is_alert else PATROL_SPEED_ECO_PCT

    @property
    def lidar_hz(self) -> int:
        """Return appropriate LiDAR rate for current mode."""
        return LIDAR_HZ_ALERT if self.is_alert else LIDAR_HZ_ECO

    @property
    def vlm_active(self) -> bool:
        """Whether VLM/LLM should process camera frames."""
        return self.is_alert

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    async def trigger_alert(self, reason: str, writer=None):
        """Transition ECO → ALERT. No-op if already in ALERT."""
        if self.is_alert:
            return

        logger.info("[Power] ECO → ALERT: %s", reason)
        self.state.mode = PowerMode.ALERT
        self.state.alert_trigger = reason
        self.state.alert_start_time = time.monotonic()
        self.state.clear_count = 0

        # Enable camera
        await self._set_camera(True, writer)
        # Switch WiFi to continuous
        await self._set_wifi_mode(WIFI_CONTINUOUS, writer)
        # Increase LiDAR rate
        await self._set_lidar_hz(LIDAR_HZ_ALERT, writer)
        # Send power mode to robot
        await self._send_config(POWER_CONFIG_KEY, POWER_ALERT, writer)

    async def report_clear(self, writer=None):
        """Report VLM says CLEAR. After N consecutive clears, de-escalate to ECO."""
        if not self.is_alert:
            return

        self.state.clear_count += 1
        logger.info("[Power] VLM CLEAR (%d/%d)", self.state.clear_count, ALERT_CLEAR_COUNT)

        if self.state.clear_count >= ALERT_CLEAR_COUNT:
            await self._deescalate(writer)

    async def report_threat(self, writer=None):
        """Report VLM confirms threat. Reset clear count."""
        self.state.clear_count = 0
        if not self.is_alert:
            await self.trigger_alert("vlm_threat", writer)

    async def check_timeout(self, writer=None) -> bool:
        """Check if ALERT has timed out. Returns True if de-escalated."""
        if not self.is_alert:
            return False

        elapsed = time.monotonic() - self.state.alert_start_time
        if elapsed >= ALERT_TIMEOUT_S:
            logger.info("[Power] ALERT timeout after %ds", int(elapsed))
            await self._deescalate(writer)
            return True
        return False

    async def force_eco(self, writer=None):
        """Force transition to ECO mode."""
        if self.is_eco:
            return
        await self._deescalate(writer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _deescalate(self, writer=None):
        """Transition ALERT → ECO."""
        logger.info("[Power] ALERT → ECO (was: %s)", self.state.alert_trigger)
        self.state.mode = PowerMode.ECO
        self.state.alert_trigger = ""
        self.state.clear_count = 0

        # Disable camera
        await self._set_camera(False, writer)
        # Switch WiFi to batch
        await self._set_wifi_mode(WIFI_BATCH, writer)
        # Reduce LiDAR rate
        await self._set_lidar_hz(LIDAR_HZ_ECO, writer)
        # Send power mode to robot
        await self._send_config(POWER_CONFIG_KEY, POWER_ECO, writer)

    async def _set_camera(self, on: bool, writer=None):
        """Control camera power GPIO via CONFIG_CMD."""
        self.state.camera_on = on
        value = CAMERA_ON if on else CAMERA_OFF
        await self._send_config(CAMERA_POWER_KEY, value, writer)
        if on:
            logger.info("[Power] Camera ON (warm-up %dms)", CAMERA_WARMUP_MS)
        else:
            logger.info("[Power] Camera OFF")

    async def _set_wifi_mode(self, mode: int, writer=None):
        await self._send_config(WIFI_MODE_KEY, mode, writer)

    async def _set_lidar_hz(self, hz: int, writer=None):
        await self._send_config(LIDAR_HZ_KEY, hz, writer)

    async def _send_config(self, key: int, value: int, writer=None):
        if writer is not None and self._send_packet is not None:
            cmd = ConfigCmd(config_key=key, value=value)
            await self._send_packet(writer, CONFIG_CMD, cmd.to_bytes())

    def __repr__(self) -> str:
        return (
            f"PowerManager(mode={self.state.mode.value}, "
            f"camera={'ON' if self.state.camera_on else 'OFF'}, "
            f"clears={self.state.clear_count}/{ALERT_CLEAR_COUNT})"
        )
