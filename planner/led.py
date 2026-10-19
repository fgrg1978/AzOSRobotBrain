# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Status LED controller — maps robot state to LED color/pattern.

Sends ConfigCmd(LED) to the robot whenever operational state changes.
The kernel driver (status_led.rs) handles GPIO + blink patterns.
"""

import logging
from typing import Callable, Awaitable, Optional

import protocol
from protocol import (
    ConfigCmd,
    CONFIG_CMD,
    LED_OFF,
    LED_GREEN,
    LED_GREEN_BLINK,
    LED_YELLOW,
    LED_YELLOW_BLINK,
    LED_RED,
    LED_RED_BLINK,
    LED_RED_STROBE,
    LED_BLUE,
    LED_BLUE_BLINK,
    LED_WHITE_FLASH,
)

logger = logging.getLogger("brain.led")

# Map of state names to LED codes — single source of truth
LED_STATE_MAP = {
    "off": LED_OFF,
    "monitoring": LED_GREEN,
    "mapping": LED_GREEN_BLINK,
    "detecting": LED_YELLOW,
    "investigating": LED_YELLOW_BLINK,
    "confirmed": LED_RED,
    "tracking": LED_RED_BLINK,
    "panic": LED_RED_STROBE,
    "returning": LED_BLUE,
    "low_battery": LED_BLUE_BLINK,
    "photo": LED_WHITE_FLASH,
}


class LedController:
    """Manages LED state and sends updates to robot."""

    def __init__(self, send_packet: Callable):
        """Args:
        send_packet: async callable(writer, pkt_type, payload)
        """
        self._send_packet = send_packet
        self._current_state: str = "off"
        self._current_code: int = LED_OFF

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def current_code(self) -> int:
        return self._current_code

    async def set_state(self, state: str, writer=None):
        """Set LED state by name. Only sends if state changed.

        Args:
            state: one of LED_STATE_MAP keys
            writer: asyncio.StreamWriter to send to robot
        """
        code = LED_STATE_MAP.get(state, LED_OFF)
        if code == self._current_code:
            return  # no change

        self._current_state = state
        self._current_code = code

        if writer is not None:
            cmd = ConfigCmd.led(code)
            await self._send_packet(writer, CONFIG_CMD, cmd.to_bytes())
            logger.info("[LED] %s (0x%02x)", state, code)

    async def flash(self, state: str, writer=None):
        """Send a one-shot LED state (e.g. WHITE_FLASH for photo).

        Does not change the persistent state — caller should restore after.
        """
        code = LED_STATE_MAP.get(state, LED_OFF)
        if writer is not None:
            cmd = ConfigCmd.led(code)
            await self._send_packet(writer, CONFIG_CMD, cmd.to_bytes())

    def __repr__(self) -> str:
        return f"LedController(state={self._current_state!r}, code=0x{self._current_code:02x})"
