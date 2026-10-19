# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/led.py — LED controller and protocol ConfigCmd."""

import asyncio
import pytest

from protocol import (
    ConfigCmd,
    CONFIG_CMD,
    LED_CONFIG_KEY,
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
from planner.led import LedController, LED_STATE_MAP

# ── ConfigCmd ────────────────────────────────────────────────────────────────


class TestConfigCmd:

    def test_led_factory(self):
        cmd = ConfigCmd.led(LED_GREEN)
        assert cmd.config_key == LED_CONFIG_KEY
        assert cmd.value == LED_GREEN

    def test_roundtrip(self):
        cmd = ConfigCmd.led(LED_RED_STROBE)
        data = cmd.to_bytes()
        assert len(data) == 4
        cmd2 = ConfigCmd.from_bytes(data)
        assert cmd2.config_key == LED_CONFIG_KEY
        assert cmd2.value == LED_RED_STROBE

    def test_all_led_codes_unique(self):
        codes = [
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
        ]
        assert len(codes) == len(set(codes))

    def test_reserved_default_zero(self):
        cmd = ConfigCmd.led(LED_GREEN)
        assert cmd.reserved == 0


# ── LedController ────────────────────────────────────────────────────────────


class TestLedController:

    def _make_controller(self):
        packets_sent = []

        async def mock_send(writer, pkt_type, payload):
            packets_sent.append((pkt_type, payload))

        ctrl = LedController(mock_send)
        return ctrl, packets_sent

    def test_initial_state_off(self):
        ctrl, _ = self._make_controller()
        assert ctrl.current_state == "off"
        assert ctrl.current_code == LED_OFF

    def test_set_state_sends_packet(self):
        ctrl, packets = self._make_controller()
        writer = object()  # dummy

        async def run():
            await ctrl.set_state("monitoring", writer)

        asyncio.run(run())
        assert ctrl.current_state == "monitoring"
        assert ctrl.current_code == LED_GREEN
        assert len(packets) == 1
        assert packets[0][0] == CONFIG_CMD

    def test_set_state_no_duplicate(self):
        ctrl, packets = self._make_controller()
        writer = object()

        async def run():
            await ctrl.set_state("monitoring", writer)
            await ctrl.set_state("monitoring", writer)  # same state

        asyncio.run(run())
        assert len(packets) == 1  # only sent once

    def test_set_state_changes(self):
        ctrl, packets = self._make_controller()
        writer = object()

        async def run():
            await ctrl.set_state("monitoring", writer)
            await ctrl.set_state("detecting", writer)
            await ctrl.set_state("confirmed", writer)

        asyncio.run(run())
        assert len(packets) == 3
        assert ctrl.current_state == "confirmed"
        assert ctrl.current_code == LED_RED

    def test_unknown_state_defaults_off(self):
        ctrl, packets = self._make_controller()
        writer = object()

        async def run():
            await ctrl.set_state("nonexistent", writer)

        asyncio.run(run())
        assert ctrl.current_code == LED_OFF

    def test_no_writer_no_send(self):
        ctrl, packets = self._make_controller()

        async def run():
            await ctrl.set_state("monitoring")  # no writer

        asyncio.run(run())
        assert ctrl.current_state == "monitoring"
        assert len(packets) == 0  # nothing sent

    def test_flash(self):
        ctrl, packets = self._make_controller()
        writer = object()

        async def run():
            await ctrl.set_state("monitoring", writer)
            await ctrl.flash("photo", writer)

        asyncio.run(run())
        assert len(packets) == 2
        # current state should still be monitoring (flash doesn't change it)
        assert ctrl.current_state == "monitoring"

    def test_repr(self):
        ctrl, _ = self._make_controller()
        r = repr(ctrl)
        assert "LedController" in r
        assert "off" in r

    def test_all_states_in_map(self):
        expected_states = [
            "off",
            "monitoring",
            "mapping",
            "detecting",
            "investigating",
            "confirmed",
            "tracking",
            "panic",
            "returning",
            "low_battery",
            "photo",
        ]
        for state in expected_states:
            assert state in LED_STATE_MAP, f"Missing state: {state}"
