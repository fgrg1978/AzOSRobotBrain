# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/deterrent.py — deterrent escalation system."""

import asyncio
import time
import pytest

from planner.deterrent import (
    DeterrentManager,
    DeterrentLevel,
    DeterrentState,
    ESCALATION_STEP_S,
    DEESCALATION_CLEAR_COUNT,
    SIREN_MAX_DURATION_S,
    ADVANCE_SPEED_PCT,
    SERVO_PAN_NEUTRAL_DEG,
    SERVO_TILT_NEUTRAL_DEG,
    SERVO_TILT_MIN_DEG,
    SERVO_TILT_MAX_DEG,
)
from protocol import (
    ConfigCmd,
    CONFIG_CMD,
    SIREN_CONFIG_KEY,
    SPOTLIGHT_CONFIG_KEY,
    LASER_CONFIG_KEY,
    SERVO_PAN_KEY,
    SERVO_TILT_KEY,
    SPEAKER_CONFIG_KEY,
    BUZZER_CONFIG_KEY,
    DEVICE_OFF,
    DEVICE_ON,
    SPOTLIGHT_STROBE,
    SPEAKER_STOP,
    SPEAKER_WARNING,
    SPEAKER_DOG_BARK,
    BUZZER_BEEP,
    BUZZER_OFF,
)


class TestDeterrentState:

    def test_defaults(self):
        s = DeterrentState()
        assert s.level == DeterrentLevel.NONE
        assert s.active is False
        assert s.siren_on is False
        assert s.spotlight_on is False
        assert s.laser_on is False
        assert s.clear_count == 0


class TestDeterrentManager:

    def _make_manager(self):
        configs_sent = []

        async def mock_send(writer, pkt_type, payload):
            cmd = ConfigCmd.from_bytes(payload)
            configs_sent.append((cmd.config_key, cmd.value))

        mgr = DeterrentManager(mock_send)
        return mgr, configs_sent

    def test_initial_inactive(self):
        mgr, _ = self._make_manager()
        assert not mgr.active
        assert mgr.level == DeterrentLevel.NONE

    def test_start_activates(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))

        assert mgr.active
        assert mgr.level == DeterrentLevel.RECORD
        # Should have sent buzzer beep
        buzzer_sent = [c for c in configs if c[0] == BUZZER_CONFIG_KEY]
        assert len(buzzer_sent) == 1
        assert buzzer_sent[0][1] == BUZZER_BEEP

    def test_start_idempotent(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        count_after_first = len(configs)
        asyncio.run(mgr.start(writer))
        assert len(configs) == count_after_first

    def test_escalate_increments_level(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Force past escalation timing
        mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1

        new_level = asyncio.run(mgr.escalate(writer))
        assert new_level == DeterrentLevel.NOTIFY

    def test_escalate_too_soon_stays(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Don't force past timing
        new_level = asyncio.run(mgr.escalate(writer))
        assert new_level == DeterrentLevel.RECORD  # no change

    def test_escalate_spotlight_sends_strobe(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Escalate to SPOTLIGHT (level 3)
        for _ in range(2):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        assert mgr.level == DeterrentLevel.SPOTLIGHT
        spotlight_sent = [c for c in configs if c[0] == SPOTLIGHT_CONFIG_KEY]
        assert len(spotlight_sent) == 1
        assert spotlight_sent[0][1] == SPOTLIGHT_STROBE

    def test_escalate_siren_sends_on(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        for _ in range(3):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        assert mgr.level == DeterrentLevel.SIREN
        siren_sent = [c for c in configs if c[0] == SIREN_CONFIG_KEY]
        assert len(siren_sent) == 1
        assert siren_sent[0][1] == DEVICE_ON

    def test_escalate_warning_sends_speaker(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        for _ in range(4):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        assert mgr.level == DeterrentLevel.WARNING
        speaker_sent = [c for c in configs if c[0] == SPEAKER_CONFIG_KEY]
        assert len(speaker_sent) == 1
        assert speaker_sent[0][1] == SPEAKER_WARNING

    def test_escalate_max_level_stays(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        for _ in range(DeterrentLevel.AGGRESSIVE):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        assert mgr.level == DeterrentLevel.AGGRESSIVE

        # Further escalation stays at max
        mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
        level = asyncio.run(mgr.escalate(writer))
        assert level == DeterrentLevel.AGGRESSIVE

    def test_escalate_when_inactive_returns_none(self):
        mgr, _ = self._make_manager()
        writer = object()
        level = asyncio.run(mgr.escalate(writer))
        assert level == DeterrentLevel.NONE

    def test_stand_down_deactivates(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Escalate to siren
        for _ in range(3):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        configs.clear()
        asyncio.run(mgr.stand_down(writer))

        assert not mgr.active
        assert mgr.level == DeterrentLevel.NONE

        # Should have sent OFF for siren, spotlight, laser, speaker, buzzer, servos
        keys_sent = [c[0] for c in configs]
        assert SIREN_CONFIG_KEY in keys_sent
        assert BUZZER_CONFIG_KEY in keys_sent
        # Servos to neutral
        servo_pan = [c for c in configs if c[0] == SERVO_PAN_KEY]
        assert len(servo_pan) >= 1
        assert servo_pan[0][1] == SERVO_PAN_NEUTRAL_DEG

    def test_stand_down_noop_when_inactive(self):
        mgr, configs = self._make_manager()
        asyncio.run(mgr.stand_down())
        assert len(configs) == 0

    def test_report_clear_deescalates(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))

        for i in range(DEESCALATION_CLEAR_COUNT):
            result = asyncio.run(mgr.report_clear(writer))
            if i < DEESCALATION_CLEAR_COUNT - 1:
                assert result is False
            else:
                assert result is True

        assert not mgr.active

    def test_report_clear_partial(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        asyncio.run(mgr.report_clear(writer))
        assert mgr.active
        assert mgr.state.clear_count == 1

    def test_report_threat_resets_clears(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        asyncio.run(mgr.report_clear(writer))
        asyncio.run(mgr.report_clear(writer))
        assert mgr.state.clear_count == 2

        mgr.report_threat()
        assert mgr.state.clear_count == 0
        assert mgr.active

    def test_report_clear_inactive_returns_false(self):
        mgr, _ = self._make_manager()
        result = asyncio.run(mgr.report_clear())
        assert result is False

    def test_silence_stops_audio(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Escalate to siren
        for _ in range(3):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        configs.clear()
        asyncio.run(mgr.silence(writer))

        # Siren and buzzer should be OFF
        siren_sent = [c for c in configs if c[0] == SIREN_CONFIG_KEY]
        buzzer_sent = [c for c in configs if c[0] == BUZZER_CONFIG_KEY]
        assert any(c[1] == DEVICE_OFF for c in siren_sent)
        assert any(c[1] == BUZZER_OFF for c in buzzer_sent)
        # But still active
        assert mgr.active

    def test_timeout_deescalates(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        # Fake start time far in the past
        mgr.state.start_time = time.monotonic() - SIREN_MAX_DURATION_S - 1

        result = asyncio.run(mgr.check_timeout(writer))
        assert result is True
        assert not mgr.active

    def test_timeout_no_deescalate_if_recent(self):
        mgr, _ = self._make_manager()
        writer = object()

        asyncio.run(mgr.start(writer))
        result = asyncio.run(mgr.check_timeout(writer))
        assert result is False
        assert mgr.active

    def test_timeout_noop_inactive(self):
        mgr, _ = self._make_manager()
        result = asyncio.run(mgr.check_timeout())
        assert result is False


class TestTurretAiming:

    def _make_manager(self):
        configs_sent = []

        async def mock_send(writer, pkt_type, payload):
            cmd = ConfigCmd.from_bytes(payload)
            configs_sent.append((cmd.config_key, cmd.value))

        mgr = DeterrentManager(mock_send)
        return mgr, configs_sent

    def test_aim_at_sends_servos(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_at(90, 90, writer))

        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert len(pan_sent) == 1
        assert pan_sent[0][1] == 90
        assert len(tilt_sent) == 1
        assert tilt_sent[0][1] == 90

    def test_aim_at_clamps_pan(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_at(200, 90, writer))
        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        assert pan_sent[0][1] == 180

        configs.clear()
        asyncio.run(mgr.aim_at(-10, 90, writer))
        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        assert pan_sent[0][1] == 0

    def test_aim_at_clamps_tilt(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_at(90, 150, writer))
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert tilt_sent[0][1] == SERVO_TILT_MAX_DEG

        configs.clear()
        asyncio.run(mgr.aim_at(90, 10, writer))
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert tilt_sent[0][1] == SERVO_TILT_MIN_DEG

    def test_aim_from_frame_center(self):
        mgr, configs = self._make_manager()
        writer = object()

        # Center of 640x480 frame
        asyncio.run(mgr.aim_from_frame(320, 240, 640, 480, writer))

        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert pan_sent[0][1] == 90  # center → 90°
        assert tilt_sent[0][1] == 90  # center → 90°

    def test_aim_from_frame_top_left(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_from_frame(0, 0, 640, 480, writer))

        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert pan_sent[0][1] == 0  # left edge → 0°
        assert tilt_sent[0][1] == SERVO_TILT_MAX_DEG  # top → look up

    def test_aim_from_frame_bottom_right(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_from_frame(640, 480, 640, 480, writer))

        pan_sent = [c for c in configs if c[0] == SERVO_PAN_KEY]
        tilt_sent = [c for c in configs if c[0] == SERVO_TILT_KEY]
        assert pan_sent[0][1] == 180  # right edge → 180°
        assert tilt_sent[0][1] == SERVO_TILT_MIN_DEG  # bottom → look down

    def test_aim_from_frame_zero_size_noop(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.aim_from_frame(100, 100, 0, 0, writer))
        assert len(configs) == 0

    def test_laser_on_off(self):
        mgr, configs = self._make_manager()
        writer = object()

        asyncio.run(mgr.laser_on(writer))
        assert mgr.state.laser_on is True
        laser_sent = [c for c in configs if c[0] == LASER_CONFIG_KEY]
        assert laser_sent[0][1] == DEVICE_ON

        asyncio.run(mgr.laser_off(writer))
        assert mgr.state.laser_on is False

    def test_no_writer_no_send(self):
        mgr, configs = self._make_manager()
        asyncio.run(mgr.aim_at(90, 90))  # no writer
        assert len(configs) == 0

    def test_repr_inactive(self):
        mgr, _ = self._make_manager()
        r = repr(mgr)
        assert "inactive" in r

    def test_repr_active(self):
        mgr, _ = self._make_manager()
        writer = object()
        asyncio.run(mgr.start(writer))
        # Escalate to siren
        for _ in range(3):
            mgr.state.last_escalation_time = time.monotonic() - ESCALATION_STEP_S - 1
            asyncio.run(mgr.escalate(writer))

        r = repr(mgr)
        assert "SIREN" in r
        assert "siren" in r  # hw list


class TestDeterrentConstants:

    def test_escalation_step(self):
        assert ESCALATION_STEP_S > 0

    def test_deescalation_clear_count(self):
        assert DEESCALATION_CLEAR_COUNT > 0

    def test_siren_max_duration(self):
        assert SIREN_MAX_DURATION_S > 0

    def test_advance_speed(self):
        assert 0 < ADVANCE_SPEED_PCT <= 100

    def test_servo_neutral_in_range(self):
        assert 0 <= SERVO_PAN_NEUTRAL_DEG <= 180
        assert SERVO_TILT_MIN_DEG <= SERVO_TILT_NEUTRAL_DEG <= SERVO_TILT_MAX_DEG

    def test_levels_ordered(self):
        assert DeterrentLevel.NONE < DeterrentLevel.RECORD
        assert DeterrentLevel.RECORD < DeterrentLevel.NOTIFY
        assert DeterrentLevel.NOTIFY < DeterrentLevel.SPOTLIGHT
        assert DeterrentLevel.SPOTLIGHT < DeterrentLevel.SIREN
        assert DeterrentLevel.SIREN < DeterrentLevel.WARNING
        assert DeterrentLevel.WARNING < DeterrentLevel.ADVANCE
        assert DeterrentLevel.ADVANCE < DeterrentLevel.AGGRESSIVE

    def test_protocol_config_keys_unique(self):
        keys = [
            SIREN_CONFIG_KEY,
            SPOTLIGHT_CONFIG_KEY,
            LASER_CONFIG_KEY,
            SERVO_PAN_KEY,
            SERVO_TILT_KEY,
            SPEAKER_CONFIG_KEY,
            BUZZER_CONFIG_KEY,
        ]
        assert len(keys) == len(set(keys))
