# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/sensors.py — multi-sensor fusion (PIR/sound/IR)."""

import time
import pytest

from planner.sensors import (
    SensorFusion,
    SensorTrigger,
    SENSOR_DEBOUNCE_S,
    SENSOR_TRIGGER_COOLDOWN_S,
    PIR_LABEL,
    SOUND_LABEL,
    IR_LABEL,
)
from protocol import SENSOR_FLAG_PIR, SENSOR_FLAG_SOUND, SENSOR_FLAG_IR


class TestSensorFusion:

    def _make_fusion(self, debounce_s=0.0, cooldown_s=0.0):
        """Create a SensorFusion with zero timing for deterministic tests."""
        return SensorFusion(debounce_s=debounce_s, cooldown_s=cooldown_s)

    def test_no_flags_no_triggers(self):
        sf = self._make_fusion()
        assert sf.process_flags(0) == []

    def test_pir_trigger(self):
        sf = self._make_fusion()
        triggers = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(triggers) == 1
        assert triggers[0].label == PIR_LABEL
        assert triggers[0].flag == SENSOR_FLAG_PIR

    def test_sound_trigger(self):
        sf = self._make_fusion()
        triggers = sf.process_flags(SENSOR_FLAG_SOUND)
        assert len(triggers) == 1
        assert triggers[0].label == SOUND_LABEL

    def test_ir_trigger(self):
        sf = self._make_fusion()
        triggers = sf.process_flags(SENSOR_FLAG_IR)
        assert len(triggers) == 1
        assert triggers[0].label == IR_LABEL

    def test_multiple_simultaneous_triggers(self):
        sf = self._make_fusion()
        flags = SENSOR_FLAG_PIR | SENSOR_FLAG_SOUND | SENSOR_FLAG_IR
        triggers = sf.process_flags(flags)
        assert len(triggers) == 3
        labels = {t.label for t in triggers}
        assert labels == {PIR_LABEL, SOUND_LABEL, IR_LABEL}

    def test_debounce_blocks_rapid_retrigger(self):
        sf = self._make_fusion(debounce_s=10.0)
        t1 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t1) == 1
        # immediate re-trigger should be blocked
        t2 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t2) == 0

    def test_debounce_allows_different_sensor(self):
        sf = self._make_fusion(debounce_s=10.0)
        t1 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t1) == 1
        # different sensor should not be blocked
        t2 = sf.process_flags(SENSOR_FLAG_SOUND)
        assert len(t2) == 1

    def test_cooldown_blocks_after_alert(self):
        sf = self._make_fusion(cooldown_s=10.0)
        t1 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t1) == 1
        sf.mark_all_alerted(t1)
        # should be blocked by cooldown
        t2 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t2) == 0

    def test_cooldown_doesnt_block_unmarked(self):
        sf = self._make_fusion(cooldown_s=10.0)
        t1 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t1) == 1
        # don't mark as alerted — should still trigger
        # (debounce is 0 so immediate re-trigger works)
        t2 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t2) == 1

    def test_disabled_sensor_not_triggered(self):
        sf = self._make_fusion()
        sf.set_enabled(pir=False, sound=True, ir=True)
        triggers = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(triggers) == 0

    def test_set_enabled_selective(self):
        sf = self._make_fusion()
        sf.set_enabled(pir=True, sound=False, ir=False)
        flags = SENSOR_FLAG_PIR | SENSOR_FLAG_SOUND | SENSOR_FLAG_IR
        triggers = sf.process_flags(flags)
        assert len(triggers) == 1
        assert triggers[0].label == PIR_LABEL

    def test_reset_clears_state(self):
        sf = self._make_fusion(debounce_s=10.0, cooldown_s=10.0)
        t1 = sf.process_flags(SENSOR_FLAG_PIR)
        sf.mark_all_alerted(t1)

        sf.reset()

        # after reset, should trigger again
        t2 = sf.process_flags(SENSOR_FLAG_PIR)
        assert len(t2) == 1

    def test_repr(self):
        sf = self._make_fusion()
        r = repr(sf)
        assert "SensorFusion" in r
        assert "PIR" in r
        assert "Sound" in r
        assert "IR" in r

    def test_repr_disabled(self):
        sf = self._make_fusion()
        sf.set_enabled(pir=False, sound=False, ir=False)
        r = repr(sf)
        assert "SensorFusion" in r

    def test_trigger_has_timestamp(self):
        sf = self._make_fusion()
        before = time.monotonic()
        triggers = sf.process_flags(SENSOR_FLAG_PIR)
        after = time.monotonic()
        assert len(triggers) == 1
        assert before <= triggers[0].timestamp <= after

    def test_default_debounce_constant(self):
        assert SENSOR_DEBOUNCE_S > 0

    def test_default_cooldown_constant(self):
        assert SENSOR_TRIGGER_COOLDOWN_S > 0
