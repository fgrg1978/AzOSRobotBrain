# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception/change_detector.py — baseline comparison and false positive tracking."""

import time

from perception.change_detector import (
    ChangeDetector,
    ChangeType,
    ChangeResult,
    BaselineEntry,
    FalsePositiveRecord,
    BASELINE_UPDATE_HOURS,
    FALSE_POSITIVE_THRESHOLD,
    FALSE_POSITIVE_DECAY_S,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_baseline_update_hours_positive(self):
        assert BASELINE_UPDATE_HOURS > 0

    def test_false_positive_threshold_positive(self):
        assert FALSE_POSITIVE_THRESHOLD >= 1

    def test_false_positive_decay_positive(self):
        assert FALSE_POSITIVE_DECAY_S > 0


# ---------------------------------------------------------------------------
# ChangeDetector — baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_set_baseline(self):
        cd = ChangeDetector()
        cd.set_baseline("zone_1", "Door closed, white")
        assert cd.has_baseline("zone_1")
        assert cd.baseline_count == 1

    def test_get_baseline(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Door closed")
        entry = cd.get_baseline("z")
        assert entry is not None
        assert entry.description == "Door closed"
        assert entry.timestamp > 0

    def test_get_baseline_not_found(self):
        cd = ChangeDetector()
        assert cd.get_baseline("nope") is None

    def test_has_baseline(self):
        cd = ChangeDetector()
        assert not cd.has_baseline("z")
        cd.set_baseline("z", "test")
        assert cd.has_baseline("z")

    def test_clear_baseline(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "test")
        cd.clear_baseline("z")
        assert not cd.has_baseline("z")

    def test_clear_all(self):
        cd = ChangeDetector()
        cd.set_baseline("a", "x")
        cd.set_baseline("b", "y")
        cd.clear_all()
        assert cd.baseline_count == 0

    def test_update_baseline(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "old")
        cd.set_baseline("z", "new")
        assert cd.get_baseline("z").description == "new"

    def test_should_refresh_no_baseline(self):
        cd = ChangeDetector()
        assert cd.should_refresh_baseline("z")

    def test_should_refresh_fresh(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "test")
        assert not cd.should_refresh_baseline("z")

    def test_should_refresh_stale(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "test")
        cd._baselines["z"].timestamp = time.time() - (BASELINE_UPDATE_HOURS + 1) * 3600
        assert cd.should_refresh_baseline("z")


# ---------------------------------------------------------------------------
# ChangeDetector — comparison
# ---------------------------------------------------------------------------


class TestComparison:
    def test_no_baseline_returns_unchanged(self):
        cd = ChangeDetector()
        result = cd.compare("z", "anything")
        assert not result.changed
        assert "no baseline" in result.description

    def test_unchanged(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Door closed, white, clean porch")
        result = cd.compare("z", "White door, closed, clean porch area")
        assert not result.changed
        assert result.change_type == ChangeType.UNCHANGED

    def test_state_changed_open_closed(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Door closed")
        result = cd.compare("z", "Door open")
        assert result.changed
        assert result.change_type == ChangeType.STATE_CHANGED
        assert result.confidence > 0

    def test_state_changed_locked_unlocked(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Gate locked")
        result = cd.compare("z", "Gate unlocked")
        assert result.changed
        assert result.change_type == ChangeType.STATE_CHANGED

    def test_state_changed_on_off(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Light on")
        result = cd.compare("z", "Light off")
        assert result.changed
        assert result.change_type == ChangeType.STATE_CHANGED

    def test_intrusion_person_detected(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Empty driveway, clean")
        result = cd.compare("z", "Person walking on driveway")
        assert result.changed
        assert result.change_type == ChangeType.INTRUSION
        assert result.confidence >= 0.9

    def test_intrusion_person_already_in_baseline(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Person sitting on bench")
        result = cd.compare("z", "Person sitting on bench, same position")
        assert not result.changed

    def test_new_object_vehicle(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Empty parking area")
        result = cd.compare("z", "Vehicle parked in lot")
        assert result.changed
        assert result.change_type == ChangeType.NEW_OBJECT

    def test_new_object_keyword(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Clean porch")
        result = cd.compare("z", "New box appeared on porch")
        assert result.changed
        assert result.change_type == ChangeType.NEW_OBJECT

    def test_missing_item(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Potted plant by door")
        result = cd.compare("z", "Plant is missing from doorstep")
        assert result.changed
        assert result.change_type == ChangeType.MISSING

    def test_case_insensitive(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "DOOR CLOSED")
        result = cd.compare("z", "door open")
        assert result.changed

    def test_confidence_intrusion_highest(self):
        cd = ChangeDetector()
        cd.set_baseline("z", "Empty")
        r_intrusion = cd.compare("z", "Person detected")
        cd.set_baseline("z2", "door closed")
        r_state = cd.compare("z2", "door open")
        assert r_intrusion.confidence >= r_state.confidence


# ---------------------------------------------------------------------------
# ChangeDetector — false positives
# ---------------------------------------------------------------------------


class TestFalsePositives:
    def test_initial_not_false_positive(self):
        cd = ChangeDetector()
        assert not cd.is_false_positive_zone("z")

    def test_below_threshold_not_suppressed(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD - 1):
            cd.record_sensor_only_trigger("z")
        assert not cd.is_false_positive_zone("z")

    def test_at_threshold_suppressed(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD):
            cd.record_sensor_only_trigger("z")
        assert cd.is_false_positive_zone("z")

    def test_false_positive_zones_list(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD):
            cd.record_sensor_only_trigger("z1")
        assert "z1" in cd.false_positive_zones()

    def test_reset_false_positive(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD):
            cd.record_sensor_only_trigger("z")
        cd.reset_false_positive("z")
        assert not cd.is_false_positive_zone("z")

    def test_decay_resets_counter(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD - 1):
            cd.record_sensor_only_trigger("z")
        # Simulate time passing beyond decay
        cd._false_positives["z"].last_trigger = time.time() - FALSE_POSITIVE_DECAY_S - 1
        cd.record_sensor_only_trigger("z")
        # Counter was reset + 1 new trigger, should not be suppressed
        assert not cd.is_false_positive_zone("z")

    def test_clear_all_clears_false_positives(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD):
            cd.record_sensor_only_trigger("z")
        cd.clear_all()
        assert not cd.is_false_positive_zone("z")

    def test_multiple_zones_independent(self):
        cd = ChangeDetector()
        for _ in range(FALSE_POSITIVE_THRESHOLD):
            cd.record_sensor_only_trigger("z1")
        cd.record_sensor_only_trigger("z2")
        assert cd.is_false_positive_zone("z1")
        assert not cd.is_false_positive_zone("z2")
