# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/zones.py — zones of interest management."""

import time

from planner.zones import (
    ZoneManager,
    ZONE_TYPES,
    HIGH_PRIORITY_ZONES,
    ZONE_SCAN_EVERY_N_PASSES,
    NORMAL_SCAN_EVERY_N_PASSES,
    ZONE_DWELL_EXTRA_S,
    ZONE_PRIORITY_HIGH,
    ZONE_PRIORITY_NORMAL,
    ZONE_PRIORITY_DEFAULT,
)
from planner.mapper import Waypoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wp(x=0, y=0, label="", zone_type="", zone_priority=0):
    return Waypoint(
        x_mm=x,
        y_mm=y,
        heading_cdeg=0,
        label=label,
        zone_type=zone_type,
        zone_priority=zone_priority,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_zone_types_not_empty(self):
        assert len(ZONE_TYPES) > 0

    def test_high_priority_subset_of_zone_types(self):
        assert HIGH_PRIORITY_ZONES.issubset(set(ZONE_TYPES))

    def test_scan_every_n_passes_positive(self):
        assert ZONE_SCAN_EVERY_N_PASSES >= 1
        assert NORMAL_SCAN_EVERY_N_PASSES >= 1

    def test_normal_scan_less_frequent(self):
        assert NORMAL_SCAN_EVERY_N_PASSES >= ZONE_SCAN_EVERY_N_PASSES

    def test_dwell_extra_positive(self):
        assert ZONE_DWELL_EXTRA_S > 0

    def test_priority_ordering(self):
        assert ZONE_PRIORITY_DEFAULT < ZONE_PRIORITY_NORMAL < ZONE_PRIORITY_HIGH


# ---------------------------------------------------------------------------
# ZoneManager — tagging
# ---------------------------------------------------------------------------


class TestZoneManagerTagging:
    def test_tag_door(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "White door, closed, main access")
        assert result == "door"
        assert wp.zone_type == "door"
        assert wp.zone_priority == ZONE_PRIORITY_HIGH

    def test_tag_window(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Large window with curtains")
        assert result == "window"
        assert wp.zone_type == "window"
        assert wp.zone_priority == ZONE_PRIORITY_HIGH

    def test_tag_gate(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Metal gate, locked")
        assert result == "gate"
        assert wp.zone_priority == ZONE_PRIORITY_HIGH

    def test_tag_open_area(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Open area with grass")
        assert result == "open_area"
        assert wp.zone_priority == ZONE_PRIORITY_NORMAL

    def test_tag_garage(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Garage door, closed")
        assert result == "garage"
        assert wp.zone_priority == ZONE_PRIORITY_HIGH

    def test_tag_stairs(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Concrete stairs leading up")
        assert result == "stairs"
        assert wp.zone_priority == ZONE_PRIORITY_NORMAL

    def test_no_zone_detected(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "Plain wall, nothing interesting")
        assert result == ""
        assert wp.zone_type == ""
        assert wp.zone_priority == ZONE_PRIORITY_DEFAULT

    def test_tag_sets_last_state(self):
        zm = ZoneManager()
        wp = _wp()
        zm.tag_waypoint(wp, "Door closed, white paint")
        assert wp.last_state == "Door closed, white paint"

    def test_tag_case_insensitive(self):
        zm = ZoneManager()
        wp = _wp()
        result = zm.tag_waypoint(wp, "DOOR to the garden")
        assert result == "door"


# ---------------------------------------------------------------------------
# ZoneManager — state change detection
# ---------------------------------------------------------------------------


class TestZoneManagerStateChange:
    def test_first_state_no_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        changed, desc = zm.check_state(wp, "Door closed, white")
        assert not changed
        assert desc == ""

    def test_open_closed_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        wp.last_state = "Door closed"
        changed, desc = zm.check_state(wp, "Door open")
        assert changed
        assert "door" in desc.lower()

    def test_closed_open_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="gate")
        wp.last_state = "Gate open"
        changed, desc = zm.check_state(wp, "Gate closed")
        assert changed

    def test_on_off_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="entrance")
        wp.last_state = "Light on at entrance"
        changed, _ = zm.check_state(wp, "Light off at entrance")
        assert changed

    def test_person_clear_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="driveway")
        wp.last_state = "Person walking on driveway"
        changed, _ = zm.check_state(wp, "Clear driveway, no one around")
        assert changed

    def test_no_change_similar_descriptions(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        wp.last_state = "Door closed, white paint, clean"
        changed, _ = zm.check_state(wp, "White door, closed, clean porch")
        assert not changed

    def test_non_zone_returns_no_change(self):
        zm = ZoneManager()
        wp = _wp()  # no zone_type
        changed, desc = zm.check_state(wp, "anything")
        assert not changed
        assert desc == ""

    def test_state_updates_after_check(self):
        zm = ZoneManager()
        wp = _wp(zone_type="window")
        wp.last_state = "Window closed"
        zm.check_state(wp, "Window open")
        assert wp.last_state == "Window open"

    def test_vehicle_clear_change(self):
        zm = ZoneManager()
        wp = _wp(zone_type="driveway")
        wp.last_state = "Vehicle parked in driveway"
        changed, _ = zm.check_state(wp, "Clear driveway")
        assert changed


# ---------------------------------------------------------------------------
# ZoneManager — scan scheduling
# ---------------------------------------------------------------------------


class TestZoneManagerScanning:
    def test_zone_always_scanned(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        # Any pass
        for _ in range(10):
            assert zm.should_scan(wp)
            zm.increment_pass()

    def test_normal_wp_scanned_first_pass(self):
        zm = ZoneManager()
        wp = _wp()  # no zone
        assert zm.should_scan(wp)

    def test_normal_wp_skipped_on_non_multiple(self):
        zm = ZoneManager()
        wp = _wp()
        zm.increment_pass()  # pass 1
        # pass 1 is not multiple of 3 → skip
        if NORMAL_SCAN_EVERY_N_PASSES > 1:
            assert not zm.should_scan(wp)

    def test_normal_wp_scanned_on_multiple(self):
        zm = ZoneManager()
        wp = _wp()
        for _ in range(NORMAL_SCAN_EVERY_N_PASSES):
            zm.increment_pass()
        assert zm.should_scan(wp)

    def test_extra_dwell_for_zone(self):
        zm = ZoneManager()
        wp_zone = _wp(zone_type="door")
        wp_normal = _wp()
        assert zm.extra_dwell_s(wp_zone) == ZONE_DWELL_EXTRA_S
        assert zm.extra_dwell_s(wp_normal) == 0.0

    def test_pass_count_increments(self):
        zm = ZoneManager()
        assert zm.pass_count == 0
        zm.increment_pass()
        assert zm.pass_count == 1
        zm.increment_pass()
        assert zm.pass_count == 2


# ---------------------------------------------------------------------------
# ZoneManager — queries
# ---------------------------------------------------------------------------


class TestZoneManagerQueries:
    def test_get_zones(self):
        zm = ZoneManager()
        wps = [
            _wp(x=0, zone_type="door"),
            _wp(x=1000),
            _wp(x=2000, zone_type="window"),
            _wp(x=3000),
        ]
        zones = zm.get_zones(wps)
        assert len(zones) == 2
        assert zones[0].zone_type == "door"
        assert zones[1].zone_type == "window"

    def test_get_zones_empty(self):
        zm = ZoneManager()
        assert zm.get_zones([]) == []
        assert zm.get_zones([_wp()]) == []

    def test_get_zones_by_type(self):
        zm = ZoneManager()
        wps = [
            _wp(x=0, zone_type="door"),
            _wp(x=1000, zone_type="window"),
            _wp(x=2000, zone_type="door"),
        ]
        doors = zm.get_zones_by_type(wps, "door")
        assert len(doors) == 2

    def test_get_high_priority_zones(self):
        zm = ZoneManager()
        wps = [
            _wp(x=0, zone_type="door", zone_priority=ZONE_PRIORITY_HIGH),
            _wp(x=1000, zone_type="open_area", zone_priority=ZONE_PRIORITY_NORMAL),
            _wp(x=2000, zone_type="gate", zone_priority=ZONE_PRIORITY_HIGH),
            _wp(x=3000),
        ]
        high = zm.get_high_priority_zones(wps)
        assert len(high) == 2

    def test_zone_summary(self):
        zm = ZoneManager()
        wps = [
            _wp(zone_type="door"),
            _wp(zone_type="door"),
            _wp(zone_type="window"),
            _wp(),
        ]
        summary = zm.zone_summary(wps)
        assert summary == {"door": 2, "window": 1}

    def test_zone_summary_empty(self):
        zm = ZoneManager()
        assert zm.zone_summary([]) == {}


# ---------------------------------------------------------------------------
# ZoneManager — state history
# ---------------------------------------------------------------------------


class TestZoneManagerHistory:
    def test_history_empty_initially(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        assert zm.get_state_history(wp) == []

    def test_history_recorded_on_tag(self):
        zm = ZoneManager()
        wp = _wp()
        zm.tag_waypoint(wp, "Door closed")
        history = zm.get_state_history(wp)
        assert len(history) == 1
        assert history[0][1] == "Door closed"

    def test_history_recorded_on_check(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        zm.check_state(wp, "Door closed")
        zm.check_state(wp, "Door open")
        history = zm.get_state_history(wp)
        assert len(history) == 2

    def test_history_timestamps_increase(self):
        zm = ZoneManager()
        wp = _wp(zone_type="door")
        zm.check_state(wp, "State 1")
        zm.check_state(wp, "State 2")
        history = zm.get_state_history(wp)
        assert history[1][0] >= history[0][0]

    def test_history_per_waypoint(self):
        zm = ZoneManager()
        wp1 = _wp(x=0, y=0, zone_type="door")
        wp2 = _wp(x=1000, y=0, zone_type="window")
        zm.check_state(wp1, "Door closed")
        zm.check_state(wp2, "Window shut")
        assert len(zm.get_state_history(wp1)) == 1
        assert len(zm.get_state_history(wp2)) == 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternals:
    def test_detect_change_open_closed(self):
        assert ZoneManager._detect_change("door closed", "door open")
        assert ZoneManager._detect_change("door open", "door closed")

    def test_detect_change_on_off(self):
        assert ZoneManager._detect_change("light on", "light off")

    def test_detect_change_no_change(self):
        assert not ZoneManager._detect_change("wall plain", "wall clean")

    def test_detect_change_case_insensitive(self):
        assert ZoneManager._detect_change("DOOR CLOSED", "door OPEN")

    def test_summarize_short(self):
        assert ZoneManager._summarize("short") == "short"

    def test_summarize_long(self):
        long = "x" * 100
        result = ZoneManager._summarize(long, max_len=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_wp_key(self):
        wp = _wp(x=100, y=200)
        assert ZoneManager._wp_key(wp) == "100_200"
