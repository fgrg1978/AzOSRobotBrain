# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/tracker.py — intruder tracking."""

import time

from planner.tracker import (
    IntrusionTracker,
    TrackState,
    TRACKING_DISTANCE_MM,
    TRACKING_TIMEOUT_S,
    TARGET_LOST_TIMEOUT_S,
    TRACKING_APPROACH_SPEED_PCT,
    TRACKING_FOLLOW_SPEED_PCT,
    TRACKING_TURN_SPEED_PCT,
    TRACKING_CLOSE_RANGE_MM,
    TRACKING_FAR_RANGE_MM,
    TRACKING_MAX_FRAMES,
    POSITION_LEFT,
    POSITION_CENTER,
    POSITION_RIGHT,
    POSITION_LOST,
    VALID_POSITIONS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_tracking_distance_positive(self):
        assert TRACKING_DISTANCE_MM > 0

    def test_tracking_timeout_positive(self):
        assert TRACKING_TIMEOUT_S > 0

    def test_target_lost_timeout_positive(self):
        assert TARGET_LOST_TIMEOUT_S > 0

    def test_target_lost_less_than_tracking(self):
        assert TARGET_LOST_TIMEOUT_S < TRACKING_TIMEOUT_S

    def test_speeds_positive(self):
        assert TRACKING_APPROACH_SPEED_PCT > 0
        assert TRACKING_FOLLOW_SPEED_PCT > 0
        assert TRACKING_TURN_SPEED_PCT > 0

    def test_close_range_less_than_far(self):
        assert TRACKING_CLOSE_RANGE_MM < TRACKING_FAR_RANGE_MM

    def test_max_frames_positive(self):
        assert TRACKING_MAX_FRAMES > 0

    def test_valid_positions(self):
        assert POSITION_LEFT in VALID_POSITIONS
        assert POSITION_CENTER in VALID_POSITIONS
        assert POSITION_RIGHT in VALID_POSITIONS
        assert POSITION_LOST not in VALID_POSITIONS


# ---------------------------------------------------------------------------
# IntrusionTracker — init
# ---------------------------------------------------------------------------


class TestTrackerInit:
    def test_initial_idle(self):
        t = IntrusionTracker()
        assert t.state == TrackState.IDLE
        assert not t.active
        assert t.target_label == ""

    def test_repr(self):
        t = IntrusionTracker()
        assert "idle" in repr(t)


# ---------------------------------------------------------------------------
# IntrusionTracker — start/stop
# ---------------------------------------------------------------------------


class TestTrackerStartStop:
    def test_start(self):
        t = IntrusionTracker()
        t.start("person")
        assert t.state == TrackState.ACQUIRING
        assert t.active
        assert t.target_label == "person"

    def test_start_custom_target(self):
        t = IntrusionTracker()
        t.start("vehicle")
        assert t.target_label == "vehicle"

    def test_stop(self):
        t = IntrusionTracker()
        t.start("person")
        t.stop()
        assert t.state == TrackState.IDLE
        assert not t.active
        assert t.target_label == ""

    def test_stop_when_idle_noop(self):
        t = IntrusionTracker()
        t.stop()
        assert t.state == TrackState.IDLE

    def test_tracking_time(self):
        t = IntrusionTracker()
        assert t.tracking_time_s == 0.0
        t.start("person")
        assert t.tracking_time_s >= 0.0


# ---------------------------------------------------------------------------
# IntrusionTracker — update with positions
# ---------------------------------------------------------------------------


class TestTrackerUpdate:
    def test_update_when_idle_returns_none(self):
        t = IntrusionTracker()
        assert t.update("center") is None

    def test_update_center_forward(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("center", distance_mm=3000)
        assert cmd is not None
        assert cmd["skill"] == "FORWARD"
        assert t.state == TrackState.TRACKING

    def test_update_left_turns_left(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("left")
        assert cmd["skill"] == "TURN_LEFT"

    def test_update_right_turns_right(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("right")
        assert cmd["skill"] == "TURN_RIGHT"

    def test_update_center_close_stops(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("center", distance_mm=TRACKING_CLOSE_RANGE_MM - 100)
        assert cmd["skill"] == "STOP"

    def test_update_center_far_approaches(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("center", distance_mm=TRACKING_FAR_RANGE_MM + 1000)
        assert cmd["skill"] == "FORWARD"
        assert cmd["args"]["speed"] == TRACKING_APPROACH_SPEED_PCT

    def test_update_center_medium_follows(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("center", distance_mm=TRACKING_DISTANCE_MM)
        assert cmd["skill"] == "FORWARD"
        assert cmd["args"]["speed"] == TRACKING_FOLLOW_SPEED_PCT

    def test_update_center_no_distance(self):
        """When distance is 0 (unknown), follow at default speed."""
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("center", distance_mm=0)
        assert cmd["skill"] == "FORWARD"
        assert cmd["args"]["speed"] == TRACKING_FOLLOW_SPEED_PCT

    def test_update_case_insensitive(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("LEFT")
        assert cmd["skill"] == "TURN_LEFT"

    def test_update_with_whitespace(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("  right  ")
        assert cmd["skill"] == "TURN_RIGHT"


# ---------------------------------------------------------------------------
# IntrusionTracker — target lost
# ---------------------------------------------------------------------------


class TestTrackerLost:
    def test_lost_starts_search(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("center")  # establish tracking
        cmd = t.update("lost")
        assert t.state == TrackState.LOST
        assert cmd is not None

    def test_lost_from_left_searches_left(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("left")
        cmd = t.update("lost")
        assert cmd["skill"] == "TURN_LEFT"

    def test_lost_from_right_searches_right(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("right")
        cmd = t.update("lost")
        assert cmd["skill"] == "TURN_RIGHT"

    def test_lost_from_center_scans(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("center")
        cmd = t.update("lost")
        assert cmd["skill"] == "SCAN_360"

    def test_re_acquire_after_lost(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("center")
        t.update("lost")
        assert t.state == TrackState.LOST
        t.update("center")
        assert t.state == TrackState.TRACKING

    def test_lost_timeout(self):
        t = IntrusionTracker()
        t.start("person")
        t.update("center")
        t.update("lost")
        # Force lost time to be expired
        t._lost_time = time.monotonic() - TARGET_LOST_TIMEOUT_S - 1
        cmd = t.update("lost")
        assert t.state == TrackState.TIMEOUT
        assert cmd["skill"] == "STOP"

    def test_invalid_position_treated_as_lost(self):
        t = IntrusionTracker()
        t.start("person")
        cmd = t.update("unknown_garbage")
        assert t.state == TrackState.LOST


# ---------------------------------------------------------------------------
# IntrusionTracker — timeout
# ---------------------------------------------------------------------------


class TestTrackerTimeout:
    def test_tracking_timeout(self):
        t = IntrusionTracker()
        t.start("person")
        # Force start time to be expired
        t._start_time = time.monotonic() - TRACKING_TIMEOUT_S - 1
        cmd = t.update("center")
        assert t.state == TrackState.TIMEOUT
        assert cmd["skill"] == "STOP"

    def test_after_timeout_update_returns_none(self):
        t = IntrusionTracker()
        t.state = TrackState.TIMEOUT
        assert t.update("center") is None
        assert not t.active


# ---------------------------------------------------------------------------
# IntrusionTracker — frame recording
# ---------------------------------------------------------------------------


class TestTrackerFrames:
    def test_record_frame(self):
        t = IntrusionTracker()
        t.start("person")
        assert t.record_frame()
        assert t.frames_recorded == 1

    def test_record_frame_max(self):
        t = IntrusionTracker()
        t.start("person")
        for _ in range(TRACKING_MAX_FRAMES):
            assert t.record_frame()
        assert not t.record_frame()  # max reached
        assert t.frames_recorded == TRACKING_MAX_FRAMES

    def test_frames_reset_on_start(self):
        t = IntrusionTracker()
        t.start("person")
        t.record_frame()
        t.record_frame()
        t.start("vehicle")  # restart
        assert t.frames_recorded == 0
