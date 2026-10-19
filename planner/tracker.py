# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Intruder tracking — follow a detected person using VLM position estimation.

When VLM detects a person, it reports position as LEFT/CENTER/RIGHT in frame.
The tracker generates turn + follow commands to keep the intruder centered
while maintaining a safe distance. Records frames throughout tracking.

State machine:
    IDLE → ACQUIRING → TRACKING → LOST → (re-acquire or timeout → IDLE)

Usage:
    tracker = IntrusionTracker()
    tracker.start("person")
    cmd = tracker.update("CENTER", distance_mm=3000)
    # cmd = {"skill": "FORWARD", "args": {"speed": 30}}
"""

import enum
import logging
import time

logger = logging.getLogger("brain.tracker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRACKING_DISTANCE_MM = 2000  # maintain this distance from intruder
TRACKING_TIMEOUT_S = 60  # stop tracking after this long
TARGET_LOST_TIMEOUT_S = 10  # give up if target lost for this long
TRACKING_APPROACH_SPEED_PCT = 30  # speed when approaching
TRACKING_FOLLOW_SPEED_PCT = 25  # speed when at distance
TRACKING_TURN_SPEED_PCT = 35  # speed for turning toward target
TRACKING_CLOSE_RANGE_MM = 1000  # stop approaching if closer than this
TRACKING_FAR_RANGE_MM = 4000  # faster approach if farther than this
TRACKING_MAX_FRAMES = 100  # max evidence frames to collect

# VLM position labels
POSITION_LEFT = "left"
POSITION_CENTER = "center"
POSITION_RIGHT = "right"
POSITION_LOST = "lost"
VALID_POSITIONS = {POSITION_LEFT, POSITION_CENTER, POSITION_RIGHT}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TrackState(enum.Enum):
    IDLE = "idle"
    ACQUIRING = "acquiring"  # first detection, turning toward target
    TRACKING = "tracking"  # following target
    LOST = "lost"  # target disappeared, searching
    TIMEOUT = "timeout"  # tracking timed out


# ---------------------------------------------------------------------------
# IntrusionTracker
# ---------------------------------------------------------------------------


class IntrusionTracker:
    """Tracks a detected intruder using VLM position feedback."""

    def __init__(self):
        self.state = TrackState.IDLE
        self._target_label: str = ""
        self._start_time: float = 0.0
        self._lost_time: float = 0.0
        self._last_position: str = ""
        self._frames_recorded: int = 0
        self._updates: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self.state in (
            TrackState.ACQUIRING,
            TrackState.TRACKING,
            TrackState.LOST,
        )

    @property
    def target_label(self) -> str:
        return self._target_label

    @property
    def tracking_time_s(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def frames_recorded(self) -> int:
        return self._frames_recorded

    def start(self, target_label: str = "person"):
        """Begin tracking a target."""
        self.state = TrackState.ACQUIRING
        self._target_label = target_label
        self._start_time = time.monotonic()
        self._lost_time = 0.0
        self._last_position = ""
        self._frames_recorded = 0
        self._updates = 0
        logger.info("[Tracker] Acquiring target: %s", target_label)

    def stop(self):
        """Stop tracking."""
        if self.active:
            logger.info(
                "[Tracker] Stopped after %.1fs, %d updates",
                self.tracking_time_s,
                self._updates,
            )
        self.state = TrackState.IDLE
        self._target_label = ""
        self._start_time = 0.0
        self._lost_time = 0.0

    def update(
        self,
        position: str,
        distance_mm: int = 0,
    ) -> dict | None:
        """Feed VLM position update. Returns skill command dict or None.

        Args:
            position: "left", "center", "right", or "lost"
            distance_mm: estimated distance to target (0 = unknown)

        Returns:
            {"skill": "...", "args": {...}} or None if idle/timeout
        """
        if not self.active:
            return None

        self._updates += 1
        now = time.monotonic()

        # Check tracking timeout
        if now - self._start_time > TRACKING_TIMEOUT_S:
            self.state = TrackState.TIMEOUT
            logger.info("[Tracker] Timeout after %.1fs", TRACKING_TIMEOUT_S)
            return {"skill": "STOP", "args": {}}

        pos = position.strip().lower()

        # Target lost
        if pos == POSITION_LOST or pos not in VALID_POSITIONS:
            return self._handle_lost(now)

        # Target found / re-acquired
        self._last_position = pos
        if self.state == TrackState.LOST:
            logger.info("[Tracker] Re-acquired target at %s", pos)
        self.state = TrackState.TRACKING
        self._lost_time = 0.0

        return self._compute_command(pos, distance_mm)

    def record_frame(self) -> bool:
        """Mark that a frame was recorded. Returns False if max reached."""
        if self._frames_recorded >= TRACKING_MAX_FRAMES:
            return False
        self._frames_recorded += 1
        return True

    # ── Internal ──────────────────────────────────────────────────────────

    def _handle_lost(self, now: float) -> dict:
        """Handle target lost — search or give up."""
        if self.state != TrackState.LOST:
            self.state = TrackState.LOST
            self._lost_time = now
            logger.info("[Tracker] Target lost, searching...")

        # Check lost timeout
        if now - self._lost_time > TARGET_LOST_TIMEOUT_S:
            self.state = TrackState.TIMEOUT
            logger.info(
                "[Tracker] Target lost for >%.1fs, giving up",
                TARGET_LOST_TIMEOUT_S,
            )
            return {"skill": "STOP", "args": {}}

        # Search: rotate toward last known position
        if self._last_position == POSITION_LEFT:
            return {"skill": "TURN_LEFT", "args": {"degrees": 30}}
        elif self._last_position == POSITION_RIGHT:
            return {"skill": "TURN_RIGHT", "args": {"degrees": 30}}
        else:
            # Was center — do a slow scan
            return {"skill": "SCAN_360", "args": {"speed": 15}}

    def _compute_command(self, position: str, distance_mm: int) -> dict:
        """Generate motor command based on target position and distance."""
        # Turn toward target if not centered
        if position == POSITION_LEFT:
            return {
                "skill": "TURN_LEFT",
                "args": {
                    "degrees": 20,
                    "speed": TRACKING_TURN_SPEED_PCT,
                },
            }

        if position == POSITION_RIGHT:
            return {
                "skill": "TURN_RIGHT",
                "args": {
                    "degrees": 20,
                    "speed": TRACKING_TURN_SPEED_PCT,
                },
            }

        # Target is centered — approach or maintain distance
        if distance_mm > 0 and distance_mm <= TRACKING_CLOSE_RANGE_MM:
            # Too close — stop and observe
            return {"skill": "STOP", "args": {}}

        if distance_mm > TRACKING_FAR_RANGE_MM:
            # Far away — faster approach
            return {
                "skill": "FORWARD",
                "args": {
                    "speed": TRACKING_APPROACH_SPEED_PCT,
                },
            }

        # At tracking distance — slow follow
        return {
            "skill": "FORWARD",
            "args": {
                "speed": TRACKING_FOLLOW_SPEED_PCT,
            },
        }

    # ── Status ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"IntrusionTracker(state={self.state.value}, "
            f"target={self._target_label!r}, "
            f"updates={self._updates})"
        )
