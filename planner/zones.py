# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Zones of Interest — VLM-tagged waypoints with priority scanning and state tracking.

During mapping, VLM tags waypoints with zone types (door, window, gate, etc.).
During patrol, zones get:
  - More frequent scans (every pass vs every N passes for normal waypoints)
  - Longer dwell time for 360° scan
  - State comparison: "door was closed, now open" → alert

Usage:
    zm = ZoneManager()
    zm.tag_waypoint(wp, "door: closed, white, front entrance")
    changed, desc = zm.check_state(wp, "door: OPEN, white, front entrance")
    # changed = True, desc = "door state changed: was 'closed' now 'OPEN'"
"""

import logging
import re
import time
from dataclasses import dataclass, field

from planner.mapper import Waypoint

logger = logging.getLogger("brain.zones")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZONE_SCAN_EVERY_N_PASSES = 1  # zones scanned every pass (vs normal every N)
NORMAL_SCAN_EVERY_N_PASSES = 3  # normal waypoints scanned every 3rd pass
ZONE_DWELL_EXTRA_S = 2.0  # extra dwell time at zones (on top of patrol default)

# Zone types recognized by VLM tagging (ordered: more specific first)
ZONE_TYPES = [
    "open_area",
    "driveway",
    "garage",
    "entrance",
    "exit",
    "window",
    "stairs",
    "fence",
    "gate",
    "door",
]

# Zone types that get highest priority (entry points)
HIGH_PRIORITY_ZONES = {"door", "window", "gate", "entrance", "garage"}

# Zone priority levels
ZONE_PRIORITY_HIGH = 2  # entry points — scan every pass
ZONE_PRIORITY_NORMAL = 1  # other zones — scan every pass but less dwell
ZONE_PRIORITY_DEFAULT = 0  # non-zone waypoints — scan every N passes


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------


class ZoneManager:
    """Manages zones of interest: tagging, priority scanning, state change detection."""

    def __init__(self):
        self._state_history: dict[str, list[tuple[float, str]]] = {}
        self._pass_count: int = 0

    # ── VLM zone tagging ──────────────────────────────────────────────────

    def tag_waypoint(self, wp: Waypoint, vlm_description: str) -> str:
        """Tag a waypoint with zone type based on VLM description.

        Scans VLM output for known zone type keywords.
        Sets wp.zone_type, wp.zone_priority, wp.last_state.
        Returns detected zone_type or "".
        """
        desc_lower = vlm_description.lower()
        detected_type = ""

        for zone_type in ZONE_TYPES:
            # Match with underscores replaced by spaces for natural language
            pattern = zone_type.replace("_", " ")
            if pattern in desc_lower or zone_type in desc_lower:
                detected_type = zone_type
                break

        if detected_type:
            wp.zone_type = detected_type
            wp.zone_priority = (
                ZONE_PRIORITY_HIGH if detected_type in HIGH_PRIORITY_ZONES else ZONE_PRIORITY_NORMAL
            )
            wp.last_state = vlm_description
            self._record_state(wp, vlm_description)
            logger.info(
                "[Zones] Tagged waypoint (%d,%d) as '%s' (priority %d)",
                wp.x_mm,
                wp.y_mm,
                detected_type,
                wp.zone_priority,
            )

        return detected_type

    # ── State change detection ────────────────────────────────────────────

    def check_state(self, wp: Waypoint, vlm_description: str) -> tuple[bool, str]:
        """Compare current VLM description with last state for change detection.

        Returns (changed: bool, description: str).
        If no previous state, saves current and returns (False, "").
        """
        if not wp.zone_type:
            return False, ""

        prev_state = wp.last_state
        if not prev_state:
            wp.last_state = vlm_description
            self._record_state(wp, vlm_description)
            return False, ""

        changed = self._detect_change(prev_state, vlm_description)
        wp.last_state = vlm_description
        self._record_state(wp, vlm_description)

        if changed:
            desc = (
                f"{wp.zone_type} state changed at ({wp.x_mm},{wp.y_mm}): "
                f"was '{self._summarize(prev_state)}' "
                f"now '{self._summarize(vlm_description)}'"
            )
            logger.warning("[Zones] %s", desc)
            return True, desc

        return False, ""

    # ── Patrol scanning decisions ─────────────────────────────────────────

    def increment_pass(self):
        """Called when patrol completes one full lap."""
        self._pass_count += 1

    @property
    def pass_count(self) -> int:
        return self._pass_count

    def should_scan(self, wp: Waypoint) -> bool:
        """Decide if this waypoint should be scanned on the current pass.

        Zones of interest: scanned every pass.
        Normal waypoints: scanned every NORMAL_SCAN_EVERY_N_PASSES passes.
        """
        if wp.zone_type:
            return True  # always scan zones
        # Normal: scan on passes that are multiples of N (and always on pass 0)
        if self._pass_count == 0:
            return True
        return (self._pass_count % NORMAL_SCAN_EVERY_N_PASSES) == 0

    def extra_dwell_s(self, wp: Waypoint) -> float:
        """Extra dwell time at this waypoint (seconds)."""
        if wp.zone_type:
            return ZONE_DWELL_EXTRA_S
        return 0.0

    # ── Zone queries ──────────────────────────────────────────────────────

    def get_zones(self, waypoints: list[Waypoint]) -> list[Waypoint]:
        """Return only waypoints that are tagged as zones."""
        return [wp for wp in waypoints if wp.zone_type]

    def get_zones_by_type(
        self,
        waypoints: list[Waypoint],
        zone_type: str,
    ) -> list[Waypoint]:
        """Return waypoints matching a specific zone type."""
        return [wp for wp in waypoints if wp.zone_type == zone_type]

    def get_high_priority_zones(
        self,
        waypoints: list[Waypoint],
    ) -> list[Waypoint]:
        """Return only high-priority zones (entry points)."""
        return [wp for wp in waypoints if wp.zone_priority >= ZONE_PRIORITY_HIGH]

    def zone_summary(self, waypoints: list[Waypoint]) -> dict[str, int]:
        """Count zones by type."""
        counts: dict[str, int] = {}
        for wp in waypoints:
            if wp.zone_type:
                counts[wp.zone_type] = counts.get(wp.zone_type, 0) + 1
        return counts

    def get_state_history(self, wp: Waypoint) -> list[tuple[float, str]]:
        """Return state history for a waypoint: [(timestamp, description), ...]."""
        key = self._wp_key(wp)
        return list(self._state_history.get(key, []))

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_change(prev: str, curr: str) -> bool:
        """Detect meaningful state change between two VLM descriptions.

        Looks for state keywords that flip: open↔closed, on↔off, present↔absent.
        """
        prev_lower = prev.lower()
        curr_lower = curr.lower()

        # State pairs to check
        state_pairs = [
            ("open", "closed"),
            ("on", "off"),
            ("present", "absent"),
            ("occupied", "empty"),
            ("locked", "unlocked"),
            ("person", "clear"),
            ("vehicle", "clear"),
            ("light on", "light off"),
            ("dark", "lit"),
        ]

        for state_a, state_b in state_pairs:
            prev_has_a = state_a in prev_lower
            prev_has_b = state_b in prev_lower
            curr_has_a = state_a in curr_lower
            curr_has_b = state_b in curr_lower

            # Detect flip: A→B or B→A
            if (prev_has_a and curr_has_b) or (prev_has_b and curr_has_a):
                return True

        return False

    @staticmethod
    def _summarize(description: str, max_len: int = 60) -> str:
        """Truncate description for logging."""
        if len(description) <= max_len:
            return description
        return description[: max_len - 3] + "..."

    @staticmethod
    def _wp_key(wp: Waypoint) -> str:
        """Unique key for waypoint in history dict."""
        return f"{wp.x_mm}_{wp.y_mm}"

    def _record_state(self, wp: Waypoint, description: str):
        """Append to state history."""
        key = self._wp_key(wp)
        if key not in self._state_history:
            self._state_history[key] = []
        self._state_history[key].append((time.time(), description))
