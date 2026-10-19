# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Change detection — baseline comparison for learning and adaptation.

Stores baseline VLM descriptions per zone during first patrol.
On subsequent patrols, compares current vs baseline via keyword matching.
Tracks false positive patterns to reduce alert noise.

Usage:
    cd = ChangeDetector()
    cd.set_baseline("zone_1", "Door closed, white, clean porch")
    result = cd.compare("zone_1", "Door OPEN, white, clean porch")
    # result = ChangeResult(changed=True, change_type="STATE_CHANGED", ...)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("brain.change_detect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_UPDATE_HOURS = 24  # refresh baseline daily during daytime
FALSE_POSITIVE_THRESHOLD = 3  # mark as false positive after N sensor-only triggers
FALSE_POSITIVE_DECAY_S = 3600  # reset false positive counter after 1 hour


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChangeType(Enum):
    UNCHANGED = "unchanged"
    NEW_OBJECT = "new_object"
    STATE_CHANGED = "state_changed"
    INTRUSION = "intrusion"
    MISSING = "missing"


@dataclass
class ChangeResult:
    changed: bool
    change_type: ChangeType = ChangeType.UNCHANGED
    description: str = ""
    confidence: float = 0.0  # 0-1, how certain the change is


@dataclass
class BaselineEntry:
    zone_id: str
    description: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FalsePositiveRecord:
    zone_id: str
    trigger_count: int = 0
    last_trigger: float = 0.0
    suppressed: bool = False


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------


class ChangeDetector:
    """Compares current observations against baselines for change detection."""

    def __init__(self):
        self._baselines: dict[str, BaselineEntry] = {}
        self._false_positives: dict[str, FalsePositiveRecord] = {}

    # ── Baseline management ───────────────────────────────────────────────

    def set_baseline(self, zone_id: str, description: str):
        """Set or update baseline for a zone."""
        self._baselines[zone_id] = BaselineEntry(
            zone_id=zone_id,
            description=description,
        )
        logger.info("[ChangeDetect] Baseline set for '%s'", zone_id)

    def get_baseline(self, zone_id: str) -> BaselineEntry | None:
        return self._baselines.get(zone_id)

    def has_baseline(self, zone_id: str) -> bool:
        return zone_id in self._baselines

    def clear_baseline(self, zone_id: str):
        if zone_id in self._baselines:
            del self._baselines[zone_id]

    def clear_all(self):
        self._baselines.clear()
        self._false_positives.clear()

    @property
    def baseline_count(self) -> int:
        return len(self._baselines)

    def should_refresh_baseline(self, zone_id: str) -> bool:
        """Check if baseline is older than BASELINE_UPDATE_HOURS."""
        entry = self._baselines.get(zone_id)
        if not entry:
            return True
        age_hours = (time.time() - entry.timestamp) / 3600
        return age_hours >= BASELINE_UPDATE_HOURS

    # ── Comparison ────────────────────────────────────────────────────────

    def compare(self, zone_id: str, current_description: str) -> ChangeResult:
        """Compare current observation against baseline.

        Returns ChangeResult indicating what changed.
        """
        baseline = self._baselines.get(zone_id)
        if not baseline:
            return ChangeResult(changed=False, description="no baseline")

        base_lower = baseline.description.lower()
        curr_lower = current_description.lower()

        # Check for intrusion (person/vehicle appeared)
        intrusion_keywords = ["person", "intruder", "someone", "figure"]
        for kw in intrusion_keywords:
            if kw in curr_lower and kw not in base_lower:
                return ChangeResult(
                    changed=True,
                    change_type=ChangeType.INTRUSION,
                    description=f"{kw} detected (not in baseline)",
                    confidence=0.9,
                )

        # Check for state changes (open/closed, on/off)
        state_pairs = [
            ("open", "closed"),
            ("on", "off"),
            ("locked", "unlocked"),
            ("present", "absent"),
        ]
        for state_a, state_b in state_pairs:
            base_a = state_a in base_lower
            base_b = state_b in base_lower
            curr_a = state_a in curr_lower
            curr_b = state_b in curr_lower

            if (base_a and curr_b) or (base_b and curr_a):
                return ChangeResult(
                    changed=True,
                    change_type=ChangeType.STATE_CHANGED,
                    description=f"state flip: {state_a}/{state_b}",
                    confidence=0.8,
                )

        # Check for new objects
        new_keywords = ["new", "appeared", "added", "placed"]
        for kw in new_keywords:
            if kw in curr_lower and kw not in base_lower:
                return ChangeResult(
                    changed=True,
                    change_type=ChangeType.NEW_OBJECT,
                    description=f"new element detected ({kw})",
                    confidence=0.6,
                )

        # Check for missing items
        missing_keywords = ["missing", "removed", "gone", "disappeared"]
        for kw in missing_keywords:
            if kw in curr_lower and kw not in base_lower:
                return ChangeResult(
                    changed=True,
                    change_type=ChangeType.MISSING,
                    description=f"item may be missing ({kw})",
                    confidence=0.6,
                )

        # Check for vehicle
        if "vehicle" in curr_lower and "vehicle" not in base_lower:
            return ChangeResult(
                changed=True,
                change_type=ChangeType.NEW_OBJECT,
                description="vehicle detected (not in baseline)",
                confidence=0.8,
            )

        return ChangeResult(changed=False, change_type=ChangeType.UNCHANGED)

    # ── False positive tracking ───────────────────────────────────────────

    def record_sensor_only_trigger(self, zone_id: str):
        """Record when a sensor triggers but VLM says CLEAR (possible false positive)."""
        now = time.time()
        rec = self._false_positives.get(zone_id)
        if not rec:
            rec = FalsePositiveRecord(zone_id=zone_id)
            self._false_positives[zone_id] = rec

        # Decay counter if enough time passed
        if rec.last_trigger > 0 and (now - rec.last_trigger) > FALSE_POSITIVE_DECAY_S:
            rec.trigger_count = 0
            rec.suppressed = False

        rec.trigger_count += 1
        rec.last_trigger = now

        if rec.trigger_count >= FALSE_POSITIVE_THRESHOLD:
            rec.suppressed = True
            logger.info(
                "[ChangeDetect] Zone '%s' marked as false positive " "(%d sensor-only triggers)",
                zone_id,
                rec.trigger_count,
            )

    def is_false_positive_zone(self, zone_id: str) -> bool:
        """Check if a zone has been flagged as generating false positives."""
        rec = self._false_positives.get(zone_id)
        if not rec:
            return False
        # Check decay
        if rec.last_trigger > 0:
            if (time.time() - rec.last_trigger) > FALSE_POSITIVE_DECAY_S:
                rec.suppressed = False
                rec.trigger_count = 0
        return rec.suppressed

    def reset_false_positive(self, zone_id: str):
        """Reset false positive status for a zone."""
        if zone_id in self._false_positives:
            del self._false_positives[zone_id]

    def false_positive_zones(self) -> list[str]:
        """Return list of zones currently flagged as false positive."""
        return [zone_id for zone_id, rec in self._false_positives.items() if rec.suppressed]
