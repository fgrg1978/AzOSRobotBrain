# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Property status report — quick scan of all zones of interest.

Triggered by Telegram `/status` or an "is everything OK?" question.
Robot does a fast scan of zones only (skipping normal waypoints),
takes a photo at each zone, and reports state via VLM.

Usage:
    reporter = StatusReporter(zone_manager, mapper)
    report = reporter.build_report(waypoints, zone_states)
    # report = "Front door: CLOSED. Side window: OPEN."
"""

import logging
import time
from dataclasses import dataclass, field

from planner.mapper import Waypoint

logger = logging.getLogger("brain.status_report")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATUS_SCAN_TIMEOUT_S = 120  # max time for full status scan
STATUS_MAX_ZONES = 20  # don't scan more than this many zones


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ZoneStatus:
    """Status of one zone from a scan."""

    waypoint_label: str
    zone_type: str
    x_mm: int
    y_mm: int
    vlm_description: str = ""
    state_summary: str = ""  # short: "CLOSED", "OPEN", "CLEAR", etc.
    photo_path: str = ""  # path to saved photo
    timestamp: float = field(default_factory=time.time)


@dataclass
class PropertyReport:
    """Complete property status report."""

    zones: list[ZoneStatus] = field(default_factory=list)
    scan_time_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    complete: bool = False


# ---------------------------------------------------------------------------
# StatusReporter
# ---------------------------------------------------------------------------


class StatusReporter:
    """Generates property status reports from zone scans."""

    def __init__(self):
        self._last_report: PropertyReport | None = None

    @property
    def last_report(self) -> PropertyReport | None:
        return self._last_report

    def build_report(
        self,
        waypoints: list[Waypoint],
        scan_time_s: float = 0.0,
    ) -> PropertyReport:
        """Build a report from waypoints that have zone data.

        Uses each waypoint's last_state for the status.
        """
        zones = []
        for wp in waypoints:
            if not wp.zone_type:
                continue
            if len(zones) >= STATUS_MAX_ZONES:
                break
            summary = self._extract_state_summary(wp.last_state, wp.zone_type)
            zones.append(
                ZoneStatus(
                    waypoint_label=wp.label or f"{wp.zone_type}_{wp.x_mm}",
                    zone_type=wp.zone_type,
                    x_mm=wp.x_mm,
                    y_mm=wp.y_mm,
                    vlm_description=wp.last_state,
                    state_summary=summary,
                )
            )

        report = PropertyReport(
            zones=zones,
            scan_time_s=scan_time_s,
            complete=True,
        )
        self._last_report = report
        return report

    def format_text(self, report: PropertyReport) -> str:
        """Format a report as human-readable text for Telegram."""
        if not report.zones:
            return "No zones of interest mapped yet. Run mapping first."

        lines = [f"🏠 Property Status ({len(report.zones)} zones):"]
        for z in report.zones:
            icon = _zone_icon(z.zone_type)
            label = z.waypoint_label or z.zone_type
            state = z.state_summary or "unknown"
            lines.append(f"  {icon} {label}: {state.upper()}")

        if report.scan_time_s > 0:
            lines.append(f"\nScan completed in {report.scan_time_s:.1f}s")
        return "\n".join(lines)

    def get_zone_route(self, waypoints: list[Waypoint]) -> list[Waypoint]:
        """Return only zone waypoints for a quick status scan."""
        return [wp for wp in waypoints if wp.zone_type][:STATUS_MAX_ZONES]

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_state_summary(vlm_description: str, zone_type: str) -> str:
        """Extract a short state summary from VLM description."""
        if not vlm_description:
            return "not scanned"

        desc = vlm_description.lower()

        # Check for common states
        state_keywords = [
            ("open", "OPEN"),
            ("closed", "CLOSED"),
            ("locked", "LOCKED"),
            ("unlocked", "UNLOCKED"),
            ("person", "PERSON DETECTED"),
            ("vehicle", "VEHICLE PRESENT"),
            ("fire", "FIRE DETECTED"),
            ("smoke", "SMOKE DETECTED"),
            ("clear", "CLEAR"),
            ("empty", "EMPTY"),
            ("occupied", "OCCUPIED"),
        ]

        for keyword, summary in state_keywords:
            if keyword in desc:
                return summary

        return "OK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zone_icon(zone_type: str) -> str:
    """Return an icon for the zone type."""
    icons = {
        "door": "🚪",
        "window": "🪟",
        "gate": "🚧",
        "garage": "🏗",
        "entrance": "🚪",
        "exit": "🚪",
        "driveway": "🛣",
        "open_area": "🌿",
        "stairs": "🪜",
        "fence": "🏗",
    }
    return icons.get(zone_type, "📍")
