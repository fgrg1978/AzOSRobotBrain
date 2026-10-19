# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/status_report.py — property status reports."""

import time

from planner.status_report import (
    StatusReporter,
    ZoneStatus,
    PropertyReport,
    STATUS_SCAN_TIMEOUT_S,
    STATUS_MAX_ZONES,
)
from planner.mapper import Waypoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wp(x=0, y=0, label="", zone_type="", last_state=""):
    return Waypoint(
        x_mm=x,
        y_mm=y,
        heading_cdeg=0,
        label=label,
        zone_type=zone_type,
        last_state=last_state,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_scan_timeout_positive(self):
        assert STATUS_SCAN_TIMEOUT_S > 0

    def test_max_zones_positive(self):
        assert STATUS_MAX_ZONES > 0


# ---------------------------------------------------------------------------
# ZoneStatus
# ---------------------------------------------------------------------------


class TestZoneStatus:
    def test_defaults(self):
        zs = ZoneStatus(
            waypoint_label="puerta",
            zone_type="door",
            x_mm=0,
            y_mm=0,
        )
        assert zs.waypoint_label == "puerta"
        assert zs.state_summary == ""
        assert zs.timestamp > 0


# ---------------------------------------------------------------------------
# PropertyReport
# ---------------------------------------------------------------------------


class TestPropertyReport:
    def test_empty_report(self):
        r = PropertyReport()
        assert r.zones == []
        assert not r.complete


# ---------------------------------------------------------------------------
# StatusReporter — build_report
# ---------------------------------------------------------------------------


class TestStatusReporterBuild:
    def test_empty_waypoints(self):
        sr = StatusReporter()
        report = sr.build_report([])
        assert len(report.zones) == 0
        assert report.complete

    def test_no_zone_waypoints(self):
        sr = StatusReporter()
        report = sr.build_report([_wp(), _wp(x=1000)])
        assert len(report.zones) == 0

    def test_zone_waypoints_included(self):
        sr = StatusReporter()
        wps = [
            _wp(label="puerta_principal", zone_type="door", last_state="Door closed"),
            _wp(x=1000),  # normal, no zone
            _wp(x=2000, label="ventana_lateral", zone_type="window", last_state="Window closed"),
        ]
        report = sr.build_report(wps)
        assert len(report.zones) == 2
        assert report.zones[0].waypoint_label == "puerta_principal"
        assert report.zones[1].waypoint_label == "ventana_lateral"

    def test_state_summary_extracted(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="door", last_state="White door, closed, clean")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "CLOSED"

    def test_state_summary_open(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="gate", last_state="Gate is open")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "OPEN"

    def test_state_summary_clear(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="driveway", last_state="Clear driveway")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "CLEAR"

    def test_state_summary_person(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="entrance", last_state="Person at entrance")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "PERSON DETECTED"

    def test_state_summary_not_scanned(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="door", last_state="")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "not scanned"

    def test_state_summary_ok_fallback(self):
        sr = StatusReporter()
        wps = [_wp(zone_type="fence", last_state="Metal fence, intact")]
        report = sr.build_report(wps)
        assert report.zones[0].state_summary == "OK"

    def test_max_zones_limit(self):
        sr = StatusReporter()
        wps = [_wp(x=i, zone_type="door", last_state="closed") for i in range(STATUS_MAX_ZONES + 5)]
        report = sr.build_report(wps)
        assert len(report.zones) == STATUS_MAX_ZONES

    def test_scan_time_recorded(self):
        sr = StatusReporter()
        report = sr.build_report([], scan_time_s=15.5)
        assert report.scan_time_s == 15.5

    def test_last_report_stored(self):
        sr = StatusReporter()
        report = sr.build_report([])
        assert sr.last_report is report


# ---------------------------------------------------------------------------
# StatusReporter — format_text
# ---------------------------------------------------------------------------


class TestStatusReporterFormat:
    def test_format_empty(self):
        sr = StatusReporter()
        report = sr.build_report([])
        text = sr.format_text(report)
        assert "no zones" in text.lower()

    def test_format_with_zones(self):
        sr = StatusReporter()
        wps = [
            _wp(label="puerta", zone_type="door", last_state="Door closed"),
            _wp(label="ventana", zone_type="window", last_state="Window open"),
        ]
        report = sr.build_report(wps)
        text = sr.format_text(report)
        assert "puerta" in text
        assert "CLOSED" in text
        assert "ventana" in text
        assert "OPEN" in text
        assert "2 zones" in text

    def test_format_with_scan_time(self):
        sr = StatusReporter()
        report = sr.build_report([_wp(zone_type="door", last_state="ok")], scan_time_s=5.3)
        text = sr.format_text(report)
        assert "5.3s" in text


# ---------------------------------------------------------------------------
# StatusReporter — get_zone_route
# ---------------------------------------------------------------------------


class TestStatusReporterRoute:
    def test_zone_route_only_zones(self):
        sr = StatusReporter()
        wps = [
            _wp(zone_type="door"),
            _wp(),
            _wp(zone_type="window"),
            _wp(),
        ]
        route = sr.get_zone_route(wps)
        assert len(route) == 2

    def test_zone_route_respects_max(self):
        sr = StatusReporter()
        wps = [_wp(x=i, zone_type="door") for i in range(STATUS_MAX_ZONES + 10)]
        route = sr.get_zone_route(wps)
        assert len(route) == STATUS_MAX_ZONES

    def test_zone_route_empty(self):
        sr = StatusReporter()
        assert sr.get_zone_route([]) == []
