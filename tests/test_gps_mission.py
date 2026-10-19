# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/gps_mission.py — GPS missions and geofencing."""

import math
from planner.gps_mission import (
    GpsPosition,
    GpsWaypoint,
    Geofence,
    GpsMission,
    haversine_m,
    bearing_deg,
    EARTH_RADIUS_M,
    WAYPOINT_REACH_M,
    GPS_STALE_S,
)


class TestConstants:
    def test_earth_radius(self):
        assert EARTH_RADIUS_M > 6_000_000

    def test_waypoint_reach(self):
        assert WAYPOINT_REACH_M > 0

    def test_gps_stale(self):
        assert GPS_STALE_S > 0


class TestGpsPosition:
    def test_from_deg7(self):
        p = GpsPosition.from_deg7(405000000, -37000000)
        assert abs(p.lat - 40.5) < 0.001
        assert abs(p.lon - (-3.7)) < 0.001

    def test_has_fix(self):
        p = GpsPosition(fix=3, satellites=8)
        assert p.has_fix()
        p2 = GpsPosition(fix=0, satellites=0)
        assert not p2.has_fix()

    def test_distance_to(self):
        a = GpsPosition(lat=40.0, lon=-3.0)
        b = GpsPosition(lat=40.001, lon=-3.0)
        d = a.distance_to(b)
        assert 100 < d < 200  # ~111m per 0.001 deg latitude


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_m(40, -3, 40, -3) == 0

    def test_known_distance(self):
        # Madrid to Barcelona ~500km
        d = haversine_m(40.4168, -3.7038, 41.3851, 2.1734)
        assert 400_000 < d < 700_000

    def test_small_distance(self):
        d = haversine_m(40.0, -3.0, 40.00001, -3.0)
        assert 0 < d < 5  # ~1.1m


class TestBearing:
    def test_north(self):
        b = bearing_deg(40.0, -3.0, 41.0, -3.0)
        assert abs(b - 0) < 5  # roughly north

    def test_east(self):
        b = bearing_deg(40.0, -3.0, 40.0, -2.0)
        assert abs(b - 90) < 10


class TestGeofence:
    def _square_fence(self):
        # 100m × 100m square centered at (40.0, -3.0)
        d = 0.001  # ~111m
        return Geofence(
            boundary=[
                (40.0 - d, -3.0 - d),
                (40.0 - d, -3.0 + d),
                (40.0 + d, -3.0 + d),
                (40.0 + d, -3.0 - d),
            ]
        )

    def test_inside(self):
        gf = self._square_fence()
        assert gf.check(40.0, -3.0)

    def test_outside(self):
        gf = self._square_fence()
        assert not gf.check(41.0, -3.0)

    def test_disabled_always_inside(self):
        gf = Geofence()
        assert gf.check(0, 0)

    def test_vertex_count(self):
        gf = self._square_fence()
        assert gf.vertex_count == 4

    def test_distance_to_boundary(self):
        gf = self._square_fence()
        d = gf.distance_to_boundary_m(40.0, -3.0)
        assert d > 0

    def test_check_with_margin(self):
        gf = self._square_fence()
        v = gf.check_with_margin(40.0, -3.0)
        assert v.inside


class TestGpsMission:
    def test_empty_mission(self):
        m = GpsMission()
        assert m.waypoint_count == 0
        assert not m.active

    def test_add_waypoints(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0, action="SCAN_360")
        m.add_waypoint(40.001, -3.0)
        assert m.waypoint_count == 2

    def test_start_stop(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0)
        m.start()
        assert m.active
        m.stop()
        assert not m.active

    def test_current_waypoint(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0, label="wp1")
        m.start()
        wp = m.current_waypoint()
        assert wp is not None
        assert wp.label == "wp1"

    def test_check_reached(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0)
        m.add_waypoint(40.001, -3.0)
        m.start()
        pos = GpsPosition(lat=40.0, lon=-3.0)
        wp = m.check_reached(pos)
        assert wp is not None
        assert m.current_index == 1

    def test_loop(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0)
        m.start(loop=True)
        pos = GpsPosition(lat=40.0, lon=-3.0)
        m.check_reached(pos)
        assert m.current_index == 0  # looped back
        assert m.laps == 1

    def test_no_loop(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0)
        m.start(loop=False)
        pos = GpsPosition(lat=40.0, lon=-3.0)
        m.check_reached(pos)
        assert not m.active

    def test_distance_to_current(self):
        m = GpsMission()
        m.add_waypoint(40.001, -3.0)
        m.start()
        pos = GpsPosition(lat=40.0, lon=-3.0)
        d = m.distance_to_current(pos)
        assert d > 50

    def test_clear(self):
        m = GpsMission()
        m.add_waypoint(40.0, -3.0)
        m.clear()
        assert m.waypoint_count == 0
