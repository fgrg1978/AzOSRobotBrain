# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""GPS missions and geofencing — Phase AA.

Manages GPS-based patrol routes with real lat/lon coordinates and
enforces geofence boundaries to keep the robot within a defined area.

Usage:
    geo = Geofence(boundary=[(lat1,lon1), (lat2,lon2), ...])
    geo.check(lat, lon)  # True if inside

    mission = GpsMission()
    mission.add_waypoint(lat, lon, action="SCAN_360")
    next_wp = mission.next_waypoint()
"""

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger("brain.gps")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000  # mean Earth radius
DEG7_TO_DEG = 1e-7  # brain_protocol uses degrees × 10^7
GEOFENCE_MARGIN_M = 5.0  # warn when this close to boundary
WAYPOINT_REACH_M = 3.0  # close enough to consider waypoint reached
GPS_STALE_S = 10.0  # GPS fix older than this is stale


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class GpsPosition:
    """GPS position in decimal degrees."""

    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    fix: int = 0  # 0=none, 2=2D, 3=3D
    satellites: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_deg7(
        cls, lat_deg7: int, lon_deg7: int, alt_mm: int = 0, fix: int = 0, sats: int = 0
    ) -> "GpsPosition":
        """Create from brain protocol format (degrees × 10^7)."""
        return cls(
            lat=lat_deg7 * DEG7_TO_DEG,
            lon=lon_deg7 * DEG7_TO_DEG,
            alt_m=alt_mm / 1000.0,
            fix=fix,
            satellites=sats,
        )

    def has_fix(self) -> bool:
        return self.fix >= 2 and self.satellites >= 4

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > GPS_STALE_S

    def distance_to(self, other: "GpsPosition") -> float:
        """Haversine distance in meters."""
        return haversine_m(self.lat, self.lon, other.lat, other.lon)


@dataclass
class GpsWaypoint:
    """A GPS waypoint with an optional action at arrival."""

    lat: float
    lon: float
    alt_m: float = 0.0
    action: str = ""  # skill to execute on arrival (SCAN_360, etc.)
    label: str = ""
    reached: bool = False


@dataclass
class GeofenceViolation:
    """Details of a geofence violation."""

    distance_to_boundary_m: float
    nearest_edge_idx: int
    inside: bool


# ---------------------------------------------------------------------------
# Geofence
# ---------------------------------------------------------------------------
class Geofence:
    """Polygon geofence — keeps robot inside a defined boundary."""

    def __init__(self, boundary: list[tuple[float, float]] | None = None):
        self._boundary = boundary or []
        self._enabled = len(self._boundary) >= 3

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def vertex_count(self) -> int:
        return len(self._boundary)

    def set_boundary(self, boundary: list[tuple[float, float]]):
        """Set geofence boundary as list of (lat, lon) vertices."""
        self._boundary = boundary
        self._enabled = len(boundary) >= 3

    def check(self, lat: float, lon: float) -> bool:
        """Check if position is inside the geofence. Returns True if inside."""
        if not self._enabled:
            return True  # no fence = always inside
        return _point_in_polygon(lat, lon, self._boundary)

    def distance_to_boundary_m(self, lat: float, lon: float) -> float:
        """Approximate distance to nearest fence edge (meters)."""
        if not self._enabled:
            return float("inf")
        min_dist = float("inf")
        n = len(self._boundary)
        for i in range(n):
            j = (i + 1) % n
            d = _point_to_segment_m(
                lat,
                lon,
                self._boundary[i][0],
                self._boundary[i][1],
                self._boundary[j][0],
                self._boundary[j][1],
            )
            if d < min_dist:
                min_dist = d
        return min_dist

    def check_with_margin(self, lat: float, lon: float) -> GeofenceViolation:
        """Check position with margin warning."""
        inside = self.check(lat, lon)
        dist = self.distance_to_boundary_m(lat, lon)
        return GeofenceViolation(
            distance_to_boundary_m=dist,
            nearest_edge_idx=0,
            inside=inside,
        )


# ---------------------------------------------------------------------------
# GpsMission
# ---------------------------------------------------------------------------
class GpsMission:
    """GPS waypoint mission — navigate a sequence of lat/lon waypoints."""

    def __init__(self):
        self._waypoints: list[GpsWaypoint] = []
        self._current_idx: int = 0
        self._loop: bool = True
        self._laps: int = 0
        self._active: bool = False

    @property
    def waypoint_count(self) -> int:
        return len(self._waypoints)

    @property
    def current_index(self) -> int:
        return self._current_idx

    @property
    def laps(self) -> int:
        return self._laps

    @property
    def active(self) -> bool:
        return self._active

    def add_waypoint(
        self,
        lat: float,
        lon: float,
        alt_m: float = 0.0,
        action: str = "",
        label: str = "",
    ):
        """Add a waypoint to the mission."""
        self._waypoints.append(
            GpsWaypoint(
                lat=lat,
                lon=lon,
                alt_m=alt_m,
                action=action,
                label=label,
            )
        )

    def clear(self):
        """Clear all waypoints."""
        self._waypoints.clear()
        self._current_idx = 0
        self._laps = 0

    def start(self, loop: bool = True):
        """Start the mission."""
        self._active = True
        self._loop = loop
        self._current_idx = 0
        self._laps = 0

    def stop(self):
        """Stop the mission."""
        self._active = False

    def current_waypoint(self) -> GpsWaypoint | None:
        """Get current target waypoint."""
        if not self._active or not self._waypoints:
            return None
        if self._current_idx >= len(self._waypoints):
            return None
        return self._waypoints[self._current_idx]

    def check_reached(self, pos: GpsPosition) -> GpsWaypoint | None:
        """Check if current waypoint is reached. Advances if so.
        Returns the reached waypoint, or None."""
        wp = self.current_waypoint()
        if wp is None:
            return None
        dist = haversine_m(pos.lat, pos.lon, wp.lat, wp.lon)
        if dist > WAYPOINT_REACH_M:
            return None

        wp.reached = True
        self._current_idx += 1

        if self._current_idx >= len(self._waypoints):
            self._laps += 1
            if self._loop:
                self._current_idx = 0
                for w in self._waypoints:
                    w.reached = False
            else:
                self._active = False

        return wp

    def distance_to_current(self, pos: GpsPosition) -> float:
        """Distance to current waypoint in meters."""
        wp = self.current_waypoint()
        if wp is None:
            return 0.0
        return haversine_m(pos.lat, pos.lon, wp.lat, wp.lon)

    def bearing_to_current(self, pos: GpsPosition) -> float:
        """Bearing to current waypoint in degrees (0=N, 90=E)."""
        wp = self.current_waypoint()
        if wp is None:
            return 0.0
        return bearing_deg(pos.lat, pos.lon, wp.lat, wp.lon)


# ---------------------------------------------------------------------------
# GPS math helpers
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two GPS coordinates in meters."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees (0-360)."""
    d_lon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _point_in_polygon(
    lat: float,
    lon: float,
    polygon: list[tuple[float, float]],
) -> bool:
    """Ray-casting algorithm for point-in-polygon test.
    polygon is list of (lat, lon) tuples."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if ((lon_i > lon) != (lon_j > lon)) and (
            lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
        ):
            inside = not inside
        j = i
    return inside


def _point_to_segment_m(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Approximate distance from point to line segment (meters)."""
    d1 = haversine_m(lat, lon, lat1, lon1)
    d2 = haversine_m(lat, lon, lat2, lon2)
    seg = haversine_m(lat1, lon1, lat2, lon2)
    if seg == 0:
        return d1
    # Project point onto segment
    t = max(
        0,
        min(
            1,
            ((lat - lat1) * (lat2 - lat1) + (lon - lon1) * (lon2 - lon1))
            / ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2 + 1e-12),
        ),
    )
    proj_lat = lat1 + t * (lat2 - lat1)
    proj_lon = lon1 + t * (lon2 - lon1)
    return haversine_m(lat, lon, proj_lat, proj_lon)
