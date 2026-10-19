# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""
Star Tracker — celestial navigation for night flight drones.

Uses star pattern recognition to determine drone attitude (orientation)
when GPS is degraded or unavailable. Works with upward-facing camera.

Pipeline:
    1. Detect bright points in night sky image (star candidates)
    2. Form geometric quads (groups of 4 stars)
    3. Match quads against star catalog using scale-invariant hashes
    4. Solve for camera attitude (RA, Dec, rotation)

This is a simplified port of skymap's plate solving algorithm adapted
for real-time drone navigation. Full precision requires astropy for
coordinate transforms; a lightweight fallback uses pure numpy.

Usage:
    from perception.star_tracker import StarTracker
    tracker = StarTracker()
    attitude = tracker.solve(image_gray)
    if attitude:
        print(f"Pointing: RA={attitude.ra_deg:.2f} Dec={attitude.dec_deg:.2f}")

Requires: numpy, scipy (core), astropy (optional, for full precision)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, center_of_mass, label

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Star detection
GAUSSIAN_SIGMA = 1.5
MEDIAN_KERNEL_SIZE = 31
LOCAL_MAX_SIZE = 5
MAX_STARS = 200
MIN_BRIGHTNESS = 0.15  # fraction of max pixel value

# Quad matching
QUAD_MATCH_TOLERANCE = 0.02  # ratio tolerance for hash matching
MIN_MATCHES = 3  # minimum quad matches for a valid solve

# Built-in catalog: 50 brightest navigation stars (mag < 2.5)
# Format: (name, RA_degrees, Dec_degrees, magnitude)
NAVIGATION_STARS = [
    ("Sirius", 101.287, -16.716, -1.46),
    ("Canopus", 95.988, -52.696, -0.74),
    ("Arcturus", 213.915, 19.182, -0.05),
    ("Vega", 279.235, 38.784, 0.03),
    ("Capella", 79.172, 45.998, 0.08),
    ("Rigel", 78.634, -8.202, 0.13),
    ("Procyon", 114.826, 5.225, 0.34),
    ("Betelgeuse", 88.793, 7.407, 0.42),
    ("Achernar", 24.429, -57.237, 0.46),
    ("Hadar", 210.956, -60.373, 0.61),
    ("Altair", 297.696, 8.868, 0.77),
    ("Acrux", 186.650, -63.099, 0.76),
    ("Aldebaran", 68.980, 16.509, 0.85),
    ("Antares", 247.352, -26.432, 0.96),
    ("Spica", 201.298, -11.161, 0.97),
    ("Pollux", 116.329, 28.026, 1.14),
    ("Fomalhaut", 344.413, -29.622, 1.16),
    ("Deneb", 310.358, 45.280, 1.25),
    ("Mimosa", 191.930, -59.689, 1.25),
    ("Regulus", 152.093, 11.967, 1.35),
    ("Adhara", 104.656, -28.972, 1.50),
    ("Castor", 113.650, 31.889, 1.58),
    ("Shaula", 263.402, -37.104, 1.62),
    ("Gacrux", 187.791, -57.113, 1.63),
    ("Bellatrix", 81.283, 6.350, 1.64),
    ("Elnath", 81.573, 28.608, 1.65),
    ("Miaplacidus", 138.300, -69.717, 1.68),
    ("Alnilam", 84.053, -1.202, 1.69),
    ("Alnair", 332.058, -46.961, 1.74),
    ("Alnitak", 85.190, -1.943, 1.77),
    ("Alioth", 193.507, 55.960, 1.77),
    ("Dubhe", 165.932, 61.751, 1.79),
    ("Mirfak", 51.081, 49.861, 1.80),
    ("Wezen", 107.098, -26.393, 1.84),
    ("Kaus Australis", 276.043, -34.384, 1.85),
    ("Sargas", 264.330, -42.998, 1.87),
    ("Avior", 125.629, -59.509, 1.86),
    ("Alkaid", 206.885, 49.313, 1.86),
    ("Menkalinan", 89.882, 44.948, 1.90),
    ("Atria", 252.166, -69.028, 1.92),
    ("Alhena", 99.428, 16.399, 1.93),
    ("Peacock", 306.412, -56.735, 1.94),
    ("Mirzam", 95.675, -17.956, 1.98),
    ("Alphard", 141.897, -8.659, 1.98),
    ("Polaris", 37.954, 89.264, 2.02),
    ("Hamal", 31.793, 23.462, 2.00),
    ("Diphda", 10.897, -17.987, 2.02),
    ("Nunki", 283.816, -26.297, 2.05),
    ("Menkent", 211.671, -36.370, 2.06),
    ("Alpheratz", 2.097, 29.091, 2.06),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DetectedStar:
    """A star candidate detected in an image."""

    x: float  # pixel X
    y: float  # pixel Y
    brightness: float  # relative brightness (0-1)


@dataclass
class Attitude:
    """Solved camera attitude in celestial coordinates."""

    ra_deg: float  # Right Ascension (0-360)
    dec_deg: float  # Declination (-90 to +90)
    roll_deg: float  # Camera roll angle
    fov_deg: float  # Field of view (estimated)
    n_matches: int  # Number of matched quads
    confidence: float  # Match confidence (0-1)


# ---------------------------------------------------------------------------
# Star detection
# ---------------------------------------------------------------------------


def detect_stars(image_gray: np.ndarray, max_stars: int = MAX_STARS) -> List[DetectedStar]:
    """Detect star-like bright points in a grayscale night sky image.

    Args:
        image_gray: 2D numpy array (uint8 or float), grayscale.
        max_stars: maximum number of stars to return.

    Returns:
        List of DetectedStar sorted by brightness (brightest first).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("star_tracker.detect_stars requires scipy")

    img = image_gray.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    # Smooth to reduce noise
    smoothed = gaussian_filter(img, sigma=GAUSSIAN_SIGMA)

    # Background subtraction
    background = median_filter(smoothed, size=MEDIAN_KERNEL_SIZE)
    subtracted = smoothed - background
    subtracted = np.clip(subtracted, 0, None)

    # Threshold
    threshold = MIN_BRIGHTNESS * subtracted.max() if subtracted.max() > 0 else 0
    mask = subtracted > threshold

    # Local maxima
    local_max = maximum_filter(subtracted, size=LOCAL_MAX_SIZE)
    peaks = (subtracted == local_max) & mask

    # Label connected regions
    labeled, n_features = label(peaks)
    if n_features == 0:
        return []

    # Centroid refinement
    stars = []
    for i in range(1, n_features + 1):
        region = labeled == i
        brightness = subtracted[region].sum()
        cy, cx = center_of_mass(subtracted, labeled, i)
        stars.append(DetectedStar(x=cx, y=cy, brightness=brightness))

    # Sort by brightness, take top N
    stars.sort(key=lambda s: s.brightness, reverse=True)
    max_bright = stars[0].brightness if stars else 1.0
    for s in stars:
        s.brightness /= max_bright  # normalize to 0-1

    return stars[:max_stars]


# ---------------------------------------------------------------------------
# Quad generation and hashing
# ---------------------------------------------------------------------------


def _quad_hash(stars_4: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Compute a scale/rotation-invariant hash from 4 star positions.

    The hash is based on the ratios of inter-star distances, following
    the Astrometry.net approach (Lang et al. 2010).

    Returns (hash_x, hash_y) or None if degenerate.
    """
    # Sort by distance from centroid
    cx = sum(s[0] for s in stars_4) / 4.0
    cy = sum(s[1] for s in stars_4) / 4.0
    indexed = [(i, (s[0] - cx) ** 2 + (s[1] - cy) ** 2) for i, s in enumerate(stars_4)]
    indexed.sort(key=lambda t: t[1])

    A = stars_4[indexed[3][0]]  # farthest from centroid
    B = stars_4[indexed[2][0]]  # second farthest

    # Distance AB is the scale reference
    dAB = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    if dAB < 1e-6:
        return None

    C = stars_4[indexed[1][0]]
    D = stars_4[indexed[0][0]]

    # Project C and D onto AB coordinate system
    ux = (B[0] - A[0]) / dAB
    uy = (B[1] - A[1]) / dAB

    cx_proj = ((C[0] - A[0]) * ux + (C[1] - A[1]) * uy) / dAB
    cy_proj = (-(C[0] - A[0]) * uy + (C[1] - A[1]) * ux) / dAB
    dx_proj = ((D[0] - A[0]) * ux + (D[1] - A[1]) * uy) / dAB
    dy_proj = (-(D[0] - A[0]) * uy + (D[1] - A[1]) * ux) / dAB

    return (cx_proj + dx_proj, cy_proj + dy_proj)


def build_catalog_index(stars: List[Tuple[float, float, float]], fov_deg: float = 30.0):
    """Build quad index from catalog stars visible in a given FOV.

    Args:
        stars: list of (ra_deg, dec_deg, magnitude).
        fov_deg: approximate field of view to consider.

    Returns:
        List of (hash, star_indices) tuples.
    """
    # For a simplified implementation, we build quads from nearby stars
    # A full implementation would use KD-trees on the celestial sphere
    index = []
    n = min(len(stars), 30)  # use brightest 30

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    positions = [
                        (stars[i][0], stars[i][1]),
                        (stars[j][0], stars[j][1]),
                        (stars[k][0], stars[k][1]),
                        (stars[l][0], stars[l][1]),
                    ]
                    h = _quad_hash(positions)
                    if h is not None:
                        index.append((h, (i, j, k, l)))

    return index


# ---------------------------------------------------------------------------
# Plate solver (simplified)
# ---------------------------------------------------------------------------


class StarTracker:
    """Star pattern matcher for attitude determination.

    Uses geometric quad hashing to match detected stars against a
    built-in catalog of bright navigation stars. Suitable for wide-field
    cameras (FOV > 10°) at moderate exposure times.
    """

    def __init__(self, catalog=None):
        """Initialize with a star catalog.

        Args:
            catalog: list of (name, ra, dec, mag) tuples.
                     Defaults to NAVIGATION_STARS (50 brightest).
        """
        self.catalog = catalog or NAVIGATION_STARS
        self._catalog_positions = [(s[1], s[2], s[3]) for s in self.catalog]
        self._index = None  # lazy build

    def solve(self, image_gray: np.ndarray, fov_hint_deg: float = 30.0) -> Optional[Attitude]:
        """Attempt to determine camera attitude from a night sky image.

        Args:
            image_gray: grayscale image (uint8 or float).
            fov_hint_deg: approximate FOV to narrow the search.

        Returns:
            Attitude if solved, None if no match found.
        """
        stars = detect_stars(image_gray)
        if len(stars) < 4:
            return None

        # Build catalog index if needed
        if self._index is None:
            self._index = build_catalog_index(self._catalog_positions, fov_hint_deg)

        # Build quads from detected stars
        n = min(len(stars), 15)
        image_quads = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        positions = [
                            (stars[i].x, stars[i].y),
                            (stars[j].x, stars[j].y),
                            (stars[k].x, stars[k].y),
                            (stars[l].x, stars[l].y),
                        ]
                        h = _quad_hash(positions)
                        if h is not None:
                            image_quads.append((h, (i, j, k, l)))

        # Match against catalog
        matches = []
        for img_hash, img_idx in image_quads:
            for cat_hash, cat_idx in self._index:
                dx = img_hash[0] - cat_hash[0]
                dy = img_hash[1] - cat_hash[1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < QUAD_MATCH_TOLERANCE:
                    matches.append((img_idx, cat_idx, dist))

        if len(matches) < MIN_MATCHES:
            return None

        # Use best match cluster to estimate attitude
        matches.sort(key=lambda m: m[2])
        best = matches[0]
        cat_stars = best[1]

        # Center of matched catalog stars = approximate pointing direction
        ra_center = np.mean([self._catalog_positions[i][0] for i in cat_stars])
        dec_center = np.mean([self._catalog_positions[i][1] for i in cat_stars])

        confidence = min(1.0, len(matches) / 10.0)

        return Attitude(
            ra_deg=ra_center,
            dec_deg=dec_center,
            roll_deg=0.0,  # TODO: compute from affine transform
            fov_deg=fov_hint_deg,
            n_matches=len(matches),
            confidence=confidence,
        )
