# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception.star_tracker — celestial navigation."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from perception.star_tracker import (
    StarTracker,
    DetectedStar,
    Attitude,
    detect_stars,
    _quad_hash,
    NAVIGATION_STARS,
)


class TestNavigationCatalog:
    def test_catalog_not_empty(self):
        assert len(NAVIGATION_STARS) >= 40

    def test_catalog_format(self):
        for name, ra, dec, mag in NAVIGATION_STARS:
            assert isinstance(name, str)
            assert 0 <= ra <= 360
            assert -90 <= dec <= 90
            assert -2 < mag < 3

    def test_sirius_is_brightest(self):
        mags = [(s[0], s[3]) for s in NAVIGATION_STARS]
        brightest = min(mags, key=lambda t: t[1])
        assert brightest[0] == "Sirius"

    def test_polaris_near_pole(self):
        polaris = [s for s in NAVIGATION_STARS if s[0] == "Polaris"]
        assert len(polaris) == 1
        assert polaris[0][2] > 89  # Dec > 89°


class TestStarDetection:
    def test_blank_image_no_stars(self):
        black = np.zeros((480, 640), dtype=np.uint8)
        stars = detect_stars(black)
        assert len(stars) == 0

    def test_single_bright_point(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[100, 100] = 255
        img[99:102, 99:102] = 200  # small bright region
        stars = detect_stars(img, max_stars=10)
        assert len(stars) >= 1
        # Star should be near (100, 100)
        assert abs(stars[0].x - 100) < 5
        assert abs(stars[0].y - 100) < 5

    def test_multiple_stars(self):
        img = np.zeros((400, 400), dtype=np.uint8)
        positions = [(100, 100), (200, 200), (300, 100), (150, 300)]
        for x, y in positions:
            img[y - 2 : y + 3, x - 2 : x + 3] = 255
        stars = detect_stars(img, max_stars=20)
        assert len(stars) >= 3  # at least most should be detected

    def test_brightness_normalized(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50, 50] = 255
        img[150, 150] = 128
        stars = detect_stars(img)
        if len(stars) >= 1:
            assert 0 <= stars[0].brightness <= 1.0

    def test_max_stars_limit(self):
        img = np.zeros((400, 400), dtype=np.uint8)
        for i in range(20):
            x, y = 50 + i * 15, 200
            img[y - 1 : y + 2, x - 1 : x + 2] = 255
        stars = detect_stars(img, max_stars=5)
        assert len(stars) <= 5


class TestQuadHash:
    def test_four_points_returns_hash(self):
        pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        h = _quad_hash(pts)
        assert h is not None
        assert len(h) == 2

    def test_degenerate_points_returns_none(self):
        pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
        h = _quad_hash(pts)
        assert h is None

    def test_scale_invariance(self):
        pts1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        pts2 = [(0, 0), (20, 0), (20, 20), (0, 20)]
        h1 = _quad_hash(pts1)
        h2 = _quad_hash(pts2)
        assert h1 is not None and h2 is not None
        # Same shape, different scale → similar hash
        assert abs(h1[0] - h2[0]) < 0.1
        assert abs(h1[1] - h2[1]) < 0.1


class TestStarTracker:
    def test_init_default_catalog(self):
        tracker = StarTracker()
        assert len(tracker.catalog) == len(NAVIGATION_STARS)

    def test_blank_sky_returns_none(self):
        tracker = StarTracker()
        black = np.zeros((480, 640), dtype=np.uint8)
        assert tracker.solve(black) is None

    def test_too_few_stars_returns_none(self):
        tracker = StarTracker()
        # Image with only 2 bright points
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50, 50] = 255
        img[150, 150] = 255
        assert tracker.solve(img) is None

    def test_attitude_fields(self):
        att = Attitude(
            ra_deg=180.0, dec_deg=45.0, roll_deg=10.0, fov_deg=30.0, n_matches=5, confidence=0.8
        )
        assert att.ra_deg == 180.0
        assert att.dec_deg == 45.0
        assert att.n_matches == 5
        assert 0 <= att.confidence <= 1
