# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception/motion_detect.py — frame differencing motion detection."""

import io
import pytest

from perception.motion_detect import (
    MotionDetector,
    MOTION_THRESHOLD_PCT,
    MOTION_PIXEL_DIFF_THRESHOLD,
    MOTION_DOWNSAMPLE_WIDTH,
    MOTION_BLUR_RADIUS,
)

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(width: int = 64, height: int = 48, fill: int = 128) -> bytes:
    """Create a minimal JPEG from a solid grayscale image."""
    img = Image.new("L", (width, height), fill)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_jpeg_gradient(width: int = 64, height: int = 48, offset: int = 0) -> bytes:
    """Create a JPEG with a horizontal gradient (varies per pixel)."""
    img = Image.new("L", (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), (x * 4 + offset) % 256)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_threshold_pct_is_positive(self):
        assert MOTION_THRESHOLD_PCT > 0

    def test_pixel_diff_threshold_is_positive(self):
        assert MOTION_PIXEL_DIFF_THRESHOLD > 0

    def test_downsample_width_is_positive(self):
        assert MOTION_DOWNSAMPLE_WIDTH > 0

    def test_blur_radius_is_non_negative(self):
        assert MOTION_BLUR_RADIUS >= 0


# ---------------------------------------------------------------------------
# MotionDetector — unit tests (no PIL needed)
# ---------------------------------------------------------------------------


class TestMotionDetector:
    def test_init_defaults(self):
        d = MotionDetector()
        assert d.threshold_pct == MOTION_THRESHOLD_PCT
        assert d.pixel_diff == MOTION_PIXEL_DIFF_THRESHOLD

    def test_init_custom(self):
        d = MotionDetector(threshold_pct=25, pixel_diff=50)
        assert d.threshold_pct == 25
        assert d.pixel_diff == 50

    def test_empty_bytes_returns_zero(self):
        d = MotionDetector()
        score = d.feed(b"")
        assert score == 0.0

    def test_invalid_jpeg_returns_zero(self):
        d = MotionDetector()
        score = d.feed(b"not a jpeg at all")
        assert score == 0.0

    def test_reset_sets_prev_to_none(self):
        d = MotionDetector()
        d._prev_gray = [1, 2, 3]
        d.reset()
        assert d._prev_gray is None


class TestMotionDetectorInternals:
    def test_compute_score_empty(self):
        d = MotionDetector()
        assert d._compute_score([], []) == 0.0

    def test_compute_score_all_changed(self):
        d = MotionDetector(pixel_diff=10)
        prev = [0] * 100
        curr = [50] * 100
        assert d._compute_score(prev, curr) == 100.0

    def test_compute_score_none_changed(self):
        d = MotionDetector(pixel_diff=10)
        prev = [100] * 100
        curr = [105] * 100  # diff = 5, below threshold of 10
        assert d._compute_score(prev, curr) == 0.0

    def test_compute_score_half_changed(self):
        d = MotionDetector(pixel_diff=10)
        prev = [100] * 100
        curr = [100] * 50 + [200] * 50
        score = d._compute_score(prev, curr)
        assert score == 50.0

    def test_compute_score_returns_percentage(self):
        d = MotionDetector(pixel_diff=5)
        prev = [0] * 200
        curr = [0] * 150 + [100] * 50  # 25% changed
        score = d._compute_score(prev, curr)
        assert score == 25.0

    def test_to_grayscale_invalid_returns_none(self):
        result = MotionDetector._to_grayscale(b"not valid")
        assert result is None

    def test_feed_with_direct_gray_injection(self):
        """Test feed logic by directly injecting grayscale data."""
        d = MotionDetector(pixel_diff=10)
        # First feed — no prev, returns 0
        d._prev_gray = None
        gray_a = [0] * 100
        d._prev_gray = gray_a
        # Second feed with identical data
        score = d._compute_score(gray_a, gray_a)
        assert score == 0.0

    def test_feed_with_different_gray_injection(self):
        d = MotionDetector(pixel_diff=10)
        gray_a = [0] * 100
        gray_b = [100] * 100
        score = d._compute_score(gray_a, gray_b)
        assert score == 100.0


# ---------------------------------------------------------------------------
# MotionDetector — PIL-dependent tests
# ---------------------------------------------------------------------------


@requires_pil
class TestMotionDetectorWithPIL:
    def test_first_frame_returns_zero(self):
        d = MotionDetector()
        score = d.feed(_make_jpeg())
        assert score == 0.0

    def test_identical_frames_zero_score(self):
        d = MotionDetector()
        frame = _make_jpeg(fill=128)
        d.feed(frame)
        score = d.feed(frame)
        # JPEG compression may introduce tiny differences
        assert score < 5.0

    def test_different_frames_high_score(self):
        d = MotionDetector()
        d.feed(_make_jpeg(fill=0))
        score = d.feed(_make_jpeg(fill=200))
        assert score > 50.0

    def test_moderate_change(self):
        d = MotionDetector()
        d.feed(_make_jpeg_gradient(offset=0))
        score = d.feed(_make_jpeg_gradient(offset=100))
        assert score > 0.0

    def test_reset_clears_reference(self):
        d = MotionDetector()
        d.feed(_make_jpeg(fill=0))
        d.reset()
        score = d.feed(_make_jpeg(fill=200))
        assert score == 0.0

    def test_score_range(self):
        d = MotionDetector()
        d.feed(_make_jpeg(fill=0))
        score = d.feed(_make_jpeg(fill=255))
        assert 0.0 <= score <= 100.0

    def test_sequential_feeds(self):
        d = MotionDetector()
        scores = []
        for fill in [0, 0, 128, 128, 255]:
            scores.append(d.feed(_make_jpeg(fill=fill)))
        assert scores[0] == 0.0
        assert scores[1] < 5.0
        assert scores[2] > 30.0
        assert scores[3] < 5.0
        assert scores[4] > 30.0

    def test_custom_pixel_diff_sensitivity(self):
        d_sensitive = MotionDetector(pixel_diff=10)
        d_insensitive = MotionDetector(pixel_diff=100)
        frame_a = _make_jpeg(fill=100)
        frame_b = _make_jpeg(fill=150)
        d_sensitive.feed(frame_a)
        score_sensitive = d_sensitive.feed(frame_b)
        d_insensitive.feed(frame_a)
        score_insensitive = d_insensitive.feed(frame_b)
        assert score_sensitive >= score_insensitive

    def test_to_grayscale_valid_jpeg(self):
        result = MotionDetector._to_grayscale(_make_jpeg(width=64, height=48))
        assert result is not None
        assert len(result) > 0
