# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for perception.surround_view — Bird's Eye View generation."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from perception.surround_view import (
    SurroundView,
    CameraConfig,
    DEFAULT_CAMERA_CONFIGS,
    DEFAULT_OUTPUT_SIZE,
    calibrate_from_checkerboard,
)


@pytest.fixture
def synthetic_images():
    """Generate 4 synthetic camera images (solid colors for position ID)."""
    return {
        "front": np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8),  # blue
        "rear": np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8),  # green
        "left": np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8),  # red
        "right": np.full((480, 640, 3), (255, 255, 0), dtype=np.uint8),  # cyan
    }


@pytest.fixture
def surround():
    """Default SurroundView instance."""
    return SurroundView()


class TestSurroundViewInit:
    def test_default_configs_loaded(self, surround):
        assert len(surround.configs) == 4
        assert set(surround.configs.keys()) == {"front", "rear", "left", "right"}

    def test_homographies_precomputed(self, surround):
        assert len(surround._homographies) == 4
        for H in surround._homographies.values():
            assert H.shape == (3, 3)

    def test_masks_precomputed(self, surround):
        assert len(surround._masks) == 4
        for mask in surround._masks.values():
            assert mask.shape == (DEFAULT_OUTPUT_SIZE[1], DEFAULT_OUTPUT_SIZE[0])

    def test_custom_output_size(self):
        sv = SurroundView(output_size=(400, 400))
        assert sv.output_size == (400, 400)


class TestSurroundViewGenerate:
    def test_all_four_cameras(self, surround, synthetic_images):
        result = surround.generate(synthetic_images)
        assert result is not None
        assert result.shape == (800, 800, 3)
        assert result.dtype == np.uint8

    def test_single_camera(self, surround):
        images = {"front": np.zeros((480, 640, 3), dtype=np.uint8)}
        result = surround.generate(images)
        assert result is not None
        assert result.shape == (800, 800, 3)

    def test_two_cameras(self, surround):
        images = {
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "rear": np.full((480, 640, 3), 128, dtype=np.uint8),
        }
        result = surround.generate(images)
        assert result is not None

    def test_empty_dict_returns_none(self, surround):
        assert surround.generate({}) is None

    def test_unknown_position_ignored(self, surround):
        images = {"top": np.zeros((480, 640, 3), dtype=np.uint8)}
        assert surround.generate(images) is None

    def test_no_robot_icon(self, surround, synthetic_images):
        result = surround.generate(synthetic_images, draw_robot=False)
        assert result is not None

    def test_different_input_sizes(self, surround):
        """Images of varying sizes should be resized internally."""
        images = {
            "front": np.zeros((240, 320, 3), dtype=np.uint8),
            "rear": np.zeros((1080, 1920, 3), dtype=np.uint8),
        }
        result = surround.generate(images)
        assert result is not None
        assert result.shape == (800, 800, 3)


class TestCameraConfig:
    def test_default_configs_have_4_points(self):
        for cfg in DEFAULT_CAMERA_CONFIGS.values():
            assert cfg.src_points.shape == (4, 2)
            assert cfg.dst_points.shape == (4, 2)

    def test_configs_are_float32(self):
        for cfg in DEFAULT_CAMERA_CONFIGS.values():
            assert cfg.src_points.dtype == np.float32
            assert cfg.dst_points.dtype == np.float32

    def test_no_calibration_by_default(self):
        for cfg in DEFAULT_CAMERA_CONFIGS.values():
            assert cfg.camera_matrix is None
            assert cfg.dist_coeffs is None


class TestCalibration:
    def test_random_image_fails(self):
        """Checkerboard detection should fail on random noise."""
        random_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cam, dist = calibrate_from_checkerboard(random_img)
        assert cam is None
        assert dist is None


class TestBlending:
    def test_weight_masks_sum_coverage(self, surround):
        """Masks should cover each camera quadrant of the BEV canvas.

        The center of the canvas is deliberately left uncovered — that is
        where the robot icon is drawn (ROBOT_ICON_WIDTH × ROBOT_ICON_HEIGHT
        in surround_view.py). The 4 cameras cover a "ring" around the
        robot: front (top), rear (bottom), left, right.
        """
        total = np.zeros((800, 800), dtype=np.float32)
        for mask in surround._masks.values():
            total += mask
        # Each camera's region (outside the central robot-icon zone) must
        # have coverage from at least one mask.
        samples = {
            "front": total[150, 400],  # top-middle → front camera
            "rear": total[650, 400],  # bottom-middle → rear camera
            "left": total[400, 150],  # left-middle → left camera
            "right": total[400, 650],  # right-middle → right camera
        }
        for name, val in samples.items():
            assert val > 0, f"{name} quadrant has no camera coverage (val={val})"
