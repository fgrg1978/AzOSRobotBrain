# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""
Surround View / Bird's Eye View (BEV) — multi-camera top-down fusion.

Generates a top-down view by combining 4 camera images (front, rear, left,
right) using homographic perspective transforms. Same technique used in
parking assist systems.

Pipeline:
    1. Undistort lens (if calibration is available)
    2. Warp each view to bird's eye perspective (homography)
    3. Blend all 4 views into a unified top-down image

Usage from brain server:
    from perception.surround_view import SurroundView
    sv = SurroundView(config)  # from config.yaml surround section
    bev = sv.generate(images)  # dict of front/rear/left/right → BGR

Adapted from skymap/surround.py for AzOSRobotBrain integration.
"""

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default input resolution expected by the homography points.
DEFAULT_INPUT_WIDTH = 640
DEFAULT_INPUT_HEIGHT = 480

# Default output BEV canvas size.
DEFAULT_OUTPUT_SIZE = (800, 800)

# Gaussian blur kernel for weight mask blending.
BLEND_KERNEL_SIZE = 51
BLEND_SIGMA = 20

# Robot icon dimensions (pixels in the BEV center).
ROBOT_ICON_WIDTH = 80
ROBOT_ICON_HEIGHT = 120


@dataclass
class CameraConfig:
    """Homography mapping for one camera position."""

    position: str  # 'front', 'rear', 'left', 'right'
    src_points: np.ndarray  # 4 corners in the original image
    dst_points: np.ndarray  # 4 corners in the BEV canvas
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


# Default trapezoid→rectangle mappings for 640×480 input → 800×800 output.
# These work for typical wide-angle cameras at ~0.5m height.
# Tune src_points per physical camera mounting.
DEFAULT_CAMERA_CONFIGS: Dict[str, CameraConfig] = {
    "front": CameraConfig(
        position="front",
        src_points=np.float32([[160, 280], [480, 280], [580, 480], [60, 480]]),
        dst_points=np.float32([[250, 0], [550, 0], [550, 300], [250, 300]]),
    ),
    "rear": CameraConfig(
        position="rear",
        src_points=np.float32([[480, 280], [160, 280], [60, 480], [580, 480]]),
        dst_points=np.float32([[250, 800], [550, 800], [550, 500], [250, 500]]),
    ),
    "left": CameraConfig(
        position="left",
        src_points=np.float32([[160, 280], [480, 280], [580, 480], [60, 480]]),
        dst_points=np.float32([[0, 250], [0, 550], [300, 550], [300, 250]]),
    ),
    "right": CameraConfig(
        position="right",
        src_points=np.float32([[480, 280], [160, 280], [60, 480], [580, 480]]),
        dst_points=np.float32([[800, 250], [800, 550], [500, 550], [500, 250]]),
    ),
}


class SurroundView:
    """Multi-camera bird's eye view generator.

    Integrates with the brain server's camera pipeline. Accepts 1-4 camera
    frames and produces a unified top-down view for SLAM, navigation, or
    operator display.
    """

    def __init__(
        self,
        configs: Optional[Dict[str, CameraConfig]] = None,
        output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    ):
        if not _HAS_CV2:
            raise RuntimeError("surround_view requires opencv-python (pip install opencv-python)")
        self.configs = configs or DEFAULT_CAMERA_CONFIGS
        self.output_size = output_size
        # Pre-compute homography matrices (they don't change per frame).
        self._homographies: Dict[str, np.ndarray] = {}
        self._masks: Dict[str, np.ndarray] = {}
        for pos, cfg in self.configs.items():
            self._homographies[pos] = cv2.getPerspectiveTransform(cfg.src_points, cfg.dst_points)
            self._masks[pos] = self._build_weight_mask(cfg)

    def generate(
        self, images: Dict[str, np.ndarray], draw_robot: bool = True
    ) -> Optional[np.ndarray]:
        """Generate bird's eye view from camera images.

        Args:
            images: dict mapping position ('front','rear','left','right')
                    to BGR numpy arrays. Not all 4 are required.
            draw_robot: if True, draw a robot icon in the center.

        Returns:
            BGR image of shape (output_h, output_w, 3), or None if no images.
        """
        warped_list = []
        mask_list = []

        for pos in ("front", "rear", "left", "right"):
            if pos not in images or pos not in self._homographies:
                continue

            img = images[pos]
            cfg = self.configs[pos]

            # Undistort if calibration is available
            if cfg.camera_matrix is not None and cfg.dist_coeffs is not None:
                img = self._undistort(img, cfg.camera_matrix, cfg.dist_coeffs)

            # Resize to expected input dimensions
            img = cv2.resize(img, (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT))

            # Warp to BEV
            warped = cv2.warpPerspective(
                img,
                self._homographies[pos],
                self.output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            warped_list.append(warped)
            mask_list.append(self._masks[pos])

        if not warped_list:
            return None

        result = self._blend(warped_list, mask_list)

        if draw_robot:
            self._draw_robot_icon(result)

        return result

    def _build_weight_mask(self, config: CameraConfig) -> np.ndarray:
        """Create a soft blending mask for one camera's BEV region."""
        mask = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.float32)
        pts = config.dst_points.astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1.0)
        mask = cv2.GaussianBlur(mask, (BLEND_KERNEL_SIZE, BLEND_KERNEL_SIZE), BLEND_SIGMA)
        return mask

    @staticmethod
    def _undistort(
        image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """Remove lens distortion using calibration parameters."""
        h, w = image.shape[:2]
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        result = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
        x, y, w2, h2 = roi
        if w2 > 0 and h2 > 0:
            result = result[y : y + h2, x : x + w2]
        return result

    @staticmethod
    def _blend(warped_images, masks):
        """Weighted blending of warped views."""
        h, w = warped_images[0].shape[:2]
        result = np.zeros((h, w, 3), dtype=np.float64)
        total_weight = np.zeros((h, w), dtype=np.float64)

        for img, mask in zip(warped_images, masks):
            for c in range(3):
                result[:, :, c] += img[:, :, c].astype(np.float64) * mask
            total_weight += mask

        valid = total_weight > 0
        for c in range(3):
            result[:, :, c][valid] /= total_weight[valid]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _draw_robot_icon(self, image: np.ndarray):
        """Draw a simple robot icon in the center of the BEV."""
        cx = self.output_size[0] // 2
        cy = self.output_size[1] // 2
        hw = ROBOT_ICON_WIDTH // 2
        hh = ROBOT_ICON_HEIGHT // 2

        cv2.rectangle(image, (cx - hw, cy - hh), (cx + hw, cy + hh), (50, 50, 50), -1)
        cv2.rectangle(image, (cx - hw, cy - hh), (cx + hw, cy + hh), (100, 100, 100), 2)
        cv2.arrowedLine(image, (cx, cy), (cx, cy - hh + 10), (0, 200, 0), 2, tipLength=0.4)


# ---------------------------------------------------------------------------
# Calibration utility
# ---------------------------------------------------------------------------


def calibrate_from_checkerboard(
    image: np.ndarray, pattern_size: Tuple[int, int] = (9, 6), square_size_mm: float = 25.0
):
    """Calibrate a camera from a checkerboard image.

    Args:
        image: BGR image of the checkerboard.
        pattern_size: (columns, rows) of internal corners.
        square_size_mm: physical size of one square in mm.

    Returns:
        (camera_matrix, dist_coeffs) or (None, None) on failure.
    """
    if not _HAS_CV2:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        [objp], [corners], gray.shape[::-1], None, None
    )

    return camera_matrix, dist_coeffs
