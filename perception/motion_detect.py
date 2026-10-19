# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Motion detection — GMM background subtraction pipeline (B04).

Replaces the legacy two-frame differencing with a full GMM pipeline
reverse-engineered from the Tapo C510W AMS-VDR system:

    JPEG -> downsample -> grayscale -> GMM -> foreground mask ->
    morphological open -> connected components -> dust filter ->
    trajectory confirmation -> motion score 0..100

The core GMM and blob logic live in `perception.gmm`. The trajectory
tracker lives in `perception.tracker`. This module glues them together
and preserves the public `MotionDetector` API:

    detector = MotionDetector(sensitivity=7, lighting_mode="auto")
    score = detector.feed(jpeg_bytes)   # float in [0.0, 100.0]
    if score > 0: ...

During the GMM warm-up window (~200 frames) the detector falls back to
the legacy frame-diff scoring so the caller still gets useful output
while the background model converges.
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.gmm import (
    Blob,
    GMMProfile,
    GMM_LEARNING_RATE,
    GMM_WARMUP_FRAMES,
    MORPH_DILATE_KSIZE_DEFAULT,
    MORPH_ERODE_KSIZE_DEFAULT,
    build_gmm,
    detect_blobs,
    morph_open,
)
from perception.tracker import (
    BlobTracker,
    TRAJECTORY_CONFIRM_FRAMES_DEFAULT,
    TRAJECTORY_MAX_GAP_FRAMES_DEFAULT,
    TRAJECTORY_SEARCH_RADIUS_PX_DEFAULT,
    Track,
)

# ---------------------------------------------------------------------------
# Work resolution (downsample target)
# ---------------------------------------------------------------------------

## Width (in pixels) used for GMM inference. Matches Tapo sub_vframe scale.
WORK_WIDTH: int = 160

## Fallback height used when the first frame has unknown aspect ratio.
WORK_HEIGHT_DEFAULT: int = 90

# Re-exports so callers can `from perception.motion_detect import GMM_*`.
__all__ = [
    "MotionDetector",
    "MotionScore",
    "PROFILES",
    "WORK_WIDTH",
    "WORK_HEIGHT_DEFAULT",
    "SENSITIVITY_AREA_MAP",
    "SENSITIVITY_MIN",
    "SENSITIVITY_MAX",
    "SENSITIVITY_DEFAULT",
    "DUST_MAX_AREA_PX",
    "DUST_IR_ZONE_TOP_PCT",
    "DUST_MAX_LIFETIME_FRAMES",
    "EDGE_MARGIN_PCT",
    "MOTION_THRESHOLD_PCT",
    "MOTION_PIXEL_DIFF_THRESHOLD",
    "MOTION_DOWNSAMPLE_WIDTH",
    "MOTION_BLUR_RADIUS",
]

# ---------------------------------------------------------------------------
# Sensitivity 1..10 -> area threshold in percent (Tapo-derived mapping).
# Index 0 = lowest sensitivity, index 9 = highest.
# ---------------------------------------------------------------------------

SENSITIVITY_MIN: int = 1
SENSITIVITY_MAX: int = 10
SENSITIVITY_DEFAULT: int = 7

## Percentage of work-resolution frame that must be foreground at each level.
SENSITIVITY_AREA_MAP: tuple[float, ...] = (
    13.5,
    8.5,
    5.0,
    3.3,
    2.3,
    1.55,
    1.2,
    0.73,
    0.63,
    0.50,
)

# ---------------------------------------------------------------------------
# Day/night profiles (B04.4). Dust filter enabled only for night_ir.
# ---------------------------------------------------------------------------

## Default day profile.
PROFILE_DAY: dict = {
    "noise_sigma": 5.0,
    "var_thresh": 4.0,
    "min_blob_area": 45,
    "learning_rate": GMM_LEARNING_RATE,
    "dust_filter": False,
}

## Night w/ IR illumination — more noise, insects, IR hotspots.
PROFILE_NIGHT_IR: dict = {
    "noise_sigma": 7.0,
    "var_thresh": 5.0,
    "min_blob_area": 60,
    "learning_rate": GMM_LEARNING_RATE,
    "dust_filter": True,
}

## Night w/ white/visible lighting — highest noise, no IR hotspots.
PROFILE_NIGHT_COLOR: dict = {
    "noise_sigma": 10.0,
    "var_thresh": 5.0,
    "min_blob_area": 60,
    "learning_rate": GMM_LEARNING_RATE,
    "dust_filter": False,
}

PROFILES: dict[str, dict] = {
    "day": PROFILE_DAY,
    "night_ir": PROFILE_NIGHT_IR,
    "night_color": PROFILE_NIGHT_COLOR,
}

## Mean-brightness threshold used by `auto` lighting mode to swap profiles.
AUTO_NIGHT_BRIGHTNESS_THRESHOLD: int = 50

# ---------------------------------------------------------------------------
# Blob/edge/dust constants (B04.2 + B04.4)
# ---------------------------------------------------------------------------

## Reject blobs whose centroid is within this % of the frame border.
EDGE_MARGIN_PCT: int = 10

## Max area (pixels) for a blob to still be considered "dust".
DUST_MAX_AREA_PX: int = 15

## Dust must disappear within this many frames to qualify.
DUST_MAX_LIFETIME_FRAMES: int = 2

## IR hotspot zone = top X% of the frame.
DUST_IR_ZONE_TOP_PCT: int = 30

## Scale factor applied to raw fg_pct/thresh ratio before capping at 100.
SCORE_MULTIPLIER: float = 50.0

## Hard cap on motion score.
SCORE_MAX: float = 100.0

## Lower bound on area-threshold denominator (guard div-by-zero).
SCORE_AREA_THRESH_FLOOR: float = 0.1

# ---------------------------------------------------------------------------
# Legacy frame-diff constants (kept for API compat + used during warm-up).
# ---------------------------------------------------------------------------

## Default "motion occurred" threshold (percent of pixels changed).
MOTION_THRESHOLD_PCT: int = 15

## Default |prev - curr| threshold, in 8-bit grayscale units.
MOTION_PIXEL_DIFF_THRESHOLD: int = 30

## Default work width for the legacy path (aliases WORK_WIDTH).
MOTION_DOWNSAMPLE_WIDTH: int = WORK_WIDTH

## Default blur radius (unused by GMM path but preserved for compat).
MOTION_BLUR_RADIUS: int = 2


# ---------------------------------------------------------------------------
# MotionScore — richer return payload if the caller wants it.
# ---------------------------------------------------------------------------


@dataclass
class MotionScore:
    """Structured output of one `feed()` call.

    `feed()` continues to return a plain float for API compat; this
    struct is populated on each call and accessible via
    `detector.last_score` for callers that want blob positions, bboxes,
    or trajectory ids.
    """

    score: float = 0.0
    blobs: list[Blob] = None  # type: ignore[assignment]
    confirmed_tracks: list[Track] = None  # type: ignore[assignment]
    warmup: bool = True

    def __post_init__(self) -> None:
        if self.blobs is None:
            self.blobs = []
        if self.confirmed_tracks is None:
            self.confirmed_tracks = []


# ---------------------------------------------------------------------------
# MotionDetector — top-level orchestrator
# ---------------------------------------------------------------------------


class MotionDetector:
    """GMM-based motion detector with trajectory confirmation.

    Drop-in replacement for the original frame-diff detector:
    `feed(jpeg_bytes) -> float in [0, 100]`.

    B04.5 config: accepts `sensitivity` (1-10), `lighting_mode`
    ("auto"|"day"|"night_ir"|"night_color"), `dust_filter` (bool),
    and `trajectory_confirm` (frames). Legacy `threshold_pct`/
    `pixel_diff` params are accepted (so existing callers/tests keep
    working) and reused for the warm-up frame-diff fallback.
    """

    def __init__(
        self,
        sensitivity: int = SENSITIVITY_DEFAULT,
        lighting_mode: str = "auto",
        dust_filter: bool = False,
        trajectory_confirm: int = TRAJECTORY_CONFIRM_FRAMES_DEFAULT,
        # Legacy compat parameters:
        threshold_pct: int = MOTION_THRESHOLD_PCT,
        pixel_diff: int = MOTION_PIXEL_DIFF_THRESHOLD,
    ):
        self.sensitivity = max(SENSITIVITY_MIN, min(SENSITIVITY_MAX, sensitivity))
        self.lighting_mode = lighting_mode
        self.dust_filter_enabled = dust_filter
        self.trajectory_confirm = trajectory_confirm
        self.threshold_pct = threshold_pct
        self.pixel_diff = pixel_diff

        # Active profile (may be swapped by auto-detect each frame).
        profile_name = lighting_mode if lighting_mode in PROFILES else "day"
        self._profile_dict = PROFILES[profile_name]
        self._profile = GMMProfile.from_dict(self._profile_dict)

        # Area threshold (% of frame) derived from sensitivity.
        self._area_thresh_pct = SENSITIVITY_AREA_MAP[self.sensitivity - 1]

        # Pipeline components — built lazily on first frame.
        self._gmm = None
        self._tracker = BlobTracker(
            confirm_frames=trajectory_confirm,
            search_radius_px=TRAJECTORY_SEARCH_RADIUS_PX_DEFAULT,
            max_gap_frames=TRAJECTORY_MAX_GAP_FRAMES_DEFAULT,
        )

        # Frame metadata.
        self._width = 0
        self._height = 0
        self._frame_count = 0

        # Warm-up fallback state.
        self._prev_gray: list[int] | None = None

        # Last full result (for callers that want structured data).
        self.last_score: MotionScore = MotionScore()

    # ── Public API ───────────────────────────────────────────────────────

    def feed(self, jpeg_bytes: bytes) -> float:
        """Feed one JPEG frame. Returns a 0..100 motion score.

        Returns 0 during the first frame (no prev reference) and uses
        frame-diff during the GMM warm-up window.
        """
        gray = self._to_grayscale_sized(jpeg_bytes)
        if gray is None:
            return 0.0

        self._frame_count += 1

        # First frame: initialize GMM and return 0 — we have no reference.
        if self._gmm is None:
            self._gmm = build_gmm(self._width, self._height, self._profile)
            self._gmm.update(gray)  # seeds internal state
            self._prev_gray = gray
            self.last_score = MotionScore(warmup=True)
            return 0.0

        # Auto-detect lighting mode (may swap profile this frame).
        if self.lighting_mode == "auto":
            self._auto_detect_lighting(gray)

        # Warm-up: use frame diff until GMM converges.
        if self._frame_count < GMM_WARMUP_FRAMES:
            # Still advance the GMM so it converges in the background.
            self._gmm.update(gray)
            score = self._legacy_score(gray)
            self._prev_gray = gray
            self.last_score = MotionScore(score=score, warmup=True)
            return score

        self._prev_gray = gray

        # Main GMM pipeline.
        fg_mask = self._gmm.update(gray)
        fg_mask = morph_open(
            fg_mask,
            self._width,
            self._height,
            erode_ksize=MORPH_ERODE_KSIZE_DEFAULT,
            dilate_ksize=MORPH_DILATE_KSIZE_DEFAULT,
        )

        min_area = self._profile.min_blob_area
        edge_margin = int(max(self._width, self._height) * EDGE_MARGIN_PCT / 100)
        blobs = detect_blobs(
            fg_mask,
            self._width,
            self._height,
            min_area=min_area,
            edge_margin_px=edge_margin,
        )

        # Dust / insect filter for IR cameras (B04.4).
        if self._profile.dust_filter or self.dust_filter_enabled:
            blobs = self._filter_dust(blobs)

        # Trajectory confirmation (B04.3).
        confirmed = self._tracker.update(blobs)

        score = self._score(confirmed)
        self.last_score = MotionScore(
            score=score,
            blobs=blobs,
            confirmed_tracks=confirmed,
            warmup=False,
        )
        return score

    def reset(self) -> None:
        """Clear all pipeline state (including warm-up counter)."""
        self._gmm = None
        self._prev_gray = None
        self._tracker.reset()
        self._frame_count = 0
        self.last_score = MotionScore()

    @property
    def warmup_progress(self) -> float:
        """GMM warm-up progress in [0.0, 1.0]."""
        return min(self._frame_count / GMM_WARMUP_FRAMES, 1.0)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Scoring ──────────────────────────────────────────────────────────

    def _score(self, confirmed_tracks: list[Track]) -> float:
        """Motion score in [0, 100] from confirmed track areas."""
        if not confirmed_tracks:
            return 0.0
        frame_area = self._width * self._height
        if frame_area <= 0:
            return 0.0
        total_area = sum(t.area for t in confirmed_tracks)
        fg_pct = (total_area / frame_area) * 100.0
        denom = max(self._area_thresh_pct, SCORE_AREA_THRESH_FLOOR)
        raw = (fg_pct / denom) * SCORE_MULTIPLIER
        return min(raw, SCORE_MAX)

    # ── Auto lighting mode ───────────────────────────────────────────────

    def _auto_detect_lighting(self, gray: list[int]) -> None:
        if not gray:
            return
        brightness = sum(gray) / len(gray)
        new_name = "night_ir" if brightness < AUTO_NIGHT_BRIGHTNESS_THRESHOLD else "day"
        if PROFILES.get(new_name) is not self._profile_dict:
            self._profile_dict = PROFILES[new_name]
            self._profile = GMMProfile.from_dict(self._profile_dict)
            if self._gmm is not None:
                self._gmm.set_profile(self._profile)

    # ── Dust / insect filter ─────────────────────────────────────────────

    def _filter_dust(self, blobs: list[Blob]) -> list[Blob]:
        ir_zone_y = self._height * DUST_IR_ZONE_TOP_PCT / 100.0
        kept: list[Blob] = []
        for b in blobs:
            is_dust = (b.area < DUST_MAX_AREA_PX) and (b.cy < ir_zone_y)
            if not is_dust:
                kept.append(b)
        return kept

    # ── Legacy frame-diff fallback (used during warm-up) ─────────────────

    def _legacy_score(self, gray: list[int]) -> float:
        prev = self._prev_gray
        if prev is None or len(prev) != len(gray):
            return 0.0
        return self._compute_score(prev, gray)

    def _compute_score(self, prev: list[int], curr: list[int]) -> float:
        """Percentage of pixels changed by more than `pixel_diff`."""
        total = len(prev)
        if total == 0:
            return 0.0
        thresh = self.pixel_diff
        changed = 0
        for p, c in zip(prev, curr):
            if abs(p - c) > thresh:
                changed += 1
        return (changed / total) * 100.0

    # ── Image decoding ───────────────────────────────────────────────────

    @staticmethod
    def _to_grayscale(jpeg_bytes: bytes) -> list[int] | None:
        """Decode JPEG -> downsampled grayscale list. None on failure.

        Pillow is used directly (already a project dependency per
        requirements.txt). Returns None on empty input, missing PIL, or
        any decode failure.
        """
        if not jpeg_bytes:
            return None
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(jpeg_bytes))
            img.load()
            ratio = WORK_WIDTH / max(img.width, 1)
            new_h = max(int(img.height * ratio), 1)
            img = img.resize((WORK_WIDTH, new_h), Image.NEAREST)
            img = img.convert("L")
            return list(img.getdata())
        except ImportError:
            return None
        except Exception:
            return None

    def _to_grayscale_sized(self, jpeg_bytes: bytes) -> list[int] | None:
        """Decode + update `_width`/`_height`."""
        if not jpeg_bytes:
            return None
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(jpeg_bytes))
            img.load()
            ratio = WORK_WIDTH / max(img.width, 1)
            new_h = max(int(img.height * ratio), 1)
            resized = img.resize((WORK_WIDTH, new_h), Image.NEAREST).convert("L")
            self._width = WORK_WIDTH
            self._height = new_h
            return list(resized.getdata())
        except ImportError:
            return None
        except Exception:
            return None
