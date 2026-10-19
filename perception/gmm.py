# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""GMM background subtraction (B04.1 + B04.2 morphology/blob detection).

Per-pixel Gaussian Mixture Model learns the background distribution over
time. A pixel that cannot be explained by any background gaussian is
classified as foreground.

Two implementations are provided with identical semantics:
- `GMMBackgroundModel` (pure Python) — no external deps beyond PIL. Slow
  but always available; used as fallback when numpy is absent.
- `GMMBackgroundModelNumpy` (vectorized) — uses numpy for 50-100x speedup.
  Auto-selected when numpy is importable.

Public factory `build_gmm(width, height, profile)` returns whichever
implementation is available.

Also exports:
- `morph_open(mask, w, h, erode_ksize, dilate_ksize)` — noise cleanup.
- `detect_blobs(mask, w, h, min_area, edge_margin_px)` — connected-component
  labeling with area + edge filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------------------
# Algorithm constants — every value named, zero magic numbers.
# ---------------------------------------------------------------------------

## Number of gaussian components per pixel (from Tapo ams.config K=3).
GMM_NUM_GAUSSIANS: int = 3

## Learning rate alpha: how fast background adapts (2% per frame).
GMM_LEARNING_RATE: float = 0.02

## Proportion of gaussians (cumulative weight) that form the background.
GMM_BACKGROUND_RATIO: float = 0.7

## Initial mixing weight assigned to a freshly-inserted gaussian.
GMM_WEIGHT_INIT: float = 0.05

## Default noise standard deviation (day profile).
GMM_NOISE_SIGMA_DEFAULT: float = 5.0

## Foreground threshold in units of sigma (|x - mean| > k*sigma => fg).
GMM_VAR_THRESH_DEFAULT: float = 4.0

## Floor on variance to avoid division-by-zero and runaway convergence.
GMM_VARIANCE_FLOOR: float = 1.0

## Number of frames required for the GMM to converge after start-up.
GMM_WARMUP_FRAMES: int = 200

## Numerical epsilon guarding division by tiny weights.
GMM_WEIGHT_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Blob detection / morphology constants
# ---------------------------------------------------------------------------

## Mask value representing "background".
MASK_BG: int = 0
## Mask value representing "foreground".
MASK_FG: int = 1

## Default morphological erode kernel size (odd).
MORPH_ERODE_KSIZE_DEFAULT: int = 3
## Default morphological dilate kernel size (odd).
MORPH_DILATE_KSIZE_DEFAULT: int = 5

## Default minimum blob area in pixels at work resolution.
BLOB_MIN_AREA_PX_DEFAULT: int = 20

## Number of 4-connected neighbour offsets used during flood-fill.
BLOB_NEIGHBOUR_COUNT: int = 4


# ---------------------------------------------------------------------------
# Profile shape (subset of perception.motion_detect.PROFILES)
# ---------------------------------------------------------------------------


@dataclass
class GMMProfile:
    """Runtime profile for GMM: noise level, learning rate, blob filter."""

    noise_sigma: float = GMM_NOISE_SIGMA_DEFAULT
    var_thresh: float = GMM_VAR_THRESH_DEFAULT
    learning_rate: float = GMM_LEARNING_RATE
    min_blob_area: int = BLOB_MIN_AREA_PX_DEFAULT
    dust_filter: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GMMProfile":
        return cls(
            noise_sigma=float(d.get("noise_sigma", GMM_NOISE_SIGMA_DEFAULT)),
            var_thresh=float(d.get("var_thresh", GMM_VAR_THRESH_DEFAULT)),
            learning_rate=float(d.get("learning_rate", GMM_LEARNING_RATE)),
            min_blob_area=int(d.get("min_blob_area", BLOB_MIN_AREA_PX_DEFAULT)),
            dust_filter=bool(d.get("dust_filter", False)),
        )


# ---------------------------------------------------------------------------
# Pure-Python gaussian component (fallback implementation)
# ---------------------------------------------------------------------------


@dataclass
class GaussianComponent:
    """Single gaussian in a per-pixel mixture (fallback path)."""

    mean: float = 0.0
    variance: float = GMM_NOISE_SIGMA_DEFAULT * GMM_NOISE_SIGMA_DEFAULT
    weight: float = 0.0


# ---------------------------------------------------------------------------
# Protocol so motion_detect.py doesn't need to know which impl is active
# ---------------------------------------------------------------------------


class GMMBackend(Protocol):
    width: int
    height: int

    def update(self, gray) -> list[int]:
        """Update model with new frame. Returns flat foreground mask."""
        ...

    def set_profile(self, profile: GMMProfile) -> None: ...


# ---------------------------------------------------------------------------
# GMMBackgroundModel — pure Python fallback
# ---------------------------------------------------------------------------


class GMMBackgroundModel:
    """Pure-Python GMM background model.

    One `GaussianComponent` list per pixel (K gaussians each). Updated
    in-place every frame. Correct but slow; used when numpy is absent.
    """

    def __init__(self, width: int, height: int, profile: GMMProfile):
        self.width = width
        self.height = height
        self._profile = profile
        self._k = GMM_NUM_GAUSSIANS
        self._components: list[list[GaussianComponent]] = []

    def set_profile(self, profile: GMMProfile) -> None:
        self._profile = profile

    def _initialize(self, gray: list[int]) -> None:
        var_init = self._profile.noise_sigma * self._profile.noise_sigma
        self._components = []
        for px_val in gray:
            row: list[GaussianComponent] = []
            for k in range(self._k):
                if k == 0:
                    row.append(
                        GaussianComponent(
                            mean=float(px_val),
                            variance=var_init,
                            weight=1.0,
                        )
                    )
                else:
                    row.append(
                        GaussianComponent(
                            mean=0.0,
                            variance=var_init,
                            weight=0.0,
                        )
                    )
            self._components.append(row)

    def update(self, gray: list[int]) -> list[int]:
        n = len(gray)
        if not self._components:
            self._initialize(gray)
            return [MASK_BG] * n

        lr = self._profile.learning_rate
        var_thresh = self._profile.var_thresh
        bg_ratio = GMM_BACKGROUND_RATIO
        sigma_init = self._profile.noise_sigma
        var_init = sigma_init * sigma_init
        weight_init = GMM_WEIGHT_INIT
        k_count = self._k

        fg_mask = [MASK_BG] * n

        for i in range(n):
            x = float(gray[i])
            comps = self._components[i]
            matched = False

            for g in comps:
                if g.weight < GMM_WEIGHT_EPSILON:
                    continue
                std = math.sqrt(max(g.variance, GMM_VARIANCE_FLOOR))
                if abs(x - g.mean) < var_thresh * std:
                    g.weight = (1.0 - lr) * g.weight + lr
                    rho = lr / max(g.weight, GMM_WEIGHT_EPSILON)
                    g.mean = (1.0 - rho) * g.mean + rho * x
                    diff = x - g.mean
                    g.variance = (1.0 - rho) * g.variance + rho * diff * diff
                    g.variance = max(g.variance, GMM_VARIANCE_FLOOR)
                    matched = True
                    break
                else:
                    g.weight = (1.0 - lr) * g.weight

            if not matched:
                weakest_idx = min(
                    range(k_count),
                    key=lambda k: comps[k].weight,
                )
                comps[weakest_idx] = GaussianComponent(
                    mean=x,
                    variance=var_init,
                    weight=weight_init,
                )

            total_w = sum(g.weight for g in comps)
            if total_w > 0:
                for g in comps:
                    g.weight /= total_w

            # Classify: sort by weight/sigma, walk down cumulative weight.
            sorted_comps = sorted(
                comps,
                key=lambda g: g.weight / math.sqrt(max(g.variance, GMM_VARIANCE_FLOOR)),
                reverse=True,
            )
            cumulative = 0.0
            is_bg = False
            for g in sorted_comps:
                cumulative += g.weight
                std = math.sqrt(max(g.variance, GMM_VARIANCE_FLOOR))
                if abs(x - g.mean) < var_thresh * std:
                    is_bg = True
                    break
                if cumulative > bg_ratio:
                    break

            fg_mask[i] = MASK_BG if is_bg else MASK_FG

        return fg_mask


# ---------------------------------------------------------------------------
# GMMBackgroundModelNumpy — vectorized fast path
# ---------------------------------------------------------------------------

try:
    import numpy as _np  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class GMMBackgroundModelNumpy:
    """Vectorized GMM using numpy arrays.

    State is stored as three (H*W, K) float arrays: means, variances,
    weights. One step of the update is expressed with numpy broadcasting;
    in practice ~50-100x faster than the pure-Python loop at 160x90 work
    resolution.
    """

    def __init__(self, width: int, height: int, profile: GMMProfile):
        import numpy as np

        self._np = np
        self.width = width
        self.height = height
        self._profile = profile
        self._k = GMM_NUM_GAUSSIANS
        self._n = width * height

        self.means = np.zeros((self._n, self._k), dtype=np.float32)
        self.variances = np.full(
            (self._n, self._k),
            profile.noise_sigma * profile.noise_sigma,
            dtype=np.float32,
        )
        self.weights = np.zeros((self._n, self._k), dtype=np.float32)
        self._initialized = False

    def set_profile(self, profile: GMMProfile) -> None:
        self._profile = profile

    def _initialize(self, gray) -> None:
        np = self._np
        arr = np.asarray(gray, dtype=np.float32).reshape(-1)
        self.means[:, 0] = arr
        self.weights[:, 0] = 1.0
        # Other components already initialized to zero weight / default variance.
        self._initialized = True

    def update(self, gray) -> list[int]:
        np = self._np
        if not self._initialized:
            self._initialize(gray)
            return [MASK_BG] * self._n

        arr = np.asarray(gray, dtype=np.float32).reshape(-1)  # (N,)
        x = arr[:, None]  # (N,1)

        lr = self._profile.learning_rate
        var_thresh = self._profile.var_thresh
        bg_ratio = GMM_BACKGROUND_RATIO
        var_init = self._profile.noise_sigma * self._profile.noise_sigma

        std = np.sqrt(np.maximum(self.variances, GMM_VARIANCE_FLOOR))  # (N,K)
        diff = np.abs(x - self.means)  # (N,K)
        within = diff < (var_thresh * std)  # (N,K)
        alive = self.weights > GMM_WEIGHT_EPSILON  # (N,K)
        candidate = within & alive  # (N,K)

        # Pick first matching component per pixel (if any).
        has_any = candidate.any(axis=1)  # (N,)
        first_match = np.argmax(candidate, axis=1)  # (N,)
        rows = np.arange(self._n)

        # Decay all gaussian weights uniformly, then bump the matched one.
        self.weights *= 1.0 - lr
        self.weights[rows[has_any], first_match[has_any]] += lr

        # Update mean + variance of the matched gaussian only.
        matched_rows = rows[has_any]
        matched_cols = first_match[has_any]
        if matched_rows.size > 0:
            w_m = self.weights[matched_rows, matched_cols]
            rho = lr / np.maximum(w_m, GMM_WEIGHT_EPSILON)
            xm = arr[matched_rows]
            mu = self.means[matched_rows, matched_cols]
            new_mu = (1.0 - rho) * mu + rho * xm
            self.means[matched_rows, matched_cols] = new_mu
            d = xm - new_mu
            var_old = self.variances[matched_rows, matched_cols]
            new_var = (1.0 - rho) * var_old + rho * d * d
            self.variances[matched_rows, matched_cols] = np.maximum(
                new_var,
                GMM_VARIANCE_FLOOR,
            )

        # Replace the weakest component for pixels with no match.
        no_match_rows = rows[~has_any]
        if no_match_rows.size > 0:
            weakest = np.argmin(self.weights[no_match_rows], axis=1)
            self.means[no_match_rows, weakest] = arr[no_match_rows]
            self.variances[no_match_rows, weakest] = var_init
            self.weights[no_match_rows, weakest] = GMM_WEIGHT_INIT

        # Normalize weights per row.
        total = self.weights.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        self.weights /= total

        # Classify foreground: walk gaussians sorted by weight/sigma.
        std2 = np.sqrt(np.maximum(self.variances, GMM_VARIANCE_FLOOR))
        rank_key = self.weights / std2
        order = np.argsort(-rank_key, axis=1)  # (N,K)
        sorted_weights = np.take_along_axis(self.weights, order, axis=1)
        sorted_means = np.take_along_axis(self.means, order, axis=1)
        sorted_vars = np.take_along_axis(self.variances, order, axis=1)
        sorted_std = np.sqrt(np.maximum(sorted_vars, GMM_VARIANCE_FLOOR))
        cumw = np.cumsum(sorted_weights, axis=1)

        # Background candidates: gaussian k is still "in the race" if the
        # cumulative weight BEFORE including k is <= bg_ratio.  Equivalently,
        # shift cumw by one (first column = 0) and test <= bg_ratio.
        prev_cumw = np.zeros_like(cumw)
        prev_cumw[:, 1:] = cumw[:, :-1]
        bg_valid = prev_cumw <= bg_ratio
        diff_sorted = np.abs(arr[:, None] - sorted_means)
        within_sorted = diff_sorted < (var_thresh * sorted_std)
        is_bg = (within_sorted & bg_valid).any(axis=1)

        mask = np.where(is_bg, MASK_BG, MASK_FG).astype(np.uint8)
        return mask.tolist()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_gmm(width: int, height: int, profile: GMMProfile) -> GMMBackend:
    """Return numpy impl if available, else pure-Python fallback."""
    if _HAS_NUMPY:
        return GMMBackgroundModelNumpy(width, height, profile)
    return GMMBackgroundModel(width, height, profile)


# ---------------------------------------------------------------------------
# Morphological open (erode followed by dilate)
# ---------------------------------------------------------------------------


def _morph_pass(
    mask: list[int],
    w: int,
    h: int,
    ksize: int,
    *,
    erode: bool,
) -> list[int]:
    """One morphological pass (erode or dilate) on a flat binary mask."""
    out = [MASK_BG] * (w * h)
    radius = ksize // 2
    if erode:
        # Erode: output FG iff the entire kernel lies inside the image
        # AND all neighbours inside it are FG (zero-padding semantics).
        for y in range(h):
            kernel_clipped_y = (y - radius < 0) or (y + radius >= h)
            for x in range(w):
                kernel_clipped_x = (x - radius < 0) or (x + radius >= w)
                if kernel_clipped_x or kernel_clipped_y:
                    out[y * w + x] = MASK_BG
                    continue
                val = MASK_FG
                for ny in range(y - radius, y + radius + 1):
                    row = ny * w
                    for nx in range(x - radius, x + radius + 1):
                        if mask[row + nx] == MASK_BG:
                            val = MASK_BG
                            break
                    if val == MASK_BG:
                        break
                out[y * w + x] = val
    else:
        # Dilate: output FG if any neighbour within kernel bounds is FG.
        for y in range(h):
            for x in range(w):
                val = MASK_BG
                y0 = max(0, y - radius)
                y1 = min(h - 1, y + radius)
                x0 = max(0, x - radius)
                x1 = min(w - 1, x + radius)
                for ny in range(y0, y1 + 1):
                    row = ny * w
                    for nx in range(x0, x1 + 1):
                        if mask[row + nx] == MASK_FG:
                            val = MASK_FG
                            break
                    if val == MASK_FG:
                        break
                out[y * w + x] = val
    return out


def morph_open(
    mask: list[int],
    w: int,
    h: int,
    erode_ksize: int = MORPH_ERODE_KSIZE_DEFAULT,
    dilate_ksize: int = MORPH_DILATE_KSIZE_DEFAULT,
) -> list[int]:
    """Morphological open: erode then dilate.

    Uses numpy (scipy-less) vectorization if numpy is available, falls
    back to the pure-Python reference implementation otherwise.
    """
    if _HAS_NUMPY:
        import numpy as np

        arr = np.asarray(mask, dtype=np.uint8).reshape(h, w)
        eroded = _morph_vec(arr, erode_ksize, erode=True)
        dilated = _morph_vec(eroded, dilate_ksize, erode=False)
        return dilated.reshape(-1).tolist()
    eroded = _morph_pass(mask, w, h, erode_ksize, erode=True)
    dilated = _morph_pass(eroded, w, h, dilate_ksize, erode=False)
    return dilated


def _morph_vec(arr, ksize: int, *, erode: bool):
    """Numpy-based morphological op using min/max over shifted views."""
    import numpy as np

    r = ksize // 2
    padded = np.pad(arr, r, mode="constant", constant_values=0)
    h, w = arr.shape
    out = np.zeros_like(arr)
    if erode:
        stack_min = np.ones_like(arr)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted = padded[r + dy : r + dy + h, r + dx : r + dx + w]
                stack_min = np.minimum(stack_min, shifted)
        out = stack_min
    else:
        stack_max = np.zeros_like(arr)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted = padded[r + dy : r + dy + h, r + dx : r + dx + w]
                stack_max = np.maximum(stack_max, shifted)
        out = stack_max
    return out


# ---------------------------------------------------------------------------
# Connected-component blob detection with centroid + area
# ---------------------------------------------------------------------------


@dataclass
class Blob:
    """Detected foreground region."""

    cx: float = 0.0
    cy: float = 0.0
    area: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1


def detect_blobs(
    mask: list[int],
    w: int,
    h: int,
    min_area: int = BLOB_MIN_AREA_PX_DEFAULT,
    edge_margin_px: int = 0,
) -> list[Blob]:
    """4-connected flood-fill blob detection with area + edge filtering."""
    visited = [False] * (w * h)
    blobs: list[Blob] = []

    # Neighbour offsets for 4-connectivity.
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for y in range(h):
        row_offset = y * w
        for x in range(w):
            idx = row_offset + x
            if visited[idx] or mask[idx] == MASK_BG:
                continue

            stack = [(x, y)]
            pixels_x: list[int] = []
            pixels_y: list[int] = []
            x0, y0, x1, y1 = x, y, x, y

            while stack:
                px, py = stack.pop()
                pidx = py * w + px
                if visited[pidx]:
                    continue
                visited[pidx] = True
                if mask[pidx] == MASK_BG:
                    continue
                pixels_x.append(px)
                pixels_y.append(py)
                if px < x0:
                    x0 = px
                if px > x1:
                    x1 = px
                if py < y0:
                    y0 = py
                if py > y1:
                    y1 = py
                for dx, dy in nbrs:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny * w + nx]:
                        stack.append((nx, ny))

            area = len(pixels_x)
            if area < min_area:
                continue

            cx = sum(pixels_x) / area
            cy = sum(pixels_y) / area

            if edge_margin_px > 0:
                if (
                    cx < edge_margin_px
                    or cx > w - edge_margin_px
                    or cy < edge_margin_px
                    or cy > h - edge_margin_px
                ):
                    continue

            blobs.append(Blob(cx=cx, cy=cy, area=area, bbox=(x0, y0, x1, y1)))

    return blobs
