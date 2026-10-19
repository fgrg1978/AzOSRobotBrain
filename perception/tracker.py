# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Blob trajectory tracker (B04.3).

Simple nearest-neighbour multi-object tracker: each `Blob` detected in
the current frame is matched against existing tracks within a search
radius. A track needs `TRAJECTORY_CONFIRM_FRAMES` consecutive hits to be
"confirmed" — this eliminates transient false positives that only pop up
for a frame or two (GMM noise, JPEG compression artifacts, etc.).

Usage:
    tracker = BlobTracker()
    confirmed = tracker.update(blobs_this_frame)

The confirmation count plays the role of the Tapo AMS-VDR
`md_min_object_on_trajectory` knob (defaults to 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.gmm import Blob

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

## Frames of continuous observation required to confirm a track.
TRAJECTORY_CONFIRM_FRAMES_DEFAULT: int = 4

## Max pixel distance (L2) a blob can move frame-to-frame and still match.
TRAJECTORY_SEARCH_RADIUS_PX_DEFAULT: int = 8

## Max frames without a detection before a track is dropped.
TRAJECTORY_MAX_GAP_FRAMES_DEFAULT: int = 3

## Initial value for the running id sequence.
TRAJECTORY_FIRST_ID: int = 1


# ---------------------------------------------------------------------------
# Tracked blob
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """State carried across frames for a single tracked blob."""

    id: int = 0
    cx: float = 0.0
    cy: float = 0.0
    area: int = 0
    frames_seen: int = 0
    frames_missing: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    def is_confirmed(self, min_frames: int) -> bool:
        return self.frames_seen >= min_frames


# ---------------------------------------------------------------------------
# BlobTracker
# ---------------------------------------------------------------------------


class BlobTracker:
    """Nearest-neighbour association + N-frame confirmation."""

    def __init__(
        self,
        confirm_frames: int = TRAJECTORY_CONFIRM_FRAMES_DEFAULT,
        search_radius_px: int = TRAJECTORY_SEARCH_RADIUS_PX_DEFAULT,
        max_gap_frames: int = TRAJECTORY_MAX_GAP_FRAMES_DEFAULT,
    ):
        self.confirm_frames = max(1, int(confirm_frames))
        self.search_radius_px = max(1, int(search_radius_px))
        self.max_gap_frames = max(0, int(max_gap_frames))
        self._tracks: list[Track] = []
        self._next_id = TRAJECTORY_FIRST_ID

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = TRAJECTORY_FIRST_ID

    def update(self, blobs: list[Blob]) -> list[Track]:
        """Associate `blobs` with existing tracks; return confirmed tracks."""
        radius_sq = self.search_radius_px * self.search_radius_px

        matched_track_indices: set[int] = set()
        matched_blob_indices: set[int] = set()

        # Greedy nearest-neighbour: each track picks the closest unmatched blob.
        for ti, track in enumerate(self._tracks):
            best_dist = radius_sq
            best_bi = -1
            for bi, blob in enumerate(blobs):
                if bi in matched_blob_indices:
                    continue
                dx = track.cx - blob.cx
                dy = track.cy - blob.cy
                d = dx * dx + dy * dy
                if d < best_dist:
                    best_dist = d
                    best_bi = bi

            if best_bi >= 0:
                b = blobs[best_bi]
                track.cx = b.cx
                track.cy = b.cy
                track.area = b.area
                track.bbox = b.bbox
                track.frames_seen += 1
                track.frames_missing = 0
                matched_track_indices.add(ti)
                matched_blob_indices.add(best_bi)

        # Age unmatched tracks.
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_indices:
                track.frames_missing += 1

        # Drop stale tracks.
        self._tracks = [t for t in self._tracks if t.frames_missing <= self.max_gap_frames]

        # Start new tracks for unmatched blobs.
        for bi, blob in enumerate(blobs):
            if bi in matched_blob_indices:
                continue
            self._tracks.append(
                Track(
                    id=self._next_id,
                    cx=blob.cx,
                    cy=blob.cy,
                    area=blob.area,
                    bbox=blob.bbox,
                    frames_seen=1,
                    frames_missing=0,
                )
            )
            self._next_id += 1

        return [t for t in self._tracks if t.is_confirmed(self.confirm_frames)]
