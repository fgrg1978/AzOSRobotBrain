# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Stream-ID-prefixed multiplexed wire format (RFC-0021) — brain side.

Byte-for-byte mirror of `crates/multi-stream/src/lib.rs` in AzOS. The
multi-stream layer is the OUTERMOST framing on the brain↔kernel TCP byte
stream: it wraps whatever the lower layers (plaintext / RFC-0019 Noise
transport) produced, so the receiver can demultiplex by stream-id BEFORE
deciding how to decode each stream. Per-stream encryption is future work
(RFC-0021 §non-goals); today the control stream carries the encrypted/auth'd
brain protocol and camera streams (Phase D) carry raw encoded video.

Wire format:

    +-----------+--------------+----------------------+
    | STREAM_ID | LEN (u16 LE) | PAYLOAD (LEN bytes)  |
    |  1 byte   |   2 bytes    |                      |
    +-----------+--------------+----------------------+
"""

from __future__ import annotations

from typing import Optional

# ── Frame header layout (mirror crate constants) ─────────────────────────────
STREAM_ID_BYTES = 1
LEN_FIELD_BYTES = 2
HEADER_LEN = STREAM_ID_BYTES + LEN_FIELD_BYTES  # 3
MAX_PAYLOAD_LEN = 65535  # u16::MAX
MIN_FRAME_LEN = HEADER_LEN

# ── Stream ID allocations ────────────────────────────────────────────────────
STREAM_CONTROL = 0x00  # sensors, status, actuator cmds (brain protocol)
STREAM_CAMERA_BASE = 0x10
STREAM_CAMERA_LAST = 0x1F
STREAM_CAMERA_COUNT = STREAM_CAMERA_LAST - STREAM_CAMERA_BASE + 1  # 16
STREAM_LIDAR = 0x20
STREAM_AUDIO = 0x21


def wrap(stream_id: int, inner_bytes: bytes) -> bytes:
    """Encode `inner_bytes` as a multiplexed frame:
    `[stream_id][len_lo][len_hi][payload...]`.

    Raises ValueError if the payload exceeds MAX_PAYLOAD_LEN (mirrors the
    crate's WrapError::PayloadTooLarge)."""
    if len(inner_bytes) > MAX_PAYLOAD_LEN:
        raise ValueError(f"multi_stream payload {len(inner_bytes)} > {MAX_PAYLOAD_LEN}")
    return bytes([stream_id]) + len(inner_bytes).to_bytes(LEN_FIELD_BYTES, "little") + inner_bytes


def unwrap(frame: bytes) -> Optional[tuple[int, bytes]]:
    """Parse a multiplexed frame. Returns `(stream_id, payload)` or None if the
    frame is shorter than the header or the LEN field claims more bytes than
    are present (length-extension / truncated — same guard as the crate)."""
    if len(frame) < HEADER_LEN:
        return None
    stream_id = frame[0]
    payload_len = int.from_bytes(frame[STREAM_ID_BYTES:HEADER_LEN], "little")
    available = len(frame) - HEADER_LEN
    if payload_len > available:
        return None
    return (stream_id, frame[HEADER_LEN : HEADER_LEN + payload_len])


def camera_stream_id(n: int) -> Optional[int]:
    """Stream ID for camera index `n` (0-based), or None if out of range."""
    if n < 0 or n >= STREAM_CAMERA_COUNT:
        return None
    return STREAM_CAMERA_BASE + n


def is_camera_stream(stream_id: int) -> bool:
    return STREAM_CAMERA_BASE <= stream_id <= STREAM_CAMERA_LAST
