# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0037 graded degrade-level — brain-side tests.

Covers the wire contract (SemanticLevelCmd / PKT_SEMANTIC_LEVEL, byte-synced
with the kernel `brain_protocol.rs` / `crates/ipc/src/cap.rs`) at the parity
and golden-byte level.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import protocol
from protocol import (
    SEMANTIC_LEVEL_CMD,
    SEMANTIC_LEVEL_FULL,
    SEMANTIC_LEVEL_CAUTIOUS,
    SEMANTIC_LEVEL_SLOW,
    SEMANTIC_LEVEL_CONTAINED,
    SemanticLevelCmd,
    build_packet,
    parse_packet,
)

# ── Wire-contract parity ──────────────────────────────────────────────────────


def test_pkt_semantic_level_value():
    # Must not collide with PKT_DEGRADE (0x8A) and must precede 0x8C.
    assert SEMANTIC_LEVEL_CMD == 0x8B


def test_level_const_values():
    # Values must match DEGRADE_LEVEL_* in crates/ipc/src/cap.rs.
    assert SEMANTIC_LEVEL_FULL == 0
    assert SEMANTIC_LEVEL_CAUTIOUS == 1
    assert SEMANTIC_LEVEL_SLOW == 2
    assert SEMANTIC_LEVEL_CONTAINED == 3


# ── Encoder golden bytes ──────────────────────────────────────────────────────


def test_semantic_level_cmd_golden_bytes():
    """Each level index encodes to the expected single byte."""
    assert SemanticLevelCmd(level=SEMANTIC_LEVEL_FULL).to_bytes() == bytes([0])
    assert SemanticLevelCmd(level=SEMANTIC_LEVEL_CAUTIOUS).to_bytes() == bytes([1])
    assert SemanticLevelCmd(level=SEMANTIC_LEVEL_SLOW).to_bytes() == bytes([2])
    assert SemanticLevelCmd(level=SEMANTIC_LEVEL_CONTAINED).to_bytes() == bytes([3])


def test_semantic_level_cmd_roundtrip():
    """Encode → decode round-trip for every defined level."""
    for level in (
        SEMANTIC_LEVEL_FULL,
        SEMANTIC_LEVEL_CAUTIOUS,
        SEMANTIC_LEVEL_SLOW,
        SEMANTIC_LEVEL_CONTAINED,
    ):
        cmd = SemanticLevelCmd(level=level)
        assert SemanticLevelCmd.from_bytes(cmd.to_bytes()).level == level


def test_semantic_level_cmd_byte_masked():
    """to_bytes masks to a single byte — large values truncate, not crash."""
    # 0x1FF & 0xFF == 0xFF (a single byte)
    assert SemanticLevelCmd(level=0x1FF).to_bytes() == bytes([0xFF])


# ── Full-frame round-trip via build_packet / parse_packet ────────────────────


def test_semantic_level_full_frame_roundtrip():
    """A complete wire frame for each level parses back cleanly."""
    for level in (
        SEMANTIC_LEVEL_FULL,
        SEMANTIC_LEVEL_CAUTIOUS,
        SEMANTIC_LEVEL_SLOW,
        SEMANTIC_LEVEL_CONTAINED,
    ):
        payload = SemanticLevelCmd(level=level).to_bytes()
        frame = build_packet(SEMANTIC_LEVEL_CMD, payload)
        parsed = parse_packet(frame)
        assert parsed is not None, f"parse_packet failed for level={level}"
        pkt_type, pkt_payload = parsed
        assert pkt_type == SEMANTIC_LEVEL_CMD
        assert pkt_payload == payload
        assert SemanticLevelCmd.from_bytes(pkt_payload).level == level


# ── Note: property-based (Hypothesis) test ───────────────────────────────────
# A Hypothesis roundtrip test for SemanticLevelCmd over all byte values lives
# in tests/test_semantic_level_property.py, co-located with the other property
# tests in test_protocol_property.py. Both files require `hypothesis` installed.
