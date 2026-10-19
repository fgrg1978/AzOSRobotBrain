# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0035 confidence-aware real-time — brain-side flag test."""

from protocol import ActuatorCmd, FLAG_LOW_CONFIDENCE, FLAG_EMERGENCY


def test_low_confidence_flag_value():
    # Must be a free bit distinct from emergency/alert (0x01/0x02).
    assert FLAG_LOW_CONFIDENCE == 0x04


def test_low_confidence_flag_roundtrips():
    cmd = ActuatorCmd.wheeled(60, 60, flags=FLAG_LOW_CONFIDENCE)
    cmd2 = ActuatorCmd.from_bytes(cmd.to_bytes())
    assert cmd2.flags & FLAG_LOW_CONFIDENCE
    assert not (cmd2.flags & FLAG_EMERGENCY)
