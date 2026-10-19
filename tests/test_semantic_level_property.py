# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0037 SemanticLevelCmd — Hypothesis property tests.

Requires the `hypothesis` package. Intentionally separated from
test_semantic_level.py so the plain tests run in environments that lack
hypothesis (mirrors the test_protocol_property.py split).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, strategies as st

from protocol import SemanticLevelCmd

# Number of examples matches the project-wide default used in test_protocol_property.py.
HYPOTHESIS_MAX_EXAMPLES = 200

# Unsigned 8-bit max.
U8_MAX = 255


@settings(max_examples=HYPOTHESIS_MAX_EXAMPLES, deadline=None)
@given(level=st.integers(0, U8_MAX))
def test_prop_semantic_level_cmd_roundtrip(level: int) -> None:
    """Any byte value 0–255 encodes to a 1-byte payload and decodes back (RFC-0037).

    to_bytes masks to u8 (& 0xFF); from_bytes inverts exactly. This pins the
    wire contract against accidental widening to u16 or narrowing to a smaller
    integer type on either side.
    """
    cmd = SemanticLevelCmd(level=level)
    data = cmd.to_bytes()
    assert len(data) == 1
    cmd2 = SemanticLevelCmd.from_bytes(data)
    assert cmd2.level == level & 0xFF  # to_bytes masks; from_bytes restores
