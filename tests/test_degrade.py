# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0036 brain-triggered degraded mode — brain-side tests.

Covers the wire contract (DegradeCmd / PKT_DEGRADE, byte-synced with the kernel
`brain_protocol.rs`) and the *real* trigger: `_perception_cycle` arms degraded
mode after a streak of perception failures and clears it on recovery.
"""

import asyncio
import types

import protocol
from protocol import (
    DegradeCmd,
    DEGRADE_CMD,
    DEGRADE_CLEAR,
    DEGRADE_REASON_PERCEPTION_BLIND,
    DEGRADE_REASON_SENSOR_INCOHERENT,
    DEGRADE_REASON_UNMODELLED_HAZARD,
    ActuatorCmd,
)
from server import BrainServer, DEGRADE_PERCEPTION_FAIL_THRESHOLD

# ── Wire contract ────────────────────────────────────────────────────────────


def test_pkt_degrade_value():
    # Must match brain_protocol.rs PKT_DEGRADE and not collide with 0x88/0x89.
    assert DEGRADE_CMD == 0x8A
    assert DEGRADE_CLEAR == 0


def test_degradecmd_roundtrip():
    for reason in (
        DEGRADE_CLEAR,
        DEGRADE_REASON_PERCEPTION_BLIND,
        DEGRADE_REASON_SENSOR_INCOHERENT,
        DEGRADE_REASON_UNMODELLED_HAZARD,
    ):
        d = DegradeCmd(reason=reason)
        assert d.to_bytes() == bytes([reason])
        assert DegradeCmd.from_bytes(d.to_bytes()).reason == reason


def test_degradecmd_reason_masked():
    assert DegradeCmd(reason=0x1FF).to_bytes() == bytes([0xFF])


# ── Real trigger: _perception_cycle ──────────────────────────────────────────


class _FakeWriter:
    """Captures wire frames so the test can decode what the brain sent."""

    def __init__(self):
        self.frames: list[bytes] = []

    def write(self, data):
        self.frames.append(bytes(data))

    async def drain(self):
        pass

    def degrade_packets(self):
        """All (reason,) decoded from PKT_DEGRADE frames, in order."""
        out = []
        for f in self.frames:
            parsed = protocol.parse_packet(f)
            if parsed and parsed[0] == DEGRADE_CMD:
                out.append(DegradeCmd.from_bytes(parsed[1]).reason)
        return out


def _make_server(vision):
    """Build a BrainServer skipping heavy __init__, wired with the fakes that
    `_perception_cycle` touches and the real RFC-0036 streak state."""
    srv = BrainServer.__new__(BrainServer)
    srv.power = types.SimpleNamespace(is_eco=False)
    srv.alert_pipeline = types.SimpleNamespace(active_evidence=False)
    srv.mode_manager = types.SimpleNamespace(uses_llm=lambda: True, current_name="patrol")
    srv.vision = vision
    srv.planner = types.SimpleNamespace(decide=lambda **kw: "FORWARD")
    srv.policy = types.SimpleNamespace(from_text=lambda action: ActuatorCmd.wheeled(10, 10))
    srv.state = types.SimpleNamespace(sensors={}, odom={})
    srv._perception_fail_streak = 0
    srv._degraded_sent = False
    srv._writer = _FakeWriter()
    return srv


def test_perception_blind_streak_arms_then_clears():
    blind = types.SimpleNamespace(
        describe=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VLM down"))
    )
    srv = _make_server(blind)
    w = srv._writer

    # Below threshold: no degrade armed yet.
    for _ in range(DEGRADE_PERCEPTION_FAIL_THRESHOLD - 1):
        asyncio.run(srv._perception_cycle(b"img", w))
    assert w.degrade_packets() == []
    assert not srv._degraded_sent

    # Reaching the threshold arms degraded mode exactly once.
    asyncio.run(srv._perception_cycle(b"img", w))
    assert srv._degraded_sent
    assert w.degrade_packets() == [DEGRADE_REASON_PERCEPTION_BLIND]

    # Still blind: armed, but NOT re-sent (idempotent).
    asyncio.run(srv._perception_cycle(b"img", w))
    assert w.degrade_packets() == [DEGRADE_REASON_PERCEPTION_BLIND]

    # Recovery: vision works again → degraded mode cleared.
    srv.vision = types.SimpleNamespace(describe=lambda *a, **k: "a clear scene")
    asyncio.run(srv._perception_cycle(b"img", w))
    assert not srv._degraded_sent
    assert srv._perception_fail_streak == 0
    assert w.degrade_packets() == [DEGRADE_REASON_PERCEPTION_BLIND, DEGRADE_CLEAR]


def test_single_hiccup_does_not_arm():
    # One failure then recovery must never arm degraded mode (threshold > 1).
    state = {"fail": True}

    def describe(*a, **k):
        if state["fail"]:
            raise RuntimeError("transient")
        return "ok"

    srv = _make_server(types.SimpleNamespace(describe=describe))
    asyncio.run(srv._perception_cycle(b"img", srv._writer))
    state["fail"] = False
    asyncio.run(srv._perception_cycle(b"img", srv._writer))
    assert srv._writer.degrade_packets() == []
    assert not srv._degraded_sent
