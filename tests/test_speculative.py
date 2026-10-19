# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0034 speculative actuation — brain-side tests (PredictCmd + predictor)."""

from protocol import PredictCmd, ActuatorCmd
from planner.speculative import predict_next, CONF_SCRIPTED, CONF_PLANNED


class FakePolicy:
    """Minimal policy stub: skill name → ActuatorCmd."""

    _MAP = {"FORWARD": (50, 50), "TURN_LEFT": (-30, 30), "STOP": (0, 0)}

    def translate(self, skill, args=None):
        l, r = self._MAP.get(skill, (0, 0))
        return ActuatorCmd.wheeled(l, r)


def test_predictcmd_roundtrip():
    p = PredictCmd(cmd=ActuatorCmd.wheeled(50, -50), confidence=200)
    p2 = PredictCmd.from_bytes(p.to_bytes())
    assert p2.cmd.channels == [50, -50]
    assert p2.confidence == 200


def test_predictcmd_wire_matches_kernel_layout():
    # Kernel decode_predict_cmd does split_at(len-1): ActuatorCmd bytes + 1 conf byte.
    cmd = ActuatorCmd.wheeled(10, 20)
    assert PredictCmd(cmd=cmd, confidence=255).to_bytes() == cmd.to_bytes() + bytes([255])


def test_predictcmd_confidence_clamped():
    cmd = ActuatorCmd.wheeled(0, 0)
    assert PredictCmd(cmd=cmd, confidence=999).to_bytes()[-1] == 255
    assert PredictCmd(cmd=cmd, confidence=-5).to_bytes()[-1] == 0


def test_predict_next_within_plan():
    plan = [{"skill": "FORWARD"}, {"skill": "TURN_LEFT"}, {"skill": "STOP"}]
    pred = predict_next(plan, 0, FakePolicy())  # next step = TURN_LEFT
    assert pred is not None
    assert pred.cmd.channels == [-30, 30]
    assert pred.confidence == CONF_PLANNED


def test_predict_next_scripted_confidence():
    plan = [{"skill": "FORWARD"}, {"skill": "STOP"}]
    pred = predict_next(plan, 0, FakePolicy(), is_scripted=True)
    assert pred is not None and pred.confidence == CONF_SCRIPTED


def test_predict_next_end_of_plan_is_none():
    assert predict_next([{"skill": "FORWARD"}], 0, FakePolicy()) is None
    assert predict_next([], 0, FakePolicy()) is None
