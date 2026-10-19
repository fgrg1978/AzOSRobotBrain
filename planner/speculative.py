# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""RFC-0034 speculative actuation — next-command predictor (brain side).

The kernel can act on a predicted next command ahead of the confirmed one
(gated by the kernel safety envelope + SPECULATIVE_ACTUATION, default off). The
prediction source is REAL, not invented: when executing a committed skill plan
(or a deterministic scripted mode), the next command is already known — we just
translate the next plan step through the same policy and attach a confidence.

This module is a pure function (no I/O) so it is unit-testable; the SkillRunner
wires it to the wire via `send_predict`.
"""

from __future__ import annotations

from protocol import ActuatorCmd, PredictCmd

# Confidence bytes (0..255). A deterministic scripted step is certain; an
# LLM-planned step is committed but the plan can still be interrupted/replanned.
CONF_SCRIPTED: int = 255
CONF_PLANNED: int = 230  # ~0.9


def predict_next(
    plan: list[dict],
    current_step: int,
    policy,
    is_scripted: bool = False,
) -> PredictCmd | None:
    """Predict the next actuator command from the committed plan.

    Returns a `PredictCmd` (next command + confidence) or `None` if there is no
    next step. Only predicts WITHIN the committed plan — it never guesses what a
    reactive VLM will see next (that would be inventing a prediction).
    """
    nxt = current_step + 1
    if nxt < 0 or nxt >= len(plan):
        return None
    step = plan[nxt]
    skill = step.get("skill", "STOP")
    args = step.get("args", {}) or {}
    cmd: ActuatorCmd = policy.translate(skill, args)
    confidence = CONF_SCRIPTED if is_scripted else CONF_PLANNED
    return PredictCmd(cmd=cmd, confidence=confidence)
