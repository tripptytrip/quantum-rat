"""Deterministic Basal Ganglia-style action selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from brain.contracts import Action, Observation


TURN_STEP = 0.3  # radians per tick equivalent


def _channel_scores(
    observation: Observation,
    wm_novelty: float,
    trn_gain: float,
    microsleep_active: bool,
    place_id: int,
) -> Dict[str, float]:
    if microsleep_active:
        return {"REST": 1.0}

    # Simple deterministic biases
    parity_bias = 0.1 if (place_id % 2 == 0) else -0.1
    scores: Dict[str, float] = {}
    scores["FORWARD"] = 0.8 * trn_gain - 0.5 * observation.pain_signal + 0.2 * wm_novelty
    scores["TURN_LEFT"] = 0.3 * wm_novelty + max(parity_bias, 0.0)
    scores["TURN_RIGHT"] = 0.3 * wm_novelty + max(-parity_bias, 0.0)
    scores["REST"] = 0.1 * (1 - trn_gain) + 0.8 * observation.pain_signal
    return scores


def select_action(
    observation: Observation,
    wm_novelty: float,
    trn_gain: float,
    microsleep_active: bool,
    place_id: int,
) -> Action:
    scores = _channel_scores(observation, wm_novelty, trn_gain, microsleep_active, place_id)
    # Deterministic tie-break order
    order = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "REST"]
    best = max(order, key=lambda name: (scores.get(name, float("-inf")), -order.index(name)))
    if best == "FORWARD":
        return Action(name="FORWARD", thrust=1.0, turn=0.0)
    if best == "TURN_LEFT":
        return Action(name="TURN_LEFT", thrust=0.3, turn=TURN_STEP)
    if best == "TURN_RIGHT":
        return Action(name="TURN_RIGHT", thrust=0.3, turn=-TURN_STEP)
    return Action(name="REST", thrust=0.0, turn=0.0)


__all__ = ["select_action", "TURN_STEP"]
