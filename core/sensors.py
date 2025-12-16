"""Sensor outputs and Observation construction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Iterable, Tuple

from brain.contracts import Observation, VisionRay
from core.entities import Agent
from core.rng import RNGStream


def _normalize(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _compute_egomotion(agent: Agent) -> Tuple[float, float]:
    dx = agent.pos[0] - agent.last_pos[0]
    dy = agent.pos[1] - agent.last_pos[1]
    # Project displacement onto heading to preserve forward/backward sign.
    heading = agent.heading
    distance = dx * math.cos(heading) + dy * math.sin(heading)
    turn_delta = agent.heading - agent.last_heading
    # Clamp egomotion to normalized ranges to satisfy validation.
    return _normalize(distance, -1.0, 1.0), _normalize(turn_delta, -math.pi, math.pi)


def gather_observation(agent: Agent, vision_stream: RNGStream, noise_stream: RNGStream) -> Observation:
    """Collect normalized sensor outputs for the agent."""
    rays = tuple(
        VisionRay(
            dist=_normalize(vision_stream.uniform(0.0, 1.0), 0.0, 1.0),
            obj_type="",
            angle=_normalize(vision_stream.uniform(-math.pi, math.pi), -math.pi, math.pi),
        )
        for _ in range(3)
    )
    whisker_hits = (noise_stream.random() > 0.8, noise_stream.random() > 0.8)
    pain_signal = _normalize(noise_stream.uniform(0.0, 1.0), 0.0, 1.0)
    forward_delta, turn_delta = _compute_egomotion(agent)
    obs = Observation(
        vision_rays=rays,
        whisker_hits=whisker_hits,
        pain_signal=pain_signal,
        forward_delta=forward_delta,
        turn_delta=turn_delta,
    )
    obs.validate()
    return obs


def observation_checksum(obs: Observation) -> str:
    """Compute a stable checksum of an observation."""
    data = asdict(obs)
    # Ensure deterministic ordering via sorted items of nested structures.
    def _normalize_obj(obj):
        if isinstance(obj, dict):
            return {k: _normalize_obj(obj[k]) for k in sorted(obj)}
        if isinstance(obj, list):
            return [_normalize_obj(v) for v in obj]
        return obj

    normalized = _normalize_obj(data)
    payload = repr(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["gather_observation", "observation_checksum"]
