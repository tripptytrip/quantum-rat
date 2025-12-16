"""Bounded working memory updated deterministically from Observation and spatial cues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

from brain.contracts import Observation


@dataclass
class WorkingMemoryState:
    load: int = 0
    novelty: float = 0.0
    last_checksum: str = ""


class WorkingMemory:
    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.buffer: Deque[Dict[str, float | int | str]] = deque(maxlen=capacity)
        self.state = WorkingMemoryState()

    def update(self, observation: Observation, place_id: int, obs_checksum: str) -> WorkingMemoryState:
        entry = {
            "pain": observation.pain_signal,
            "forward": observation.forward_delta,
            "turn": observation.turn_delta,
            "place_id": place_id,
            "checksum": obs_checksum,
        }
        self.buffer.append(entry)
        self.state.load = len(self.buffer)
        self.state.novelty = 1.0 if obs_checksum != self.state.last_checksum else 0.0
        self.state.last_checksum = obs_checksum
        return self.state


__all__ = ["WorkingMemory", "WorkingMemoryState"]
