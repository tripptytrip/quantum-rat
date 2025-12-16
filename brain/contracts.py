"""Data contracts for Brain IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VisionRay:
    """Vision ray sample."""

    dist: float  # normalized 0-1 distance
    obj_type: str  # semantic label (may be empty for no hit)
    angle: float  # radians relative to heading

    def validate(self) -> None:
        if not 0.0 <= self.dist <= 1.0:
            raise ValueError(f"dist must be in [0,1], got {self.dist}")
        if not -3.14159 <= self.angle <= 3.14159:
            raise ValueError(f"angle must be in [-pi, pi], got {self.angle}")


@dataclass(frozen=True)
class Observation:
    """Immutable snapshot provided to the Brain."""

    vision_rays: Tuple[VisionRay, ...]
    whisker_hits: Tuple[bool, bool]
    pain_signal: float  # 0.0 - 1.0
    forward_delta: float  # egomotion: signed distance traveled this tick
    turn_delta: float  # egomotion: heading change this tick (radians)

    def validate(self) -> None:
        if len(self.whisker_hits) != 2 or not all(isinstance(v, bool) for v in self.whisker_hits):
            raise ValueError("whisker_hits must be a tuple of two booleans")
        if not 0.0 <= self.pain_signal <= 1.0:
            raise ValueError(f"pain_signal must be in [0,1], got {self.pain_signal}")
        if not -1.0 <= self.forward_delta <= 1.0:
            raise ValueError(f"forward_delta must be normalized to [-1,1], got {self.forward_delta}")
        if not -3.14159 <= self.turn_delta <= 3.14159:
            raise ValueError(f"turn_delta must be in [-pi, pi], got {self.turn_delta}")
        for ray in self.vision_rays:
            ray.validate()


@dataclass(frozen=True)
class Action:
    """Minimal action contract."""

    name: str
    thrust: float  # forward magnitude
    turn: float  # turn delta (radians)
