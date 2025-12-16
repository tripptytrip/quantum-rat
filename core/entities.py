"""Entity definitions for the simulation world."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

Vector = Tuple[float, float]


@dataclass
class Entity:
    """Base entity."""

    id: int
    pos: Vector
    heading: float = 0.0  # radians


@dataclass
class Agent(Entity):
    """Agent entity with egomotion tracking."""

    last_pos: Vector = field(default_factory=lambda: (0.0, 0.0))
    last_heading: float = 0.0

    def remember_state(self) -> None:
        self.last_pos = self.pos
        self.last_heading = self.heading

    def apply_motion(self, forward_delta: float, turn_delta: float) -> None:
        """Update heading and position based on forward distance and turn."""
        self.remember_state()
        self.heading += turn_delta
        dx = forward_delta * math.cos(self.heading)
        dy = forward_delta * math.sin(self.heading)
        x, y = self.pos
        self.pos = (x + dx, y + dy)
