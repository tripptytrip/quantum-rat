"""Deterministic neuromodulator updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.rng import RNGStream


@dataclass
class NeuromodulatorSystem:
    levels: Dict[str, float] = field(
        default_factory=lambda: {"DA": 0.0, "5HT": 0.0, "NE": 0.0, "ACh": 0.0}
    )
    drift: float = 0.05

    def update(self, stream: RNGStream, throttle: float = 1.0) -> Dict[str, float]:
        """Apply deterministic drift with bounded randomness."""
        for key in self.levels:
            noise = stream.uniform(-self.drift, self.drift)
            level = self.levels[key] + noise
            level *= throttle  # energy-constrained modulation
            self.levels[key] = max(0.0, min(1.0, level))
        return dict(self.levels)


__all__ = ["NeuromodulatorSystem"]
