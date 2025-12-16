"""CriticalityField lattice dynamics with avalanche tracking and kappa EMA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from core.rng import RNGStream


@dataclass
class CriticalityConfig:
    field_size: int = 16
    fire_threshold: float = 0.6
    coupling: float = 0.25
    noise_drive: float = 0.02
    refractory_reset: bool = True
    kappa_ema_alpha: float = 0.05
    max_activity: float = 1.0


@dataclass
class CriticalityMetrics:
    active: int
    avalanche_size: int
    kappa: float


class CriticalityField:
    """Simple branching-process-inspired lattice."""

    def __init__(self, stream: RNGStream, config: CriticalityConfig | None = None) -> None:
        self.config = config or CriticalityConfig()
        self.stream: RNGStream = stream
        size = self.config.field_size
        self.field = [[0.0 for _ in range(size)] for _ in range(size)]
        self.prev_active = 0
        self.kappa = 0.0
        self._avalanche_active = False
        self._avalanche_size = 0

    def _neighbor_indices(self, i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        size = self.config.field_size
        nbrs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = (i + di) % size, (j + dj) % size
            nbrs.append((ni, nj))
        return tuple(nbrs)

    def _propagate(self) -> int:
        size = self.config.field_size
        new_field = [[0.0 for _ in range(size)] for _ in range(size)]
        active_count = 0
        for i in range(size):
            for j in range(size):
                drive = self.stream.uniform(0.0, self.config.noise_drive)
                influence = 0.0
                for ni, nj in self._neighbor_indices(i, j):
                    influence += self.field[ni][nj] * self.config.coupling
                potential = drive + influence
                if potential > self.config.fire_threshold:
                    active_count += 1
                    if self.config.refractory_reset:
                        new_field[i][j] = 0.0
                    else:
                        new_field[i][j] = min(self.config.max_activity, potential - self.config.fire_threshold)
                else:
                    new_field[i][j] = potential
        self.field = new_field
        return active_count

    def _update_avalanche(self, active: int) -> int:
        avalanche_size = 0
        if active > 0:
            self._avalanche_active = True
            self._avalanche_size += active
        elif self._avalanche_active:
            avalanche_size = self._avalanche_size
            self._avalanche_active = False
            self._avalanche_size = 0
        return avalanche_size

    def _update_kappa(self, active: int) -> float:
        if self.prev_active > 0:
            ratio = active / float(self.prev_active)
            alpha = self.config.kappa_ema_alpha
            self.kappa = (1 - alpha) * self.kappa + alpha * ratio
        self.prev_active = active
        return self.kappa

    def step(self) -> CriticalityMetrics:
        active = self._propagate()
        avalanche_size = self._update_avalanche(active)
        kappa = self._update_kappa(active)
        return CriticalityMetrics(active=active, avalanche_size=avalanche_size, kappa=kappa)


__all__ = ["CriticalityField", "CriticalityConfig", "CriticalityMetrics"]
