"""Simple deterministic astrocyte energy model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Astrocyte:
    atp: float = 1.0
    glycogen: float = 3.0
    atp_baseline: float = 1.0
    atp_cost: float = 0.05
    glycogen_to_atp_yield: float = 0.5
    glycogen_recharge: float = 0.02
    atp_floor: float = 0.0

    def tick(self, demand: float = 1.0) -> float:
        """Update energy stores; return throttle factor [0,1] based on ATP level."""
        cost = self.atp_cost * max(demand, 0.0)
        if self.atp >= cost:
            self.atp -= cost
        else:
            deficit = cost - self.atp
            self.atp = self.atp_floor
            pull = min(self.glycogen, deficit / max(self.glycogen_to_atp_yield, 1e-6))
            self.glycogen -= pull
            self.atp += pull * self.glycogen_to_atp_yield

        # Passive recharge from glycogen when possible.
        if self.glycogen > 0 and self.atp < self.atp_baseline:
            transfer = min(self.glycogen_recharge, self.glycogen)
            self.glycogen -= transfer
            self.atp += transfer * self.glycogen_to_atp_yield

        # Clamp values to non-negative ranges.
        self.atp = max(self.atp_floor, self.atp)
        self.glycogen = max(0.0, self.glycogen)

        return max(0.0, min(1.0, self.atp / self.atp_baseline))


__all__ = ["Astrocyte"]
