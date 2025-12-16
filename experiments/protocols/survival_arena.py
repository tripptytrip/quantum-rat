from __future__ import annotations

from typing import Any, Dict

from experiments.protocols.base import Protocol
from metrics.schema import TickData


class SurvivalArenaProtocol(Protocol):
    name = "survival_arena"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.arena_r = float(cfg.get("arena_radius", 10.0))
        self.damage_limit = float(cfg.get("damage_limit", 20.0))
        self.hazard_x_thresh = float(cfg.get("hazard_x_thresh", 0.7 * self.arena_r))
        self.safe_x_thresh = float(cfg.get("safe_x_thresh", -0.7 * self.arena_r))
        self.damage = 0.0
        self.hazard_ticks = 0
        self.safe_ticks = 0
        self.ticks_survived = 0
        self.dead = False

    def setup(self, engine: Any) -> None:
        engine.agent.pos = (0.0, 0.0)
        engine.agent.heading = 0.0

    def on_tick(self, engine: Any, tickdata: TickData, tick_index: int) -> None:
        self.ticks_survived += 1
        x, y = tickdata.pos
        if x > self.hazard_x_thresh:
            self.hazard_ticks += 1
            self.damage += 1.0
        elif x < self.safe_x_thresh:
            self.safe_ticks += 1
            self.damage = max(0.0, self.damage - 0.5)

        if self.damage >= self.damage_limit:
            self.dead = True

    def is_done(self, engine: Any, tickdata: TickData, tick_index: int) -> bool:
        return self.dead

    def summarize(self) -> Dict[str, float | int | bool]:
        score = self.ticks_survived - 2.0 * self.hazard_ticks + 0.5 * self.safe_ticks
        return {
            "ticks_run": self.ticks_survived,
            "dead": self.dead,
            "hazard_ticks": self.hazard_ticks,
            "safe_ticks": self.safe_ticks,
            "damage": self.damage,
            "score": score,
            "arena_radius": self.arena_r,
            "damage_limit": self.damage_limit,
        }


__all__ = ["SurvivalArenaProtocol"]
