from __future__ import annotations

from typing import Any, Dict

from experiments.protocols.base import Protocol
from metrics.schema import TickData


class OpenFieldProtocol(Protocol):
    name = "open_field"

    def __init__(self, config: dict | None = None) -> None:
        self.distance_travelled = 0.0
        self.turn_energy = 0.0
        self.microsleep_count = 0
        self.replay_ticks = 0
        self.ticks_run = 0

    def setup(self, engine: Any) -> None:
        # Place agent at origin facing 0
        engine.agent.pos = (0.0, 0.0)
        engine.agent.heading = 0.0

    def on_tick(self, engine: Any, tickdata: TickData, tick_index: int) -> None:
        self.ticks_run += 1
        self.distance_travelled += abs(tickdata.obs_forward_delta)
        self.turn_energy += abs(tickdata.obs_turn_delta)
        if tickdata.microsleep_active:
            self.microsleep_count += 1
        if tickdata.replay_active:
            self.replay_ticks += 1

    def is_done(self, engine: Any, tickdata: TickData, tick_index: int) -> bool:
        return False

    def summarize(self) -> Dict[str, float]:
        exploration_score = (
            self.distance_travelled - 0.5 * self.replay_ticks - 2.0 * self.microsleep_count
        )
        return {
            "ticks_run": self.ticks_run,
            "distance_travelled": self.distance_travelled,
            "turn_energy": self.turn_energy,
            "microsleep_count": self.microsleep_count,
            "replay_ticks": self.replay_ticks,
            "exploration_score": exploration_score,
            "score": exploration_score,
        }


__all__ = ["OpenFieldProtocol"]
