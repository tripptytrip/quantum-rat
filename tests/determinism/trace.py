"""Deterministic trace generator for tests and baselines."""

from __future__ import annotations

from typing import Iterable, List

from core.engine import Engine
from metrics.schema import TickData


def generate_trace(seed: int, ticks: int, *, agent_offset: int = 0) -> List[TickData]:
    """Produce a deterministic TickData sequence for testing determinism gates."""
    engine = Engine(seed=seed, agent_offset=agent_offset)
    return engine.run(ticks)


__all__ = ["generate_trace"]
