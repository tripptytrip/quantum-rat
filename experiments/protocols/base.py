"""Protocol base classes for headless experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from metrics.schema import TickData


class Protocol(ABC):
    name: str = "base"

    @abstractmethod
    def setup(self, engine: Any) -> None:
        """Deterministically configure engine/world before running."""

    @abstractmethod
    def on_tick(self, engine: Any, tickdata: TickData, tick_index: int) -> None:
        """Hook invoked every tick."""

    @abstractmethod
    def is_done(self, engine: Any, tickdata: TickData, tick_index: int) -> bool:
        """Return True to end the episode early."""

    @abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """Return summary metrics and score."""


__all__ = ["Protocol"]
