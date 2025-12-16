"""Pipeline ordering for the simulation tick loop."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple

# Order derived from spec spine: prep -> world -> brain -> metrics/logging.
PIPELINE_ORDER: Tuple[str, ...] = (
    "pre_tick",
    "physiology",
    "world",
    "sensors",
    "trn",
    "spatial",
    "wm",
    "brain",
    "log",
)

# The context is intentionally duck-typed to avoid import cycles with Engine.
Step = Callable[[Any], None]


class Pipeline:
    """Executes named steps in a fixed, explicit order."""

    def __init__(self, handlers: Dict[str, Step], order: Iterable[str] = PIPELINE_ORDER) -> None:
        self.handlers = handlers
        self.order = tuple(order)

    def run(self, context: Any) -> None:
        for name in self.order:
            handler = self.handlers.get(name)
            if handler:
                handler(context)


__all__ = ["PIPELINE_ORDER", "Pipeline", "Step"]
