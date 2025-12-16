"""Hash utilities for TickData traces."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Union

from .logger import _canonical_json, _normalize_tick
from .schema import TickData

TickLike = Union[TickData, Mapping[str, Any]]


def tick_hash(tick: TickLike) -> str:
    """Return a deterministic hash for a single tick."""
    payload = _normalize_tick(tick)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunHash:
    """Accumulator for a full run; feeds on per-tick hashes."""

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()

    def update(self, tick: TickLike) -> str:
        digest = tick_hash(tick)
        self._hasher.update(digest.encode("utf-8"))
        return digest

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


__all__ = ["RunHash", "tick_hash"]
