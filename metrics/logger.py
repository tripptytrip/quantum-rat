"""Deterministic JSONL logging for TickData."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import IO, Any, Mapping, Union

from .schema import TickData

TickLike = Union[TickData, Mapping[str, Any]]


def _normalize_tick(tick: TickLike) -> Mapping[str, Any]:
    if isinstance(tick, TickData):
        return tick.to_ordered_dict()
    if is_dataclass(tick):
        return asdict(tick)
    if isinstance(tick, Mapping):
        return dict(tick)
    raise TypeError(f"Unsupported tick type: {type(tick)!r}")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    # sort_keys enforces stable key order; separators remove whitespace noise.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class JsonlLogger:
    """Append-only JSONL writer with deterministic ordering."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def __enter__(self) -> "JsonlLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write_tick(self, tick: TickLike) -> None:
        if self._fh is None:
            self.open()
        assert self._fh is not None  # for type checkers
        payload = _normalize_tick(tick)
        line = _canonical_json(payload)
        self._fh.write(line + "\n")
        self._fh.flush()


__all__ = ["JsonlLogger", "_canonical_json", "_normalize_tick"]
