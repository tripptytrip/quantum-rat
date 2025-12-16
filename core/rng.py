"""Deterministic RNG authority for Agent 02.

All randomness should flow through this module to ensure reproducibility.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _derive_seed(base_seed: int, name: str, agent_offset: int = 0) -> int:
    """Derive a deterministic child seed from a base seed, stream name, and agent offset."""
    payload = f"{base_seed}:{agent_offset}:{name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Constrain to Python's Random seed range while preserving entropy.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass
class RNGStream:
    """Named random stream with a dedicated Random instance."""

    seed: int
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def random(self) -> float:
        return self._random.random()

    def randint(self, a: int, b: int) -> int:
        return self._random.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._random.uniform(a, b)

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._random.choice(seq)

    def shuffle(self, seq: list[Any]) -> None:
        self._random.shuffle(seq)

    def sample(self, population: Sequence[Any], k: int) -> list[Any]:
        return self._random.sample(population, k)


class RNG:
    """Seeded RNG factory that spawns named, isolated streams."""

    def __init__(self, seed: int, agent_offset: int = 0) -> None:
        self.seed = seed
        self.agent_offset = agent_offset
        self._streams: Dict[str, RNGStream] = {}

    def stream(self, name: str) -> RNGStream:
        """Return a deterministic RNGStream for a given name (cached)."""
        if name not in self._streams:
            child_seed = _derive_seed(self.seed, name=name, agent_offset=self.agent_offset)
            self._streams[name] = RNGStream(seed=child_seed)
        return self._streams[name]

    def scoped(self, *, agent_offset: Optional[int] = None) -> "RNG":
        """Create a new RNG factory with the same base seed but a different agent offset."""
        return RNG(seed=self.seed, agent_offset=agent_offset if agent_offset is not None else self.agent_offset)


def spawn_streams(seed: int, names: Iterable[str], agent_offset: int = 0) -> Mapping[str, RNGStream]:
    """Convenience helper to build multiple streams in one call."""
    rng = RNG(seed=seed, agent_offset=agent_offset)
    return {name: rng.stream(name) for name in names}
