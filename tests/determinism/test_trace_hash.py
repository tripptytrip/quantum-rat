from __future__ import annotations

import json
from pathlib import Path

from metrics.hash import RunHash, tick_hash
from .trace import generate_trace


BASELINE_PATH = Path(__file__).with_name("baseline_hashes.json")
DEFAULT_SEED = 1337
DEFAULT_TICKS = 200


def load_baseline() -> list[dict]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline file missing: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


def build_current_trace(seed: int = DEFAULT_SEED, ticks: int = DEFAULT_TICKS) -> list[dict]:
    run_hash = RunHash()
    hashes = []
    for tick in generate_trace(seed=seed, ticks=ticks):
        digest = run_hash.update(tick)
        hashes.append({"tick": tick.tick, "hash": digest})
    hashes.append({"tick": "run", "hash": run_hash.hexdigest()})
    return hashes


def test_trace_hash_matches_baseline() -> None:
    baseline = load_baseline()
    current = build_current_trace()
    assert current == baseline, "Determinism regression: trace hash differs from baseline"
