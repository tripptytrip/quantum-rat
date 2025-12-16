"""Criticality validation sweep with basic monotonicity assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from brain.systems.criticality import CriticalityConfig, CriticalityField
from core.rng import spawn_streams


@dataclass
class SweepResult:
    coupling: float
    mean_avalanche: float
    avalanche_rate: float
    mean_kappa: float


def run_trial(coupling: float, steps: int = 200) -> SweepResult:
    streams = spawn_streams(seed=123, names=["criticality"])
    field = CriticalityField(stream=streams["criticality"], config=CriticalityConfig(coupling=coupling))
    avalanche_sizes: List[int] = []
    kappa_values: List[float] = []
    for _ in range(steps):
        metrics = field.step()
        if metrics.avalanche_size > 0:
            avalanche_sizes.append(metrics.avalanche_size)
        kappa_values.append(metrics.kappa)
    mean_avalanche = sum(avalanche_sizes) / len(avalanche_sizes) if avalanche_sizes else 0.0
    avalanche_rate = len(avalanche_sizes) / steps
    mean_kappa = sum(kappa_values) / len(kappa_values) if kappa_values else 0.0
    return SweepResult(
        coupling=coupling,
        mean_avalanche=mean_avalanche,
        avalanche_rate=avalanche_rate,
        mean_kappa=mean_kappa,
    )


def run_sweep(couplings: Tuple[float, float, float] = (0.05, 0.25, 0.6), steps: int = 200) -> List[SweepResult]:
    return [run_trial(c, steps=steps) for c in couplings]


def assert_monotonic_trends(results: List[SweepResult]) -> None:
    # Expect increasing avalanche size and rate as coupling grows.
    sizes = [r.mean_avalanche for r in results]
    rates = [r.avalanche_rate for r in results]
    kappas = [r.mean_kappa for r in results]
    assert sizes[0] <= sizes[1] <= sizes[2], "Mean avalanche size not monotonic"
    assert rates[0] <= rates[1] <= rates[2], "Avalanche rate not monotonic"
    assert kappas[0] <= kappas[1] <= kappas[2], "Kappa not monotonic"


def main() -> None:
    results = run_sweep()
    assert_monotonic_trends(results)
    for r in results:
        print(
            f"coupling={r.coupling:.2f} mean_avalanche={r.mean_avalanche:.2f} "
            f"avalanche_rate={r.avalanche_rate:.3f} mean_kappa={r.mean_kappa:.3f}"
        )


if __name__ == "__main__":
    main()
