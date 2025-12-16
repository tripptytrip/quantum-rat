from core.engine import Engine


def test_engine_criticality_deterministic():
    engine_a = Engine(seed=2024)
    engine_b = Engine(seed=2024)

    trace_a = engine_a.run(20, reset=True)
    trace_b = engine_b.run(20, reset=True)

    pairs_a = [(t.kappa, t.avalanche_size) for t in trace_a]
    pairs_b = [(t.kappa, t.avalanche_size) for t in trace_b]
    assert pairs_a == pairs_b
    assert all(isinstance(t.avalanche_size, int) and t.avalanche_size >= 0 for t in trace_a)
