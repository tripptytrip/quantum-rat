from metrics.hash import tick_hash
from core.engine import Engine


def test_observation_sequence_deterministic_and_nontrivial():
    engine_a = Engine(seed=1234)
    engine_b = Engine(seed=1234)

    trace_a = engine_a.run(30, reset=True)
    trace_b = engine_b.run(30, reset=True)

    # Deterministic across runs
    hashes_a = [tick_hash(t) for t in trace_a]
    hashes_b = [tick_hash(t) for t in trace_b]
    assert hashes_a == hashes_b

    # Observations are produced and change over time
    fwd_deltas = [t.obs_forward_delta for t in trace_a]
    assert any(v != 0.0 for v in fwd_deltas)
    assert len(set(fwd_deltas[:10])) > 1
