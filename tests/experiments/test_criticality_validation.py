from experiments.criticality_validation import assert_monotonic_trends, run_sweep


def test_reduced_criticality_sweep_monotonic():
    results = run_sweep(steps=50)  # reduced for CI speed
    assert_monotonic_trends(results)
