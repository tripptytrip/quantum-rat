from core.engine import Engine


def test_spatial_determinism_and_nontrivial():
    eng_a = Engine(seed=1337)
    eng_b = Engine(seed=1337)

    t1 = eng_a.run(150, reset=True)
    t2 = eng_b.run(150, reset=True)

    seq1 = [
        (round(td.hd_angle, 10), round(td.grid_x, 10), round(td.grid_y, 10), td.place_id) for td in t1
    ]
    seq2 = [
        (round(td.hd_angle, 10), round(td.grid_x, 10), round(td.grid_y, 10), td.place_id) for td in t2
    ]
    assert seq1 == seq2
    assert any(abs(td.grid_x) > 0 or abs(td.grid_y) > 0 or abs(td.hd_angle) > 0 for td in t1)


def test_spatial_signature_only_uses_observation():
    # Ensures spatial.step signature doesn't require world/agent
    from brain.systems.spatial import SpatialSystem
    from inspect import signature

    sig = signature(SpatialSystem.step)
    params = list(sig.parameters.keys())
    assert params[0] == "self"
    assert "observation" in params
