from core.engine import Engine


def test_wm_bounded_and_actions_deterministic():
    eng_a = Engine(seed=1337)
    eng_b = Engine(seed=1337)
    t1 = eng_a.run(200, reset=True)
    t2 = eng_b.run(200, reset=True)

    seq1 = [(td.action_name, td.action_thrust, td.action_turn, td.wm_load, td.wm_novelty) for td in t1]
    seq2 = [(td.action_name, td.action_thrust, td.action_turn, td.wm_load, td.wm_novelty) for td in t2]
    assert seq1 == seq2

    capacity_reached = any(td.wm_load >= 32 for td in t1)
    assert capacity_reached
    assert all(td.wm_load <= 32 for td in t1)

    action_set = set(td.action_name for td in t1 if not td.microsleep_active)
    assert len(action_set) > 1  # non-trivial actions


def test_microsleep_forces_rest_action():
    eng = Engine(seed=1337)
    ticks = eng.run(300, reset=True)
    for td in ticks:
        if td.microsleep_active:
            assert td.action_name == "REST"
