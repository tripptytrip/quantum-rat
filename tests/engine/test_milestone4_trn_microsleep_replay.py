from core.engine import Engine


def test_replay_only_during_microsleep_and_trn_states_present():
    engine = Engine(seed=1337)
    ticks = engine.run(400, reset=True)

    assert any(td.microsleep_active for td in ticks)
    assert all((not td.replay_active) or td.microsleep_active for td in ticks)
    allowed = {"OPEN", "NARROW", "CLOSED"}
    assert all(td.trn_state in allowed for td in ticks)


def test_trn_reproducible():
    eng_a = Engine(seed=2025)
    eng_b = Engine(seed=2025)
    t1 = eng_a.run(150, reset=True)
    t2 = eng_b.run(150, reset=True)
    seq1 = [(i, td.trn_state, td.microsleep_active, td.replay_active, td.replay_index) for i, td in enumerate(t1)]
    seq2 = [(i, td.trn_state, td.microsleep_active, td.replay_active, td.replay_index) for i, td in enumerate(t2)]
    assert seq1 == seq2
