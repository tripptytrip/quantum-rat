import copy
import hashlib

import numpy as np

import app


def state_fingerprint(sim: app.SimulationState) -> str:
    h = hashlib.sha256()
    h.update(sim.rat_pos.astype(np.float64).tobytes())
    h.update(sim.rat_vel.astype(np.float64).tobytes())
    h.update(np.array(sim.brain.basal_ganglia.w_gate, dtype=np.float64).tobytes())
    h.update(np.array(sim.brain.basal_ganglia.w_motor, dtype=np.float64).tobytes())
    h.update(sim.brain.soma.psi.astype(np.complex128).tobytes())
    h.update(sim.brain.mt_theta.psi.astype(np.complex128).tobytes())
    h.update(np.array([sim.score, sim.deaths, sim.frames_alive], dtype=np.int64).tobytes())
    return h.hexdigest()


def run_steps(sim: app.SimulationState, steps: int):
    for _ in range(steps):
        sim.update_whiskers()
        sim.process_vision()
        sim.frames_alive += 1
        # Minimal per-step work: avoid HTTP layer, reuse core logic where possible
        sim.brain.update_hd_cells(0.0)
        sim.rat_heading = float(np.arctan2(sim.rat_vel[1], sim.rat_vel[0]))
        sim.brain.process_votes(
            sim.frustration,
            sim.dopamine,
            sim.rat_vel,
            0,
            0,
            len(sim.pheromones),
            sim.brain.get_head_direction(),
            sim.vision_buffer,
            anesthetic_level=0.0,
            fear_gain=1.0,
            memory_gain=1.0,
            energy_constraint=True,
            whisker_hits=sim.whisker_hits,
            collision=sim.last_collision,
            sensory_blend=0.0,
            enable_sensory_cortex=False,
            enable_thalamus=False,
            extra_arousal=0.0,
            rat_pos=sim.rat_pos,
            panic=0.0,
            cortisol=0.0,
            near_death=False,
            panic_trn_gain=0.5,
            replay_len=int(sim.config.get("replay_len", 240)),
            dt=float(sim.config.get("dt", 0.05)),
        )


def test_determinism_smoke():
    base = app.SimulationState()
    base.config["deterministic"] = 1.0
    base.config["seed"] = 123
    base.reseed(123)
    base.deterministic_reset()
    base.apply_runtime_config()

    sim_a = base
    sim_b = copy.deepcopy(base)

    steps = 50
    run_steps(sim_a, steps)
    run_steps(sim_b, steps)

    assert state_fingerprint(sim_a) == state_fingerprint(sim_b)
