## Call chain snippets
- Pipeline order: `pre_tick -> physiology -> world -> sensors -> trn -> spatial -> wm -> brain -> log` (`core/pipeline.py`).
- `core/engine.py`:
  - `_wm_step` updates bounded WM from `Observation` + `place_id` + checksum.
  - `_brain_step` calls `select_action(...)` using Observation, WM novelty, TRN gain, microsleep flag, place_id; sets `self.last_action`.
  - `_world_step` applies `self.last_action` via `World.step(action=..., energy_scale=...)` (no RNG).
- `core/world.py` applies `Action` deterministically to agent heading/position; no random wander.

## 30-tick sample (seed=1337)
| tick | trn_state | microsleep_active | wm_load | wm_novelty | action_name | thrust | turn | obs_forward_delta | obs_turn_delta | grid_x | grid_y | place_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | OPEN | False | 1 | 1.0 | FORWARD | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 1 | OPEN | False | 2 | 1.0 | FORWARD | 1.0 | 0.0 | 0.92 | 0.0 | 0.92 | 0.0 | 65536 |
| 2 | OPEN | False | 3 | 1.0 | FORWARD | 1.0 | 0.0 | 0.88 | 0.0 | 1.8 | 0.0 | 196608 |
| 3 | OPEN | False | 4 | 1.0 | FORWARD | 1.0 | 0.0 | 0.84 | 0.0 | 2.64 | 0.0 | 327680 |
| 4 | OPEN | False | 5 | 1.0 | FORWARD | 1.0 | 0.0 | 0.8 | 0.0 | 3.44 | 0.0 | 393216 |
| 5 | OPEN | False | 6 | 1.0 | REST | 0.0 | 0.0 | 0.76 | 0.0 | 4.2 | 0.0 | 524288 |
| 6 | OPEN | False | 7 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 4.2 | 0.0 | 524288 |
| 7 | OPEN | False | 8 | 1.0 | FORWARD | 1.0 | 0.0 | 0.0 | 0.0 | 4.2 | 0.0 | 524288 |
| 8 | OPEN | False | 9 | 1.0 | FORWARD | 1.0 | 0.0 | 0.64 | 0.0 | 4.84 | 0.0 | 589824 |
| 9 | OPEN | False | 10 | 1.0 | FORWARD | 1.0 | 0.0 | 0.6 | 0.0 | 5.44 | 0.0 | 655360 |
| 10 | OPEN | False | 11 | 1.0 | FORWARD | 1.0 | 0.0 | 0.56 | 0.0 | 6.0 | 0.0 | 720896 |
| 11 | NARROW | False | 12 | 1.0 | FORWARD | 1.0 | 0.0 | 0.52 | 0.0 | 6.208 | 0.0 | 786432 |
| 12 | NARROW | False | 13 | 1.0 | REST | 0.0 | 0.0 | 0.48 | 0.0 | 6.4 | 0.0 | 786432 |
| 13 | NARROW | False | 14 | 1.0 | FORWARD | 1.0 | 0.0 | 0.0 | 0.0 | 6.4 | 0.0 | 786432 |
| 14 | NARROW | False | 15 | 1.0 | REST | 0.0 | 0.0 | 0.4 | 0.0 | 6.56 | 0.0 | 851968 |
| 15 | NARROW | False | 16 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 16 | CLOSED | False | 17 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 17 | CLOSED | False | 18 | 1.0 | TURN_LEFT | 0.3 | 0.3 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 18 | CLOSED | False | 19 | 1.0 | TURN_LEFT | 0.3 | 0.3 | 0.072 | 0.072 | 6.56 | 0.0 | 851968 |
| 19 | CLOSED | False | 20 | 1.0 | REST | 0.0 | 0.0 | 0.06 | 0.06 | 6.56 | 0.0 | 851968 |
| 20 | CLOSED | False | 21 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 21 | CLOSED | False | 22 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 22 | CLOSED | False | 23 | 1.0 | TURN_LEFT | 0.3 | 0.3 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 23 | CLOSED | False | 24 | 1.0 | REST | 0.0 | 0.0 | 0.012 | 0.012 | 6.56 | 0.0 | 851968 |
| 24 | CLOSED | False | 25 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 25 | CLOSED | False | 26 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 26 | CLOSED | True | 27 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 27 | CLOSED | True | 28 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 28 | CLOSED | True | 29 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |
| 29 | CLOSED | True | 30 | 1.0 | REST | 0.0 | 0.0 | 0.0 | 0.0 | 6.56 | 0.0 | 851968 |

## Determinism proof (first 20 ticks, seed=1337)
- Run #1 tuples: `[(0, 1, 1.0, 'FORWARD', 1.0, 0.0), (1, 2, 1.0, 'FORWARD', 1.0, 0.0), (2, 3, 1.0, 'FORWARD', 1.0, 0.0), (3, 4, 1.0, 'FORWARD', 1.0, 0.0), (4, 5, 1.0, 'FORWARD', 1.0, 0.0), (5, 6, 1.0, 'REST', 0.0, 0.0), (6, 7, 1.0, 'REST', 0.0, 0.0), (7, 8, 1.0, 'FORWARD', 1.0, 0.0), (8, 9, 1.0, 'FORWARD', 1.0, 0.0), (9, 10, 1.0, 'FORWARD', 1.0, 0.0), (10, 11, 1.0, 'FORWARD', 1.0, 0.0), (11, 12, 1.0, 'FORWARD', 1.0, 0.0), (12, 13, 1.0, 'REST', 0.0, 0.0), (13, 14, 1.0, 'FORWARD', 1.0, 0.0), (14, 15, 1.0, 'REST', 0.0, 0.0), (15, 16, 1.0, 'REST', 0.0, 0.0), (16, 17, 1.0, 'REST', 0.0, 0.0), (17, 18, 1.0, 'TURN_LEFT', 0.3, 0.3), (18, 19, 1.0, 'TURN_LEFT', 0.3, 0.3), (19, 20, 1.0, 'REST', 0.0, 0.0)]`
- Run #2 (same seed) matches exactly.

## Command transcript
- `python3 tools/update_determinism_baseline.py --i-know-what-im-doing` — exit 0
- `pytest -q` — exit 0
- `pytest -q tests/determinism/test_trace_hash.py` — exit 0
- `pytest -q tests/engine/test_milestone5_action_wm_determinism.py` — exit 0
