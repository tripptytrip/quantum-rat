## Protocol list
- `python3 -m experiments.runner --list` →
```
open_field
t_maze
```

## Two-run determinism proof (Open Field, seed=1337, ticks=300)
- `diff -q /tmp/of_1337/summary.json /tmp/of_1337_b/summary.json` → no diff
- `diff -q /tmp/a /tmp/b` (first 50 tick lines) → no diff

## T-Maze outcome proof (seed=1337, ticks=2000)
`/tmp/tmaze_1337/summary.json`:
```json
{"ticks_run": 2000, "reward_reached": true, "time_to_reward": 9, "score": 9.91, "reward_threshold": 5.0, "protocol": "t_maze", "seed": 1337, "ticks_requested": 2000, "schema_version": "2.1.4", "run_hash": "51ee51df49f6dd6a5f96e1015a685363a6b0aad234988af9bbc304b9e9692c08"}
```

## Command transcript
- `python3 -m experiments.runner --protocol open_field --seed 1337 --ticks 300 --out /tmp/of_1337` (exit 0)
- `python3 -m experiments.runner --protocol open_field --seed 1337 --ticks 300 --out /tmp/of_1337_b` (exit 0)
- `diff -q /tmp/of_1337/summary.json /tmp/of_1337_b/summary.json` (exit 0)
- `head -n 50 /tmp/of_1337/ticks.jsonl > /tmp/a; head -n 50 /tmp/of_1337_b/ticks.jsonl > /tmp/b; diff -q /tmp/a /tmp/b` (exit 0)
- `python3 -m experiments.runner --protocol t_maze --seed 1337 --ticks 2000 --out /tmp/tmaze_1337` (exit 0)
- `python3 -m experiments.runner --protocol morris_water_maze --seed 1337 --ticks 500 --out /tmp/mwm_1337` (exit 0)
- `python3 -m experiments.runner --protocol morris_water_maze --seed 1337 --ticks 500 --out /tmp/mwm_1337_b` (exit 0)
- `diff -q /tmp/mwm_1337/summary.json /tmp/mwm_1337_b/summary.json` (exit 0)
- `head -n 50 /tmp/mwm_1337/ticks.jsonl > /tmp/a; head -n 50 /tmp/mwm_1337_b/ticks.jsonl > /tmp/b; diff -q /tmp/a /tmp/b` (exit 0)
- `python3 -m experiments.runner --protocol survival_arena --seed 1337 --ticks 500 --out /tmp/surv_1337` (exit 0)
- `pytest -q` (exit 0)
