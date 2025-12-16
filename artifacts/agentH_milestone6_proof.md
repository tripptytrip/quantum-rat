## Protocol list
- `python3 -m experiments.runner --list` output:
```
morris_water_maze
open_field
survival_arena
t_maze
```

## Water Maze determinism proof
- `diff -q /tmp/mwm_1337/summary.json /tmp/mwm_1337_b/summary.json` → no diff
- First 50 tick lines identical: `diff -q /tmp/a /tmp/b` → no diff

## Summaries
- morris_water_maze (seed=1337, ticks=500):
```json
{"path_length":4.199999999999999,"platform_radius":1.0,"platform_reached":true,"platform_x":5.0,"platform_y":0.0,"pool_radius":10.0,"protocol":"morris_water_maze","protocol_config":{},"run_hash":"fab0e640d92b5dfdf32162d0c68b5070e2aed5a60f77a48c0b0a9384927cd030","schema_version":"2.1.4","score":99.33,"seed":1337,"thigmotaxis_ticks":0,"ticks_requested":500,"ticks_run":6,"time_to_platform":5}
```
- survival_arena (seed=1337, ticks=500):
```json
{"arena_radius":10.0,"damage":20.0,"damage_limit":20.0,"dead":true,"hazard_ticks":20,"protocol":"survival_arena","protocol_config":{},"run_hash":"e005f8d69af5f9584390e2e6cd1685d8c9b69f0cc3cbe3597def8e0827390c8c","safe_ticks":0,"schema_version":"2.1.4","score":-6.0,"seed":1337,"ticks_requested":500,"ticks_run":34}
```

## Command transcript
- `python3 -m experiments.runner --list` (exit 0)
- `python3 -m experiments.runner --protocol morris_water_maze --seed 1337 --ticks 500 --out /tmp/mwm_1337` (exit 0)
- `python3 -m experiments.runner --protocol morris_water_maze --seed 1337 --ticks 500 --out /tmp/mwm_1337_b` (exit 0)
- `diff -q /tmp/mwm_1337/summary.json /tmp/mwm_1337_b/summary.json` (exit 0)
- `head -n 50 /tmp/mwm_1337/ticks.jsonl > /tmp/a; head -n 50 /tmp/mwm_1337_b/ticks.jsonl > /tmp/b; diff -q /tmp/a /tmp/b` (exit 0)
- `python3 -m experiments.runner --protocol survival_arena --seed 1337 --ticks 500 --out /tmp/surv_1337` (exit 0)
