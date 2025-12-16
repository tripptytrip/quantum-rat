
# AgentJ Milestone 7 Proof

## A) Config + Protocol List

**CLI Command:**
```bash
python3 -m tournaments.runner --seed 1337 --n-agents 4 --ticks 200 --out /tmp/tourn_1337 --protocols open_field,morris_water_maze
```

**`config.json`:**
```json
{"protocols":[{"name":"open_field","ticks":200},{"name":"morris_water_maze","ticks":200}],"seed":1337}
```

## B) Determinism Proof

**`leaderboard.json` comparison:**
```bash
$ diff -q /tmp/tourn_1337/leaderboard.json /tmp/tourn_1337_b/leaderboard.json || true
```
*(No output, files are identical)*

**`agents.json` comparison:**
```bash
$ diff -q /tmp/tourn_1337/agents.json /tmp/tourn_1337_b/agents.json || true
```
*(No output, files are identical)*


## C) Leaderboard Sample

**`leaderboard.json`:**
```json
[{"agent_fingerprint":"6e1ccc6ca230f6af2833021debcd8e5688e6d20b78dcc9193acd030601d29d7b","agent_id":"agent_0001","protocols":{"morris_water_maze":{"agent_fingerprint":"6e1ccc6ca230f6af2833021debcd8e5688e6d20b78dcc9193acd030601d29d7b","agent_id":"agent_0001","agent_offset":1,"path_length":4.479999999999999,"platform_radius":1.0,"platform_reached":true,"platform_x":5.0,"platform_y":0.0,"pool_radius":10.0,"protocol":"morris_water_maze","run_hash":"7f26debd0d8d0c94ccc39a65d4a5881b18e2e472f009fb41542040d2de86dd84","schema_version":"2.1.4","score":99.102,"seed":3172,"thigmotaxis_ticks":0,"ticks_requested":200,"ticks_run":10,"time_to_platform":9},"open_field":{"agent_fingerprint":"6e1ccc6ca230f6af2833021debcd8e5688e6d20b78dcc9193acd030601d29d7b","agent_id":"agent_0001","agent_offset":1,"distance_travelled":8.263999999999996,"exploration_score":-304.236,"microsleep_count":125,"protocol":"open_field","replay_ticks":125,"run_hash":"23ac633310a1342e20204fcf665f6bc180a863e76d4e39cfb9d5e7ff2d09b066","schema_version":"2.1.4","score":-304.236,"seed":2383,"ticks_requested":200,"ticks_run":200,"turn_energy":0.2639999999999997}},"total_score":-205.134},{"agent_fingerprint":"c008a77800d40d276ed1caf9435da743286c030f30de5c5cb59a971e184be87d","agent_id":"agent_0000","protocols":{"morris_water_maze":{"agent_fingerprint":"c008a77800d40d276ed1caf9435da743286c030f30de5c5cb59a971e184be87d","agent_id":"agent_0000","agent_offset":0,"path_length":4.199999999999999,"platform_radius":1.0,"platform_reached":true,"platform_x":5.0,"platform_y":0.0,"pool_radius":10.0,"protocol":"morris_water_maze","run_hash":"edaa81b858f6f16329c736d2db1903ad01d144edb63553c6d5f7ef9d70d62980","schema_version":"2.1.4","score":99.33,"seed":3171,"thigmotaxis_ticks":0,"ticks_requested":200,"ticks_run":6,"time_to_platform":5},"open_field":{"agent_fingerprint":"c008a77800d40d276ed1caf9435da743286c030f30de5c5cb59a971e184be87d","agent_id":"agent_0000","agent_offset":0,"distance_travelled":7.775999999999996,"exploration_score":-304.724,"microsleep_count":125,"protocol":"open_field","replay_ticks":125,"run_hash":"bf912a9de54e499461ad467b4eb8bf608cfd2abbc5c3cf521783ef421e53e10d","schema_version":"2.1.4","score":-304.724,"seed":2382,"ticks_requested":200,"ticks_run":200,"turn_energy":0.3359999999999993}},"total_score":-205.394},{"agent_fingerprint":"b4d80c387a85bc07b871c56096e3c3b47452105e7812280ca3030fe1d8a06048","agent_id":"agent_0002","protocols":{"morris_water_maze":{"agent_fingerprint":"b4d80c387a85bc07b871c56096e3c3b47452105e7812280ca3030fe1d8a06048","agent_id":"agent_0002","agent_offset":2,"path_length":4.599999999999999,"platform_radius":1.0,"platform_reached":true,"platform_x":5.0,"platform_y":0.0,"pool_radius":10.0,"protocol":"morris_water_maze","run_hash":"40be9f9b4a5a9bbea30f9564f42bc131659a32f90c484c1e9cdd60cd97abd317","schema_version":"2.1.4","score":99.14,"seed":3173,"thigmotaxis_ticks":0,"ticks_requested":200,"ticks_run":9,"time_to_platform":8},"open_field":{"agent_fingerprint":"b4d80c387a85bc07b871c56096e3c3b47452105e7812280ca3030fe1d8a06048","agent_id":"agent_0002","agent_offset":2,"distance_travelled":7.785999999999996,"exploration_score":-304.714,"microsleep_count":125,"protocol":"open_field","replay_ticks":125,"run_hash":"38beaa8615d7da9979ec14a72555b48fd43823eee30b430f1e65419cb00687a2","schema_version":"2.1.4","score":-304.714,"seed":2384,"ticks_requested":200,"ticks_run":200,"turn_energy":0.1859999999999999}},"total_score":-205.574},{"agent_fingerprint":"8de5e7aa73d105d015987445d1597bfb7d8b79abb0c26f2d4584e19dff5ec5cc","agent_id":"agent_0003","protocols":{"morris_water_maze":{"agent_fingerprint":"8de5e7aa73d105d015987445d1597bfb7d8b79abb0c26f2d4584e19dff5ec5cc","agent_id":"agent_0003","agent_offset":3,"path_length":4.479999999999999,"platform_radius":1.0,"platform_reached":true,"platform_x":5.0,"platform_y":0.0,"pool_radius":10.0,"protocol":"morris_water_maze","run_hash":"637313f1bb65fa6db41208d5e8d1eb620d6f4e9980f02ebb16635d38ab4ce0cc","schema_version":"2.1.4","score":99.052,"seed":3174,"thigmotaxis_ticks":0,"ticks_requested":200,"ticks_run":11,"time_to_platform":10},"open_field":{"agent_fingerprint":"8de5e7aa73d105d015987445d1597bfb7d8b79abb0c26f2d4584e19dff5ec5cc","agent_id":"agent_0003","agent_offset":3,"distance_travelled":6.693999999999996,"exploration_score":-305.806,"microsleep_count":125,"protocol":"open_field","replay_ticks":125,"run_hash":"b55fa9d47d1dd8620ab6a2b8454a74b9a8fa5fc127d8e74b62e4d3f4b67ba099","schema_version":"2.1.4","score":-305.806,"seed":2385,"ticks_requested":200,"ticks_run":200,"turn_energy":0.17400000000000002}},"total_score":-206.75399999999996}]
```

## D) Per-agent run hash proof

| agent_id   | fingerprint                                                        | protocol            | score     | run_hash                                                         | ticks_run |
|------------|--------------------------------------------------------------------|---------------------|-----------|------------------------------------------------------------------|-----------|
| agent_0000 | c008a77800d40d276ed1caf9435da743286c030f30de5c5cb59a971e184be87d | morris_water_maze   | 99.33     | edaa81b858f6f16329c736d2db1903ad01d144edb63553c6d5f7ef9d70d62980 | 6         |
| agent_0000 | c008a77800d40d276ed1caf9435da743286c030f30de5c5cb59a971e184be87d | open_field          | -304.724  | bf912a9de54e499461ad467b4eb8bf608cfd2abbc5c3cf521783ef421e53e10d | 200       |
| agent_0001 | 6e1ccc6ca230f6af2833021debcd8e5688e6d20b78dcc9193acd030601d29d7b | morris_water_maze   | 99.102    | 7f26debd0d8d0c94ccc39a65d4a5881b18e2e472f009fb41542040d2de86dd84 | 10        |
| agent_0001 | 6e1ccc6ca230f6af2833021debcd8e5688e6d20b78dcc9193acd030601d29d7b | open_field          | -304.236  | 23ac633310a1342e20204fcf665f6bc180a863e76d4e39cfb9d5e7ff2d09b066 | 200       |
| agent_0002 | b4d80c387a85bc07b871c56096e3c3b47452105e7812280ca3030fe1d8a06048 | morris_water_maze   | 99.14     | 40be9f9b4a5a9bbea30f9564f42bc131659a32f90c484c1e9cdd60cd97abd317 | 9         |
| agent_0002 | b4d80c387a85bc07b871c56096e3c3b47452105e7812280ca3030fe1d8a06048 | open_field          | -304.714  | 38beaa8615d7da9979ec14a72555b48fd43823eee30b430f1e65419cb00687a2 | 200       |
| agent_0003 | 8de5e7aa73d105d015987445d1597bfb7d8b79abb0c26f2d4584e19dff5ec5cc | morris_water_maze   | 99.052    | 637313f1bb65fa6db41208d5e8d1eb620d6f4e9980f02ebb16635d38ab4ce0cc | 11        |
| agent_0003 | 8de5e7aa73d105d015987445d1597bfb7d8b79abb0c26f2d4584e19dff5ec5cc | open_field          | -305.806  | b55fa9d47d1dd8620ab6a2b8454a74b9a8fa5fc127d8e74b62e4d3f4b67ba099 | 200       |


## E) Command transcript

```bash
$ pytest -q
................................X. [100%]
32 passed, 1 xpassed in 1.41s

$ python3 -m tournaments.runner --seed 1337 --n-agents 4 --ticks 200 --out /tmp/tourn_1337 --protocols open_field,morris_water_maze
Tournament finished. Results are in /tmp/tourn_1337

$ python3 -m tournaments.runner --seed 1337 --n-agents 4 --ticks 200 --out /tmp/tourn_1337_b --protocols open_field,morris_water_maze
Tournament finished. Results are in /tmp/tourn_1337_b

$ diff -q /tmp/tourn_1337/leaderboard.json /tmp/tourn_1337_b/leaderboard.json || true

$ diff -q /tmp/tourn_1337/agents.json /tmp/tourn_1337_b/agents.json || true
```
