## Call chain proof

- `core/engine.py` pipeline handler order includes `sensors` between `world` and `brain` (`PIPELINE_ORDER`).
- `Engine._sensors_step` calls `gather_observation(self.agent, vision_stream=self.streams["sensors_vision"], noise_stream=self.streams["sensors_noise"])`, then stores `ctx.observation` and `ctx.observation_checksum`.
- `Engine._brain_step` signature uses `ctx.observation` (raises if missing) and proceeds with criticality/neuromod updates.

## Sample run (seed=1337, ticks=10)

| tick | pos (x,y) | egomotion_forward_delta | egomotion_turn_delta | observation_checksum | kappa |
| --- | --- | --- | --- | --- | --- |
| 0 | (-0.19345, -0.04153) | 0.19786 | 0.21145 | 4da46dacf4203b91b2742773e2cdf41f5f1cfa1e552f91c8526f830390d6db23 | 0.0 |
| 1 | (-0.37659, -0.07978) | 0.18709 | -0.00550 | 86d3cb28053400360496fbe5d96ee440ece5d1510e16855be576e2fb761f87db | 0.0 |
| 2 | (-0.16453, -0.03268) | 0.21722 | 0.01261 | 1a4bf2685cc0c0e74756d96603c398c679cf46cb0ad30fbe3f709f68fdfe5aac | 0.0 |
| 3 | (-0.09358, -0.01157) | 0.07402 | 0.07062 | 59af17de0f922c5b649236c098dcd06c2884c379f003904ed6eca6f697df572c | 0.0 |
| 4 | (0.01685, 0.01030) | 0.11257 | -0.09362 | 1b02e5225af82a9aa1e10b0d747db0178e883f270c00b2b2e6eeed039a5c444f | 0.0 |
| 5 | (0.11473, 0.03695) | 0.10145 | 0.07029 | a9504be8f51bf9994ceea1f558c1327d98e220bd270454dc0532a08f7b466a98 | 0.0 |
| 6 | (0.38144, 0.07325) | 0.26916 | -0.13060 | 7b3e52931dd35bdfb3f89b5f8ff1606a2ca136775a9b2d7ac9db06a499ba8972 | 0.0 |
| 7 | (0.47506, 0.08596) | 0.09448 | -0.00029 | 5c277c3fa1c53005e6ef7e900045ca6f3eba1a63fa2ab8a83575de73c3a13f48 | 0.0 |
| 8 | (0.26799, 0.04378) | 0.21133 | 0.06599 | 0214b303a4a19e7acc6e99aee86d995eefc85258ef891efe4a2c83e5c9cd2b2e | 0.0 |
| 9 | (0.36287, 0.05649) | 0.09573 | -0.06781 | 8bd6dcf3e3667dddc23836261e698b8c8e107757c39fbf620a48cbd25543a953 | 0.0 |

## Determinism proof

- Observation checksums (first 10) run #1: `['4da46dacf4203b91b2742773e2cdf41f5f1cfa1e552f91c8526f830390d6db23', '86d3cb28053400360496fbe5d96ee440ece5d1510e16855be576e2fb761f87db', '1a4bf2685cc0c0e74756d96603c398c679cf46cb0ad30fbe3f709f68fdfe5aac', '59af17de0f922c5b649236c098dcd06c2884c379f003904ed6eca6f697df572c', '1b02e5225af82a9aa1e10b0d747db0178e883f270c00b2b2e6eeed039a5c444f', 'a9504be8f51bf9994ceea1f558c1327d98e220bd270454dc0532a08f7b466a98', '7b3e52931dd35bdfb3f89b5f8ff1606a2ca136775a9b2d7ac9db06a499ba8972', '5c277c3fa1c53005e6ef7e900045ca6f3eba1a63fa2ab8a83575de73c3a13f48', '0214b303a4a19e7acc6e99aee86d995eefc85258ef891efe4a2c83e5c9cd2b2e', '8bd6dcf3e3667dddc23836261e698b8c8e107757c39fbf620a48cbd25543a953']`
- Observation checksums (first 10) run #2 (same seed=1337): identical list above.

## Commands run

- `python3 tools/update_determinism_baseline.py --i-know-what-im-doing` (seed=1337, ticks=200) — exit 0
- `pytest -q` — exit 0
- `pytest -q tests/determinism/test_trace_hash.py` — exit 0
- `pytest -q tests/engine/test_engine_observation_integration.py` — exit 0
