## Call chain snippet
- Pipeline order: `pre_tick -> physiology -> world -> sensors -> trn -> brain -> log` (`core/pipeline.py`).
- `core/engine.py`:
  - `_sensors_step` computes `Observation` and checksum.
  - `_trn_step` updates criticality metrics, microsleep, TRN state, and replay (asserts replay never outside microsleep).
  - `_brain_step` consumes `ctx.observation` (raises if missing) and updates neuromodulators.

## Microsleep window (seed=1337, ticks 6–45)
| tick | atp | glycogen | kappa | trn_state | microsleep_active | microsleep_ticks_remaining | replay_active | replay_index | obs_checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.72 | 2.86 | 0.0 | OPEN | False | 0 | False | -1 | 7b3e52931dd35bdfb3f89b5f8ff1606a2ca136775a9b2d7ac9db06a499ba8972 |
| 7 | 0.68 | 2.84 | 0.0 | OPEN | False | 0 | False | -1 | 5c277c3fa1c53005e6ef7e900045ca6f3eba1a63fa2ab8a83575de73c3a13f48 |
| 8 | 0.64 | 2.82 | 0.0 | OPEN | False | 0 | False | -1 | 0214b303a4a19e7acc6e99aee86d995eefc85258ef891efe4a2c83e5c9cd2b2e |
| 9 | 0.6 | 2.8 | 0.0 | OPEN | False | 0 | False | -1 | 8bd6dcf3e3667dddc23836261e698b8c8e107757c39fbf620a48cbd25543a953 |
| 10 | 0.56 | 2.78 | 0.0 | OPEN | False | 0 | False | -1 | 885c6a3a27000581b1096415c3069dfb039c05e7eba1407b947008f11b490e4b |
| 11 | 0.52 | 2.76 | 0.0 | NARROW | False | 0 | False | -1 | 1c8fc04e47e25a7f13f97c732434bb6807067681193bba19d1715fd55fbf2e05 |
| 12 | 0.48 | 2.74 | 0.0 | NARROW | False | 0 | False | -1 | 10bd511a74abb29c32f25299900f6270e358c38d008ea8b6dff5a4d85aeb7700 |
| 13 | 0.44 | 2.72 | 0.0 | NARROW | False | 0 | False | -1 | 44d18d2e2379080f405cdf88402b9c7d27c856eb08db2cd8efef498a44f08f40 |
| 14 | 0.4 | 2.7 | 0.0 | NARROW | False | 0 | False | -1 | e239b5de4fbcb94ceec9d22fb98f1111f49c9d86b19336aeba075c101322a4ab |
| 15 | 0.36 | 2.68 | 0.0 | NARROW | False | 0 | False | -1 | 396683e247843ea50d92dd87b6bc42fb264a9fe9c797271396c1e8d9d09582f1 |
| 16 | 0.32 | 2.66 | 0.0 | CLOSED | False | 0 | False | -1 | 7255ed514ebc74c93ad83a32f159f96a2c06ef357820bda7a67075a7f420d376 |
| 17 | 0.28 | 2.64 | 0.0 | CLOSED | False | 0 | False | -1 | b4088dfae2f2aaee700ef3e87dbd66935de2e5bad87ce9bba2eb9e084f047174 |
| 18 | 0.24 | 2.62 | 0.0 | CLOSED | False | 0 | False | -1 | cd1dc125e65900eb88f15e31a5a2a5f09f3bbd91192925872976c5b5d31d7c6c |
| 19 | 0.2 | 2.6 | 0.0 | CLOSED | False | 0 | False | -1 | 5f6e627c36c67320ba4f9f23542016df39349525778771f7d8f079cb9c9e9961 |
| 20 | 0.16 | 2.58 | 0.0 | CLOSED | False | 0 | False | -1 | 103a0933212d411cc213b0730f1257589b2aeb01c9f79242eb944efa93b8149f |
| 21 | 0.12 | 2.56 | 0.0 | CLOSED | False | 0 | False | -1 | 25ed6ae78cf1c2742c43bb951c25a9ef79389cdebf9a5be882df832cd8b9306a |
| 22 | 0.08 | 2.54 | 0.0 | CLOSED | False | 0 | False | -1 | 4a45122fb87ce981d09ca4d43ff4f323d8102637a6ac87d12a6c90dd4555bc1b |
| 23 | 0.04 | 2.52 | 0.0 | CLOSED | False | 0 | False | -1 | 29c513ff688e533255ecae7d3afbb2dd9c7c234025f329ef6c4abafca85cc1a1 |
| 24 | 0.02 | 2.48 | 0.0 | CLOSED | False | 0 | False | -1 | ace024ab02edfb8436f6b5ffe8ec21b2a3cbfaef42a832554fa295967569bf37 |
| 25 | 0.04 | 2.4 | 0.0 | CLOSED | False | 0 | False | -1 | e9fbdb75c9ebc43552eada2c89bb1b0c80e188a1ca2eaf4960da1c303b36f098 |
| 26 | 0.02 | 2.36 | 0.0 | CLOSED | True | 25 | True | 0 | 24d317a27757ba1c69f57a24210aed0738685a3d40ecfcd62b2937ea2f8179f1 |
| 27 | 0.04 | 2.28 | 0.0 | CLOSED | True | 24 | True | 1 | 1c2c0f29bf6f89a2295d7bf075f092f5bf37c7f3efb27e278654a448889e6ac0 |
| 28 | 0.02 | 2.24 | 0.0 | CLOSED | True | 23 | True | 2 | efd5728c72d111ed35f7b7721c19cca1f2bf8dd14959a9e8b0d1ca025396a607 |
| 29 | 0.04 | 2.16 | 0.0 | CLOSED | True | 22 | True | 3 | 3d0e5f106cdbd874c01899c3aea01a7fc08f3d6827625f6c2da13115f119690e |
| 30 | 0.02 | 2.12 | 0.0 | CLOSED | True | 21 | True | 4 | 9d55fc9297f5fc87090ff307bfb6da4f72afcd47a3c013837cadc4ff2b21859b |
| 31 | 0.04 | 2.04 | 0.0 | CLOSED | True | 20 | True | 5 | 6cdbde081c35ece24162de73c540431a3f572c9dfa3f0a64026da4f05f66e6f9 |
| 32 | 0.02 | 2.0 | 0.0 | CLOSED | True | 19 | True | 6 | affa509fdff096bbedcad16ed765a0e7df97709cd1d06dc3f892037a69398552 |
| 33 | 0.04 | 1.92 | 0.0 | CLOSED | True | 18 | True | 7 | be7abf3a8773d34f44de2a411176e137967f4095ed15cd5cf6c9a09e86b52484 |
| 34 | 0.02 | 1.88 | 0.0 | CLOSED | True | 17 | True | 8 | 2f610fae7162811364baa8caf4bf3cea86be2cce7879490a80a243c90faf9b7c |
| 35 | 0.04 | 1.8 | 0.0 | CLOSED | True | 16 | True | 9 | 4aea0855f84d64399afb832af0148512ab1c86b4a57b97440afe400b37588bd8 |
| 36 | 0.02 | 1.76 | 0.0 | CLOSED | True | 15 | True | 10 | 8b7b9122709ff71d97781c8c227030b9b223dd888fb0bffff2d92399358d9923 |
| 37 | 0.04 | 1.68 | 0.0 | CLOSED | True | 14 | True | 11 | 2427426ee4215e4d0049cee972d7023d67598c111180a8230c0bb01de28fd483 |
| 38 | 0.02 | 1.64 | 0.0 | CLOSED | True | 13 | True | 12 | 93b9b119a6962cb6feb6bf2e974472442bad2696710085557ed2518b4a123312 |
| 39 | 0.04 | 1.56 | 0.0 | CLOSED | True | 12 | True | 13 | 6bdd1cc6f49a4436223367a28825fa158031d182af19b194279732ed29094acc |
| 40 | 0.02 | 1.52 | 0.0 | CLOSED | True | 11 | True | 14 | bd9eafe81148efa5ebcd70afa9a8ddb2ada3312545673b49fb0c73c4570288ee |
| 41 | 0.04 | 1.44 | 0.0 | CLOSED | True | 10 | True | 15 | 22ad2eb02f46bb6ac1a302deda4acb5cd951836286a5b9f9c2a9eb6b076cdc68 |
| 42 | 0.02 | 1.4 | 0.0 | CLOSED | True | 9 | True | 16 | 981242a1746a62fb1c9919bf1a5c0bbc2d23532fb023472459d92b597330d4bf |
| 43 | 0.04 | 1.32 | 0.0 | CLOSED | True | 8 | True | 17 | b37232ac212972c4f8cba8491add750d47c033d030d8527bd6d15f37725d9add |
| 44 | 0.02 | 1.28 | 0.0 | CLOSED | True | 7 | True | 18 | a682b8285ba0f61e606fbdd9548eb0ba9fe31812867acee7ffdd656cb8642479 |
| 45 | 0.04 | 1.2 | 0.0 | CLOSED | True | 6 | True | 19 | ad8a6d9a514705f78e0c473067bfd7cb3baefd4f4037073eef05d0908764fd64 |

## Hard gating proof
- Count of ticks where `replay_active == True` and `microsleep_active == False`: **0**

## Determinism proof (first 20 ticks, seed=1337)
- Run #1: `[(0, 'OPEN', False, False, -1), (1, 'OPEN', False, False, -1), (2, 'OPEN', False, False, -1), (3, 'OPEN', False, False, -1), (4, 'OPEN', False, False, -1), (5, 'OPEN', False, False, -1), (6, 'OPEN', False, False, -1), (7, 'OPEN', False, False, -1), (8, 'OPEN', False, False, -1), (9, 'OPEN', False, False, -1), (10, 'OPEN', False, False, -1), (11, 'NARROW', False, False, -1), (12, 'NARROW', False, False, -1), (13, 'NARROW', False, False, -1), (14, 'NARROW', False, False, -1), (15, 'NARROW', False, False, -1), (16, 'CLOSED', False, False, -1), (17, 'CLOSED', False, False, -1), (18, 'CLOSED', False, False, -1), (19, 'CLOSED', False, False, -1)]`
- Run #2 (same seed) matches exactly.

## Commands run
- `python3 tools/update_determinism_baseline.py --i-know-what-im-doing` — exit 0
- `pytest -q` — exit 0
- `pytest -q tests/determinism/test_trace_hash.py` — exit 0
- `pytest -q tests/engine/test_milestone4_trn_microsleep_replay.py` — exit 0
