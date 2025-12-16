# Quantum Rat v2.1 — Build Checklist (Agent-Gated)

This is the single source of truth for build progress.
Rule: no box may be checked unless its DoD is proven (tests/logs/artifacts).

## 0) Legacy preservation + baseline capture (MUST DO BEFORE REFACTOR)
- [ ] Tag legacy state (git tag `legacy-v0.9`) and/or move `app.py` to `legacy/` without edits.
- [ ] Capture legacy baseline trace for drift comparison:
  - [ ] Run legacy for 2000 frames in deterministic mode
  - [ ] Save `tests/fixtures/legacy_baseline.json` containing at minimum: position, score, frustration per tick
  - [ ] Document exact command + seed used in `docs/legacy_baseline.md`

## 1) Repo skeleton matches v2.1 spec (Milestone 0.1)
- [ ] Directory tree exists: `app/ core/ brain/ experiments/ metrics/ tests/ docs/`
- [ ] Imports resolve (no circular deps)
- [ ] Placeholder README updated to point to v2.1 spec + backlog

## 2) Deterministic RNG infrastructure (Milestone 0.2)
- [ ] Single RNG authority implemented (`core/rng.py`)
- [ ] No raw `random` / `np.random` usage outside RNG authority
- [ ] Agent-level seed offset support exists

## 3) TickData + logging (Milestone 0.3)
- [ ] `metrics/schema.py` defines TickData (+ schema_version)
- [ ] `metrics/logger.py` emits JSONL per tick
- [ ] Per-tick hash helper exists (stable ordering, deterministic)

## 4) CI determinism gate (Milestone 0.4)
- [ ] `tests/determinism/` baseline trace hashes committed
- [ ] CI fails on any determinism regression
- [ ] Baseline update requires explicit flag/script (auditable)

## 5) World + sensors + Observation contract (Milestone 1)
- [ ] Deterministic world stepping (`core/world.py`, `core/entities.py`)
- [ ] Sensors produce normalized Observation (`brain/contracts.py`)
- [ ] Observation includes egomotion/proprioception (no position cheating)

## 6) Physiology + neuromodulation (Milestone 2)
- [ ] ATP/glycogen dynamics implemented, logged in TickData
- [ ] DA/5HT/NE/ACh updates deterministic + logged

## 7) Criticality core (Milestone 3 — highest priority)
- [ ] CriticalityField lattice implemented
- [ ] Avalanche detection + κ (EMA) implemented + logged
- [ ] Validation sweep script exists + has assertions
- [ ] Reduced sweep runs in CI

## 8) TRN, microsleep, replay, memory (Milestone 4)
- [ ] TRN gating states logged
- [ ] Microsleep episodes reproducible
- [ ] Replay only occurs during microsleep

## 9) Spatial + action selection (Milestone 5)
- [ ] Grid/place/HD uses egomotion
- [ ] Working memory bounded + logged
- [ ] Basal ganglia action selection deterministic

## 10) Assays + UI parity (Milestone 6)
- [ ] Headless experiment runner works
- [ ] Open Field / T-Maze / Water Maze / Survival implemented as protocols
- [ ] Flask UI reads from Engine history (no direct World access)

## 11) Agent container + tournaments (Milestone 7)
- [ ] AgentDNA + Agent container implemented
- [ ] TournamentManager runs N agents fairly + reproducibly
- [ ] Leaderboard output stable under same seed

## 12) Regression + analysis tooling (Milestone 8)
- [ ] Regression harness compares against legacy baseline
- [ ] Analysis scripts reproduce κ/performance and avalanche distributions

## Notes / Decisions log
- [ ] `docs/decisions.md` exists and records any spec deviations + rationale
