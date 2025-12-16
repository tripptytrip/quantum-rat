# Quantum Rat v2.1 – Milestone Backlog

This backlog is **directly derived from the frozen v2.1 Engineering Specification**.  
Each milestone corresponds to a coherent, testable slice of the system.  
No milestone introduces speculative features beyond the spec.

---

## Milestone 0 — Repository & Determinism Scaffolding

**Goal:** Establish a deterministic, modular skeleton that can run a no-op simulation loop and produce valid TickData.

> **This milestone is load-bearing.** No downstream work is valid unless determinism gates pass.

### Epic 0.1 — Repository Structure

**Stories**
- Create top-level folders: `app/`, `core/`, `brain/`, `assays/`, `experiments/`, `docs/`
- Add empty module files matching the spec
- Add placeholder README pointing to frozen spec

**Definition of Done (DoD)**
- Repo structure matches spec
- Imports resolve without circular dependencies

---

### Epic 0.2 — Deterministic RNG Infrastructure

**Stories**
- Implement single RNG factory (seeded, injectable)
- Ensure all random calls route through RNG
- Add deterministic seed offset support (Agent-level)

**DoD**
- Two identical runs with same seed produce identical TickData hashes
- RNG usage audited (no `random` / `np.random` leakage)

---

### Epic 0.3 — TickData Skeleton

**Stories**
- Define TickData dataclass with schema_version
- Emit TickData every tick
- JSONL serialization + per-tick hash helper

**DoD**
- TickData emitted with correct schema
- Determinism check passes on empty brain

---

### Epic 0.4 — CI Determinism Gate

**Stories**
- Create fixed-seed determinism fixture (N ticks)
- Persist baseline trace hashes under `tests/determinism/`
- Add CI job that compares traces on every PR
- Require explicit override flag to update baseline

**DoD**
- CI fails on any determinism regression
- Baseline update is explicit and auditable

---

### Epic 0.2 — Deterministic RNG Infrastructure

**Stories**
- Implement single RNG factory (seeded, injectable)
- Ensure all random calls route through RNG
- Add deterministic seed offset support

**DoD**
- Two identical runs with same seed produce identical TickData hashes
- RNG usage audited (no `random` / `np.random` leakage)

---

### Epic 0.3 — TickData Skeleton

**Stories**
- Define TickData dataclass with schema_version
- Emit TickData every tick
- JSON serialization + checksum helper

**DoD**
- TickData emitted with correct schema
- Determinism check passes on empty brain

---

## Milestone 1 — World, Sensors, and Observation Contract

**Goal:** Produce a correct, normalized Observation object from a dynamic world.

### Epic 1.1 — World Grid & Entities

**Stories**
- Implement grid world with walls, food, predator
- Collision detection
- Deterministic world stepping

**DoD**
- World evolves deterministically
- Entity positions reproducible per seed

---

### Epic 1.2 — Sensors

**Stories**
- Vision ray casting
- Whisker contact sensors
- Predator proximity signal

**DoD**
- Sensor outputs normalized (0–1)
- Values invariant under replay

---

### Epic 1.3 — Observation Object

**Stories**
- Construct immutable Observation
- Add egomotion (forward/turn deltas)
- Enforce normalization rules

**DoD**
- Observation passes validation checks
- Brain cannot mutate Observation

---

## Milestone 2 — Physiology & Neuromodulation

**Goal:** Establish energy constraints and neuromodulatory control signals.

### Epic 2.1 — Astrocyte Metabolism

**Stories**
- Implement ATP & glycogen dynamics
- Cognitive effort costs
- Energy depletion effects

**DoD**
- Low ATP degrades performance
- Energy curves logged in TickData

---

### Epic 2.2 — Neuromodulator Systems

**Stories**
- Dopamine (reward, plasticity)
- Norepinephrine (arousal)
- Acetylcholine (precision)
- Serotonin (stability)

**DoD**
- Neuromodulators update deterministically
- Levels visible in TickData

---

## Milestone 3 — Criticality Core (Highest Priority)

**Goal:** Replace quantum metaphors with measurable criticality dynamics.

### Epic 3.1 — CriticalityField Component

**Stories**
- Implement lattice activity field
- E/I balance as control parameter
- Activity propagation + decay

**DoD**
- Field runs deterministically
- Activity responds to E/I tuning

---

### Epic 3.2 — Avalanche Detection & Metrics

**Stories**
- Detect avalanche start/end (per spec definition)
- Track size and duration
- Compute κ (EMA-smoothed, threshold-conditioned)

**DoD**
- Subcritical → avalanches die out
- Near-critical → broad size distribution
- Supercritical → runaway activity

---

### Epic 3.2a — Criticality Validation Sweep

**Stories**
- Implement fast E/I sweep script
- Log κ, avalanche rate, mean size
- Assert monotonic trends across regimes

**DoD**
- Validation script fails on invariant violation
- CI runs reduced sweep variant

---

### Epic 3.3 — Logging & Validation

**Stories**
- Log κ, distance, avalanche stats
- Add validation utilities

**DoD**
- Criticality invariants validated via script

---

## Milestone 4 — TRN, Microsleep, and Memory

**Goal:** Introduce gating, offline learning, and internal state transitions.

### Epic 4.1 — TRN Gating

**Stories**
- Sensory vs memory gating
- Burst vs tonic modes

**DoD**
- Gating state visible in TickData

---

### Epic 4.2 — Microsleep

**Stories**
- Implement trigger conditions
- Disable motor/sensory input
- Enable replay

**DoD**
- Microsleep episodes reproducible
- Replay only occurs during microsleep

---

### Epic 4.3 — Replay & Consolidation

**Stories**
- Record recent experience
- Replay sequences
- Update downstream weights

**DoD**
- Replay improves later performance

---

## Milestone 5 — Spatial Cognition & Action Selection

**Goal:** Enable navigation, memory-based decisions, and behavior.

### Epic 5.1 — Spatial System

**Stories**
- Head direction cells
- Grid cell path integration
- Place cell learning

**DoD**
- Position can be reconstructed from egomotion

---

### Epic 5.2 — Working Memory

**Stories**
- Limited slots
- Decay and overwrite rules

**DoD**
- WM contents logged and bounded

---

### Epic 5.3 — Basal Ganglia Action Selection

**Stories**
- Action value computation
- Exploration/exploitation threshold

**DoD**
- Action output respects clamps
- Deterministic given state

---

## Milestone 6 — Behavioral Assays & UI

**Goal:** Validate behavior against mammalian benchmarks.

### Epic 6.1 — Lab Runner

**Stories**
- Trial scheduler
- Reset rules
- Results aggregation

**DoD**
- Assays run without manual intervention

---

### Epic 6.2 — Assays

**Stories**
- Open Field
- T-Maze
- Morris Water Maze
- Survival Arena

**DoD**
- Metrics comparable across runs

---

## Milestone 7 — Agents, DNA, and Tournaments

**Goal:** Enable ablation, evolution, and comparative evaluation.

### Epic 7.1 — Agent & AgentDNA

**Stories**
- Define Agent wrapper
- Define serializable AgentDNA

**DoD**
- Agent fully defined by DNA + seed

---

### Epic 7.2 — TournamentManager

**Stories**
- Instantiate agents
- Run identical assays
- Rank and score

**DoD**
- Fair, reproducible comparisons

---

### Epic 7.3 — Evolution Loop (Optional)

**Stories**
- Selection
- Mutation
- Lineage persistence

**DoD**
- Evolution improves metrics over generations

---

## Milestone 8 — Regression & Analysis Tooling

**Goal:** Lock correctness and support research claims.

### Epic 8.1 — Regression Harness

**Stories**
- Fixed-seed baseline traces (JSONL)
- Include extended fields:
  - action outputs
  - energy (ATP/glycogen)
  - neuromodulators
  - criticality metrics
- Checksum comparison per tick

**DoD**
- Any unintended behavioral or internal drift detected

---

### Epic 8.2 — Analysis Scripts

**Stories**
- κ vs performance plots
- Avalanche distributions

**DoD**
- Plots reproducible from logs

---

_End of backlog._

