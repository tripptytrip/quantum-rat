## B1 — Determinism baseline regenerated after Criticality integration
**Date:** 2025-12-16  
**Why:** `kappa` and `avalanche_size` switched from placeholder values to real `CriticalityField` outputs, which are included in the determinism hash.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` updated; determinism gate remains enforced.

## B2 — Determinism baseline regenerated after Observation wiring
**Date:** 2025-12-16  
**Why:** TickData now logs observation-derived fields and checksums; hashing reflects real Observation output each tick.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` and `tests/determinism/baseline_meta.json` updated; determinism gate remains enforced.

## D1 — Baseline regenerated for TRN/microsleep/replay logging (Milestone 4)
**Date:** 2025-12-16  
**Why:** TickData schema v2.1.2 logs TRN state, microsleep, and replay signals; determinism hash now includes these fields.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` and `tests/determinism/baseline_meta.json` updated; determinism gate remains enforced.

## E1 — Baseline regenerated for Spatial logging (Milestone 5A)
**Date:** 2025-12-16  
**Why:** TickData schema v2.1.3 logs spatial outputs (hd_angle, grid_x, grid_y, place_id) derived from egomotion; hashes include these fields.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` and `tests/determinism/baseline_meta.json` updated; determinism gate remains enforced.

## F1 — Baseline regenerated for Working Memory + Action logging (Milestone 5B)
**Date:** 2025-12-16  
**Why:** TickData schema v2.1.4 logs working memory load/novelty and action outputs; world movement now driven by deterministic actions.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` and `tests/determinism/baseline_meta.json` updated; determinism gate remains enforced.

## F2 — Baseline regenerated after egomotion sign fix
**Date:** 2025-12-16  
**Why:** Egomotion forward_delta now preserves sign (projection onto heading); determinism hash includes observation-derived fields.  
**Command:** `python3 tools/update_determinism_baseline.py --i-know-what-im-doing`  
**Params:** seed=1337, ticks=200  
**Impact:** `tests/determinism/baseline_hashes.json` and `tests/determinism/baseline_meta.json` updated; determinism gate remains enforced.
