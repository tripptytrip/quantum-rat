# AgentM Analysis Proof (Audited)

This document serves as proof for the successful implementation and audit of the analysis pipeline.

## 1. Commands

The following commands were run to generate the analysis data and the audited report:

**1. Generate tournament data with ticks:**
```bash
python3 -m tournaments.runner --seed 1337 --n-agents 4 --ticks 200 --out analysis/data --protocols open_field,morris_water_maze --include-ticks
```

**2. Extract data into Parquet tables (with overwrite):**
```bash
python3 -m analysis.extract analysis/data
```
**Output:**
```
Leaderboard table saved to analysis/processed/leaderboard.parquet
Episodes table saved to analysis/processed/episodes.parquet
Ticks table saved to analysis/processed/ticks.parquet
```

**3. Generate the analysis report (with grouped rates and sanity checks):**
```bash
python3 -m analysis.report
```
**Output:**
```
Report saved to analysis/report/report.md
```

## 2. Generated Report (Audited)

The following report was generated at `analysis/report/report.md`. It includes the requested metadata, grouped rates, and sanity checks.

---

# Analysis Report

## Run Metadata

| Processed Directory   |   Agents |   Protocols |   Episodes |   Ticks |
|:----------------------|---------:|------------:|-----------:|--------:|
| `analysis/processed`  |        4 |           2 |          8 |     828 |


## Kappa Distribution

|     mean |      std |   min |   max |
|---------:|---------:|------:|------:|
| 0.836362 | 0.640579 |     0 |   2.3 |

![Kappa Distribution](kappa_dist.png)

## Avalanche Size Distribution

|   count |    mean |     std |   min |   max |
|--------:|--------:|--------:|------:|------:|
|      45 | 53.8222 | 40.8786 |     1 |   124 |

![Avalanche Distribution](avalanche_dist.png)

## Microsleep and Replay Rates

| protocol          |   microsleep_rate |   replay_rate |
|:------------------|------------------:|--------------:|
| morris_water_maze |             0     |         0     |
| open_field        |             0.625 |         0.625 |


### Gating Sanity Checks

|   % replay_active & ~microsleep_active |   % microsleep_active & ~replay_active |
|---------------------------------------:|---------------------------------------:|
|                                      0 |                                      0 |


## Protocol Score Distributions

| protocol          |   count |     mean |       std |      min |       25% |      50% |       75% |      max |
|:------------------|--------:|---------:|----------:|---------:|----------:|---------:|----------:|---------:|
| morris_water_maze |       4 |   99.289 | 0.0362399 |   99.242 |   99.2765 |   99.292 |   99.3045 |   99.33  |
| open_field        |       4 | -305.555 | 1.21121   | -306.988 | -306.336  | -305.407 | -304.627  | -304.418 |

![Score Distribution](score_dist.png)

---

The sanity check confirms that `replay_active` is only true when `microsleep_active` is true in this dataset, which aligns with the expected behavior of the system. The analysis pipeline is now more robust and provides more detailed and trustworthy insights.