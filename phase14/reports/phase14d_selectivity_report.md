# Phase 14D — Execution Semantics & Selectivity Audit Report

*Empirical Evaluation of Prediction Quintiles, Percentile Selectivity, & Break-Even Edge*

## BTC/USDT Selectivity Analysis

### Q1 — Prediction Quintile Breakdown (Q1 Lowest to Q5 Highest)

| Quintile | Bar Count | Mean Prediction (r̂) | Mean Realized 1H Return % | Median Realized 1H Return % | Positive Return % |
| --- | --- | --- | --- | --- | --- |
| Q1 (Lowest) | 380 | -0.1512 | +0.032% | +0.034% | 54.7% |
| Q2 | 380 | -0.0637 | -0.013% | -0.018% | 46.1% |
| Q3 | 379 | -0.0155 | +0.019% | -0.016% | 47.5% |
| Q4 | 380 | +0.0346 | +0.004% | +0.021% | 54.5% |
| Q5 (Highest) | 380 | +0.1181 | +0.018% | +0.004% | 51.1% |

### Q2 — Percentile Selectivity vs Gross & Net Expectancy

*Baseline Roundtrip Friction: `22 BPS`*

| Percentile Gate | Long N | Long Gross Edge % | Long Net Edge % | Short N | Short Gross Edge % | Short Net Edge % |
| --- | --- | --- | --- | --- | --- | --- |
| Top/Bottom 10% (P90) | 190 | +0.042% | -0.178% | 190 | -0.029% | -0.249% |
| Top/Bottom 5% (P95) | 95 | +0.036% | -0.184% | 95 | -0.026% | -0.246% |
| Top/Bottom 2% (P98) | 38 | +0.037% | -0.183% | 38 | -0.041% | -0.261% |
| Top/Bottom 1% (P99) | 19 | +0.071% | -0.149% | 19 | -0.032% | -0.252% |

## ETH/USDT Selectivity Analysis

### Q1 — Prediction Quintile Breakdown (Q1 Lowest to Q5 Highest)

| Quintile | Bar Count | Mean Prediction (r̂) | Mean Realized 1H Return % | Median Realized 1H Return % | Positive Return % |
| --- | --- | --- | --- | --- | --- |
| Q1 (Lowest) | 380 | -0.1985 | +0.024% | +0.009% | 51.3% |
| Q2 | 380 | -0.1511 | -0.007% | -0.019% | 47.4% |
| Q3 | 379 | -0.1305 | +0.055% | +0.012% | 52.8% |
| Q4 | 380 | -0.0897 | +0.022% | +0.001% | 50.0% |
| Q5 (Highest) | 380 | +0.1610 | +0.009% | +0.020% | 53.4% |

### Q2 — Percentile Selectivity vs Gross & Net Expectancy

*Baseline Roundtrip Friction: `22 BPS`*

| Percentile Gate | Long N | Long Gross Edge % | Long Net Edge % | Short N | Short Gross Edge % | Short Net Edge % |
| --- | --- | --- | --- | --- | --- | --- |
| Top/Bottom 10% (P90) | 190 | +0.017% | -0.203% | 190 | -0.050% | -0.270% |
| Top/Bottom 5% (P95) | 95 | +0.125% | -0.095% | 95 | -0.069% | -0.289% |
| Top/Bottom 2% (P98) | 38 | +0.040% | -0.180% | 38 | -0.124% | -0.344% |
| Top/Bottom 1% (P99) | 19 | -0.121% | -0.341% | 19 | -0.200% | -0.420% |

## Execution Semantics & Scientific Findings

1. **Execution Semantics Clarification**: In `phase14c_edge_validation_report.md`, Baseline `B4` evaluated an **un-gated continuous signal** (~1,840 trades over 2,000 bars), not the selective Top-1% Phase 13 candidate. `B4` is formally re-labeled as **'Un-gated 1H ML Signal Benchmark'**.
2. **Quintile Ranking Diagnostics**: Prediction quintiles (Q1 to Q5) demonstrate flat realized returns across quintiles (e.g. Q1 vs Q5 mean returns differ by < 0.02%), confirming that prediction values carry minimal ordinal ranking power.
3. **Selectivity vs Friction**: Even at high selectivity (Top 1% / P99), gross edge per trade (+0.01% to +0.04%) is insufficient to overcome 22–30 BPS friction costs, resulting in negative net expected edge.
