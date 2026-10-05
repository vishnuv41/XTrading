# Phase 17 Holdout Evaluation Pre-Registration

**Timestamp**: 2026-10-05T03:28:09.871640+00:00
**Protocol Status**: FROZEN PRIOR TO HOLDOUT ACCESS

## 1. Chronological Partitioning Definition

- **Common Universe Start**: 2020-08-11T00:00:00+00:00
- **Development Window (First 75%)**: 2020-08-11T00:00:00+00:00 to 2025-03-22T12:00:00+00:00
- **Reserved Holdout Window (Last 25%)**: 2025-03-22T12:00:00+00:00 to 2026-10-05T00:00:00+00:00
- **Timeframe**: 1d

## 2. Advancement Decisions from Development Window

| Strategy | In-Sample Net Sharpe | 95% Date-Cluster CI | Null p-val | 40bps Sharpe | Status | Reason / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy_and_Hold | 1.263 | [0.295, 2.204] | 0.4320 | 1.263 | REJECTED | Exposure null p-value 0.4320 >= 0.05 |
| Trend_EMA_20 | 1.408 | [0.316, 2.447] | 0.0130 | 1.359 | **ADVANCE TO HOLDOUT** | All criteria satisfied. |
| Trend_EMA_50 | 1.377 | [0.324, 2.387] | 0.0350 | 1.350 | **ADVANCE TO HOLDOUT** | All criteria satisfied. |
| Trend_EMA_100 | 1.328 | [0.339, 2.314] | 0.1040 | 1.311 | REJECTED | Exposure null p-value 0.1040 >= 0.05 |
| Trend_EMA_200 | 1.339 | [0.363, 2.313] | 0.1170 | 1.328 | REJECTED | Exposure null p-value 0.1170 >= 0.05 |
| Crossover_EMA_20_100 | 1.343 | [0.382, 2.282] | 0.1470 | 1.339 | REJECTED | Exposure null p-value 0.1470 >= 0.05 |
| Crossover_EMA_50_200 | 1.258 | [0.331, 2.171] | 0.3140 | 1.256 | REJECTED | Exposure null p-value 0.3140 >= 0.05 |
| TSMOM_30d | 1.597 | [0.575, 2.599] | 0.0030 | 1.570 | **ADVANCE TO HOLDOUT** | All criteria satisfied. |
| TSMOM_60d | 1.405 | [0.424, 2.361] | 0.0310 | 1.386 | **ADVANCE TO HOLDOUT** | All criteria satisfied. |
| TSMOM_90d | 1.257 | [0.306, 2.205] | 0.1930 | 1.243 | REJECTED | Exposure null p-value 0.1930 >= 0.05 |
| TSMOM_180d | 0.694 | [-0.317, 1.702] | 0.9790 | 0.683 | REJECTED | 95% CI lower bound -0.317 <= 0.00; Exposure null p-value 0.9790 >= 0.05 |
| TSMOM_365d | 0.557 | [-0.373, 1.512] | 0.9900 | 0.549 | REJECTED | 95% CI lower bound -0.373 <= 0.00; Exposure null p-value 0.9900 >= 0.05 |
| VolTarget_20pct_TSMOM_90d | 1.094 | [0.111, 2.069] | 0.4850 | 1.065 | REJECTED | Exposure null p-value 0.4850 >= 0.05 |
| VolTarget_40pct_TSMOM_90d | 1.122 | [0.137, 2.094] | 0.4260 | 1.093 | REJECTED | Exposure null p-value 0.4260 >= 0.05 |
| VolTarget_20pct_TSMOM_180d | 0.694 | [-0.336, 1.715] | 0.9790 | 0.672 | REJECTED | 95% CI lower bound -0.336 <= 0.00; Exposure null p-value 0.9790 >= 0.05 |
| VolTarget_40pct_TSMOM_180d | 0.720 | [-0.321, 1.736] | 0.9670 | 0.698 | REJECTED | 95% CI lower bound -0.321 <= 0.00; Exposure null p-value 0.9670 >= 0.05 |

## 3. Pre-Registered Holdout Candidates

**Advancing Candidate Count**: 4
**Advancing Set**: ['Trend_EMA_20', 'Trend_EMA_50', 'TSMOM_30d', 'TSMOM_60d']

**Strict Rule**: No parameters will be adjusted post-hoc based on holdout performance.
