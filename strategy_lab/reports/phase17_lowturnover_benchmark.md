# Phase 17 Low-Turnover Benchmark Suite Report

**Date**: 2026-10-05 03:29:24 UTC
**Protocol**: Pre-registered low-turnover systematic trend & momentum benchmark suite.
**Universe**: BTC/USDT, ETH/USDT, BNB/USDT, XRP/USDT, ADA/USDT, LTC/USDT, SOL/USDT, DOGE/USDT, LINK/USDT (9 assets)
**Friction Model**: Canonical 30.0 bps round-trip (15.0 bps per leg)
**Timeline Split**: Development (2020-08-11 to 2025-03-22) | Holdout (2025-03-22 to 2026-10-05)

## 1. Development Window Performance Matrix (Frozen Grid)

| Strategy | Ann. Return (Net) | Volatility | Net Sharpe | 95% Date-Cluster CI | Max DD | Calmar | Annual Turnover | Bull Sharpe | Bear Sharpe | Null p-val | Advance? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy_and_Hold | 102.2% | 80.9% | **1.263** | [0.295, 2.204] | -78.7% | 1.30 | 0.22x | 3.05 | -1.16 | 0.4320 | NO |
| Trend_EMA_20 | 65.0% | 46.2% | **1.408** | [0.316, 2.447] | -48.0% | 1.36 | 44.62x | 2.63 | -1.40 | 0.0130 | **YES** |
| Trend_EMA_50 | 73.3% | 53.3% | **1.377** | [0.324, 2.387] | -56.5% | 1.30 | 28.86x | 2.68 | -2.13 | 0.0350 | **YES** |
| Trend_EMA_100 | 73.7% | 55.5% | **1.328** | [0.339, 2.314] | -61.9% | 1.19 | 18.82x | 2.65 | -2.30 | 0.1040 | NO |
| Trend_EMA_200 | 77.1% | 57.5% | **1.339** | [0.363, 2.313] | -59.3% | 1.30 | 13.43x | 2.81 | -2.16 | 0.1170 | NO |
| Crossover_EMA_20_100 | 79.8% | 59.4% | **1.343** | [0.382, 2.282] | -55.7% | 1.43 | 4.79x | 2.62 | -1.38 | 0.1470 | NO |
| Crossover_EMA_50_200 | 78.2% | 62.1% | **1.258** | [0.331, 2.171] | -55.3% | 1.41 | 2.48x | 2.61 | -1.28 | 0.3140 | NO |
| TSMOM_30d | 86.2% | 54.0% | **1.597** | [0.575, 2.599] | -47.9% | 1.80 | 29.07x | 2.90 | -1.58 | 0.0030 | **YES** |
| TSMOM_60d | 81.7% | 58.2% | **1.405** | [0.424, 2.361] | -63.8% | 1.28 | 22.29x | 2.78 | -1.67 | 0.0310 | **YES** |
| TSMOM_90d | 72.3% | 57.5% | **1.257** | [0.306, 2.205] | -72.3% | 1.00 | 16.10x | 2.60 | -1.82 | 0.1930 | NO |
| TSMOM_180d | 37.8% | 54.5% | **0.694** | [-0.317, 1.702] | -68.2% | 0.55 | 11.82x | 2.09 | -1.53 | 0.9790 | NO |
| TSMOM_365d | 23.2% | 41.7% | **0.557** | [-0.373, 1.512] | -49.4% | 0.47 | 5.97x | 1.56 | -1.52 | 0.9900 | NO |
| VolTarget_20pct_TSMOM_90d | 12.5% | 11.4% | **1.094** | [0.111, 2.069] | -25.2% | 0.50 | 6.60x | 2.33 | -2.18 | 0.4850 | NO |
| VolTarget_40pct_TSMOM_90d | 25.6% | 22.8% | **1.122** | [0.137, 2.094] | -43.4% | 0.59 | 13.06x | 2.35 | -2.17 | 0.4260 | NO |
| VolTarget_20pct_TSMOM_180d | 8.1% | 11.7% | **0.694** | [-0.336, 1.715] | -23.1% | 0.35 | 5.04x | 1.88 | -1.93 | 0.9790 | NO |
| VolTarget_40pct_TSMOM_180d | 16.8% | 23.4% | **0.720** | [-0.321, 1.736] | -41.5% | 0.41 | 9.94x | 1.90 | -1.90 | 0.9670 | NO |

## 2. Cost Sensitivity Matrix (Development Window)

| Strategy | 0 bps | 10 bps | 20 bps | 30 bps (Base) | 40 bps | 50 bps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy_and_Hold | 1.263 | 1.263 | 1.263 | 1.263 | 1.263 | 1.262 |
| Trend_EMA_20 | 1.553 | 1.504 | 1.456 | 1.408 | 1.359 | 1.311 |
| Trend_EMA_50 | 1.458 | 1.431 | 1.404 | 1.377 | 1.350 | 1.323 |
| Trend_EMA_100 | 1.379 | 1.362 | 1.345 | 1.328 | 1.311 | 1.294 |
| Trend_EMA_200 | 1.374 | 1.363 | 1.351 | 1.339 | 1.328 | 1.316 |
| Crossover_EMA_20_100 | 1.355 | 1.351 | 1.347 | 1.343 | 1.339 | 1.335 |
| Crossover_EMA_50_200 | 1.264 | 1.262 | 1.260 | 1.258 | 1.256 | 1.254 |
| TSMOM_30d | 1.678 | 1.651 | 1.624 | 1.597 | 1.570 | 1.543 |
| TSMOM_60d | 1.463 | 1.444 | 1.424 | 1.405 | 1.386 | 1.367 |
| TSMOM_90d | 1.299 | 1.285 | 1.271 | 1.257 | 1.243 | 1.229 |
| TSMOM_180d | 0.726 | 0.716 | 0.705 | 0.694 | 0.683 | 0.672 |
| TSMOM_365d | 0.578 | 0.571 | 0.564 | 0.557 | 0.549 | 0.542 |
| VolTarget_20pct_TSMOM_90d | 1.180 | 1.151 | 1.122 | 1.094 | 1.065 | 1.036 |
| VolTarget_40pct_TSMOM_90d | 1.208 | 1.179 | 1.150 | 1.122 | 1.093 | 1.064 |
| VolTarget_20pct_TSMOM_180d | 0.758 | 0.737 | 0.715 | 0.694 | 0.672 | 0.651 |
| VolTarget_40pct_TSMOM_180d | 0.783 | 0.762 | 0.741 | 0.720 | 0.698 | 0.677 |

## 3. Reserved Holdout Performance Matrix (Single Evaluation)

| Strategy | Ann. Return (Net) | Volatility | Net Sharpe | 95% Date-Cluster CI | Max DD | Calmar | Annual Turnover | Holdout Lift vs B&H |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Trend_EMA_50 | 13.4% | 32.9% | **0.406** | [-1.450, 2.121] | -38.5% | 0.35 | 26.77x | +0.259 |
| Trend_EMA_20 | 3.8% | 31.3% | **0.120** | [-1.727, 1.814] | -40.5% | 0.09 | 46.98x | -0.027 |
| TSMOM_30d | 9.9% | 34.3% | **0.288** | [-1.541, 2.003] | -45.6% | 0.22 | 31.82x | +0.141 |
| TSMOM_60d | 11.6% | 33.4% | **0.348** | [-1.277, 1.997] | -28.4% | 0.41 | 16.67x | +0.201 |
| Buy_and_Hold | 8.7% | 59.3% | **0.147** | [-1.349, 1.686] | -66.9% | 0.13 | 0.65x | 0.000 (Base) |

## 4. Key Findings & Empirical Synthesis

- **Static Exposure vs. Dynamic Timing**: Against a static cash/crypto mix matched to empirical time-in-market (~40–50% cash), dynamic trend timing does *not* consistently reduce maximum drawdown across all regimes:
  - *Development (2020–2025)*: Trend MaxDD $-56.5\%$ vs. Static Exposure $-49.6\%$ (slightly worse).
  - *Holdout (2025–2026)*: Trend MaxDD $-38.5\%$ vs. Static Exposure $-34.9\%$ (comparable).
  - *Out-of-Time Crash (2018–2020)*: Trend MaxDD $-35.5\%$ vs. Static Exposure $-45.9\%$ (improved) and Full B&H $-81.0\%$.
  - *Conclusion*: The drawdown reduction relative to 100% Buy & Hold is predominantly driven by holding ~40–50% cash (de-leveraging). Dynamic timing added return (+40.9% vs +7.4% static in 2018–2020), but does not systematically beat static cash on drawdown alone.
- **Sharpe Outperformance is Unproven**: While point-estimate Net Sharpe improved over Buy & Hold across all three windows (+0.11 dev, +0.26 holdout, +0.78 out-of-time), all paired date-cluster 95% bootstrap CIs span zero.
- **Episode Concentration**: The out-of-time $p < 0.01$ reflects defensive cash positioning during the single prolonged 2018 bear market episode, not a large sample of independent regime tests.
- **Reference Baseline Selection**: `Trend_EMA_50` (Daily) is adopted as the single reference baseline strictly because it represents the midpoint of the empirical parameter plateau (EMA 20 to 100) with moderate turnover (23–28x/yr). No un-registered ensembles are introduced.
- **Data Accounting**: Historical Binance data (2018–2026) is spent for these 16 trend/momentum rules. Any subsequent model or ML candidate must be judged against `Trend_EMA_50` on forward data.
