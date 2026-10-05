# Phase 18 Alpha Research: Hypothesis H18-B (Relative Strength Momentum)

**Status**: ❌ **ALL 8 PARAMETER CONFIGURATIONS FAILED TO PASS GATES**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Friction**: Canonical 30.0 bps round-trip transaction friction  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples)

---

## 1. Executive Summary

Hypothesis H18-B tested whether selecting and allocating capital into the top $K \in \{2, 3\}$ momentum assets over lookback $L \in \{14\text{d}, 30\text{d}\}$ at rebalance interval $R \in \{1\text{d}, 7\text{d}\}$ provides incremental risk-adjusted alpha over the reference 9-asset `Trend_EMA_50` baseline.

### Empirical Verdict
- **Gate 1 (Incremental Net Return)**: 6 of 8 configurations increased nominal annualized return (e.g., +165.27% vs +93.41% for Top 3, 14d lookback, 7d rebalance).
- **Gate 2 (Incremental Sharpe $\ge +0.20$)**: ❌ **FAILED**. Best lift was $\Delta\text{Sharpe} = +0.155$ (below the $+0.20$ hurdle).
- **Gate 3 (Paired Bootstrap 95% CI Lower Bound $> 0$)**: ❌ **FAILED**. All 8 configurations had 95% confidence intervals spanning zero (best was $[-0.352, +0.661]$, $p=0.2764$).
- **Gate 4 (Friction Robustness @ 45 bps)**: ✅ **PASSED**. Net Sharpe remained positive across all configurations.
- **Gate 5 (Turnover Drag $\le 50\%$ of Excess Alpha)**: Weekly rebalancing ($R=7\text{d}$) cut turnover drag to ~4.7%, passing Gate 5, whereas daily rebalancing ($R=1\text{d}$) suffered ~13–16% friction drag.
- **Gate 6 (Regime Stability & Drawdown $\le 50\%$)**: ❌ **FAILED**. Concentrating into 2 or 3 altcoins amplified maximum drawdown to $-73.9\%$ to $-88.9\%$ (compared to $-56.5\%$ for the 9-asset diversified baseline).

---

## 2. Parameter Grid Results Table

| Candidate Configuration | Net Annualized Return | Baseline Return | Net Sharpe | Baseline Sharpe | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | One-Tailed $p$-value | Annual Turnover Drag | Max Drawdown | All Gates Passed? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Top2_L14d_R1d` | +97.81% | +93.41% | 1.191 | 1.484 | -0.293 | `[-0.888, +0.265]` | 0.8451 | 16.65% | -88.92% | ❌ **FAIL** |
| `Top2_L14d_R7d` | +83.41% | +93.41% | 1.136 | 1.484 | -0.348 | `[-0.955, +0.268]` | 0.8691 | 5.77% | -82.33% | ❌ **FAIL** |
| `Top2_L30d_R1d` | +141.78% | +93.41% | 1.420 | 1.484 | -0.064 | `[-0.647, +0.493]` | 0.5981 | 12.45% | -81.43% | ❌ **FAIL** |
| `Top2_L30d_R7d` | +86.83% | +93.41% | 1.156 | 1.484 | -0.328 | `[-0.948, +0.287]` | 0.8551 | 4.82% | -83.05% | ❌ **FAIL** |
| `Top3_L14d_R1d` | +159.33% | +93.41% | 1.637 | 1.484 | +0.153 | `[-0.301, +0.583]` | 0.2581 | 13.58% | -80.76% | ❌ **FAIL** |
| `Top3_L14d_R7d` | +165.27% | +93.41% | 1.639 | 1.484 | +0.155 | `[-0.352, +0.661]` | 0.2764 | 4.68% | -73.90% | ❌ **FAIL** |
| `Top3_L30d_R1d` | +129.24% | +93.41% | 1.478 | 1.484 | -0.006 | `[-0.477, +0.447]` | 0.5234 | 10.54% | -77.04% | ❌ **FAIL** |
| `Top3_L30d_R7d` | +96.00% | +93.41% | 1.262 | 1.484 | -0.222 | `[-0.776, +0.337]` | 0.7883 | 3.86% | -77.04% | ❌ **FAIL** |

---

## 3. Core Insights & Why Gate 6 / Gate 3 Failed

1. **Concentration Penalty**: Restricting exposure to 2 or 3 coins reduces portfolio diversification. While top momentum coins rally harder in bull phases, they crash severely during broad crypto drawdowns, worsening the max drawdown from $-56.5\%$ (in the 9-asset baseline) to $-73.9\%$ – $-88.9\%$.
2. **Turnover & Rebalance Frequency**: Daily rebalancing ($R=1\text{d}$) consumes ~10.5%–16.6% in annual friction alone. Weekly rebalancing ($R=7\text{d}$) cuts friction to ~3.8%–5.8%, but concentration risk remains excessive.
3. **Statistical Insignificance**: With paired bootstrap $p$-values ranging between $0.258$ and $0.869$, we cannot reject the null hypothesis that relative strength momentum gains are indistinguishable from sampling noise and beta leverage.

---

## 4. Archival Decision

Per pre-registration rules, **Hypothesis H18-B is formally rejected and archived**. It will NOT advance to Holdout evaluation or live execution.
