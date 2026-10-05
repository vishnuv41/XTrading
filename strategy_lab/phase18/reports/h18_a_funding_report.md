# Phase 18 Alpha Research: Hypothesis H18-A (Continuous Funding Rate Overlay)

**Status**: ❌ **ALL 8 PARAMETER CONFIGURATIONS FAILED TO PASS GATES**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Friction**: Canonical 30.0 bps round-trip transaction friction  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples)

---

## 1. Executive Summary

Hypothesis H18-A tested whether continuous 8h perpetual funding-rate information (rolling z-score over lookback $L \in \{14\text{d}, 30\text{d}\}$ with scaling coefficient $W \in \{0.25, 0.50\}$ under contrarian vs momentum modes) provides incremental predictive or risk-adjusted value beyond `Trend_EMA_50` after 30 bps friction.

### Empirical Verdict
- **Contrarian Overlay Mode**: ❌ **STATISTICALLY SIGNIFICANT DEGRADATION**. Dampening trend exposure during high funding periods reduced Net Sharpe by $-0.134$ to $-0.281$ with paired bootstrap 95% CIs strictly below zero (e.g. `[-0.260, -0.015]`, $p=0.9853$).
- **Momentum Overlay Mode**: ❌ **FAILED GATES**. Boosting exposure on high funding yielded a negligible point-estimate Sharpe lift (+0.007 to +0.045), but net return was lower than baseline due to ~6.0%–8.4% annual turnover drag. All paired bootstrap CIs spanned zero ($p \ge 0.316$).
- **Gate 1 ($\Delta\text{Net Return} > 0$)**: ❌ **FAILED across all 8 variants**.
- **Gate 2 ($\Delta\text{Sharpe} \ge +0.20$)**: ❌ **FAILED across all 8 variants**.
- **Gate 3 (Paired Bootstrap CI Lower Bound $> 0$)**: ❌ **FAILED across all 8 variants**.

---

## 2. Parameter Grid Results Table

| Candidate Configuration | Net Annual Return | Baseline Return | Net Sharpe | Base Sharpe | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | One-Tailed $p$-value | Ann. Turnover Drag | Max Drawdown | All Gates Passed? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Contrarian_L14d_W25%` | +80.20% | +93.41% | 1.344 | 1.484 | -0.140 | `[-0.260, -0.015]` | 0.9853 | 6.51% | -56.26% | ❌ **FAIL** |
| `Contrarian_L14d_W50%` | +67.41% | +93.41% | 1.203 | 1.484 | -0.281 | `[-0.516, -0.038]` | 0.9894 | 8.65% | -57.74% | ❌ **FAIL** |
| `Contrarian_L30d_W25%` | +80.33% | +93.41% | 1.350 | 1.484 | -0.134 | `[-0.254, -0.006]` | 0.9798 | 6.18% | -56.70% | ❌ **FAIL** |
| `Contrarian_L30d_W50%` | +68.03% | +93.41% | 1.204 | 1.484 | -0.280 | `[-0.519, -0.028]` | 0.9844 | 8.01% | -59.68% | ❌ **FAIL** |
| `Momentum_L14d_W25%` | +89.02% | +93.41% | 1.493 | 1.484 | +0.009 | `[-0.099, +0.109]` | 0.4442 | 6.44% | -59.10% | ❌ **FAIL** |
| `Momentum_L14d_W50%` | +83.98% | +93.41% | 1.491 | 1.484 | +0.007 | `[-0.205, +0.205]` | 0.4867 | 8.36% | -61.20% | ❌ **FAIL** |
| `Momentum_L30d_W25%` | +90.49% | +93.41% | 1.503 | 1.484 | +0.019 | `[-0.076, +0.108]` | 0.3499 | 6.07% | -57.90% | ❌ **FAIL** |
| `Momentum_L30d_W50%` | +88.61% | +93.41% | 1.529 | 1.484 | +0.045 | `[-0.140, +0.220]` | 0.3166 | 7.67% | -58.91% | ❌ **FAIL** |

---

## 3. Core Insights & Why Funding Overlay Failed

1. **The Contrarian Fallacy**:
   Assuming high funding rates indicate that "retail is wrong and due for liquidation" is empirically refuted. In major crypto bull trends, positive funding rates persist for months while prices continue to double or triple. De-leveraging during high funding systematically clipped the best upside bars.
2. **Turnover Friction Neutralizes Momentum Tilt**:
   Continuously adjusting position sizes proportional to daily funding z-scores introduced 6%–8% in annual turnover drag, which fully absorbed the slight gross gain.
3. **Statistical Indistinguishability**:
   With paired bootstrap $p$-values $\ge 0.316$ for momentum mode and statistically negative CIs for contrarian mode, funding overlay provides no statistically distinguishable edge over static equal weighting.

---

## 4. Archival Decision

Per pre-registration rules, **Hypothesis H18-A is formally rejected and archived**. It will NOT advance to Holdout evaluation or live execution.
