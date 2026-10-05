# Phase 19 Alpha Research: Causal Market Regime Classifier Report

**Status**: ❌ **ALL 3 CANDIDATE RULES FAILED ACCEPTANCE GATES**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Canonical Friction**: 30.0 bps round-trip transaction friction (swept 0 to 50 bps)  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples per candidate)

---

## 1. Executive Summary Table

| Candidate Rule | Net Annual Return | Baseline Return | Net Sharpe (30 bps) | Baseline Sharpe | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | One-Tailed $p$-val | Friction Sweep (0 / 30 / 50 bps) | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Regime_Breadth_Filter` (Breadth $\ge 0.30$) | +96.58% | +93.41% | 1.519 | 1.484 | +0.035 | `[-0.062, +0.136]` | 0.2293 | 1.598 / 1.519 / 1.467 | -55.33% | ❌ **FAIL** (CI/G2) |
| `Regime_Strong_Breadth_Filter` (Breadth $\ge 0.60$) | +94.71% | +93.41% | 1.534 | 1.484 | +0.050 | `[-0.211, +0.333]` | 0.3593 | 1.618 / 1.534 / 1.478 | **-41.32%** | ❌ **FAIL** (CI/G2) |
| `Regime_Composite_Filter` (Trend+Breadth+Vol) | +37.64% | +93.41% | 0.996 | 1.484 | -0.489 | `[-0.991, -0.034]` | 0.9828 | 1.092 / 0.996 / 0.931 | -46.28% | ❌ **FAIL** (Severe Lag) |

---

## 2. Core Quantitative Discoveries

1. **Market-Wide Breadth Provides Meaningful Drawdown Dampening**:
   Filtering entries to when $\ge 60\%$ of universe assets are in confirmed trend (`Regime_Strong_Breadth_Filter`) reduced maximum drawdown from **$-56.47\%$** down to **$-41.32\%$** without sacrificing nominal return (+94.71% vs +93.41%).
2. **Statistical Significance Barrier**:
   Despite the point-estimate Sharpe improvement (+1.534 vs 1.484) and cost robustness across 0 to 50 bps (1.478 at 50 bps), the paired bootstrap confidence interval spans zero (`[-0.211, +0.333]`, $p=0.3593$). The lift cannot be distinguished from random sampling noise at the pre-registered $p < 0.05$ threshold.
3. **Over-Filtering Penalty (`Regime_Composite_Filter`)**:
   Adding simultaneous multi-indicator restrictions (EMA20 crossover + Breadth + ATR percentile) resulted in severe over-filtering, reducing Net Sharpe by $-0.489$ with a strictly negative confidence interval (`[-0.991, -0.034]`, $p=0.9828$).

---

## 3. Archival Decision

Per pre-registration rules:
- All 3 Phase 19 candidate rules are **formally archived**.
- None will be promoted to Holdout evaluation or live trading.
- **Phase 17 Forward Baseline remains the sole active benchmark** accumulating prospective evidence.
