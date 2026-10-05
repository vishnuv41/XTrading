# Phase 18 Alpha Research: Final Synthesis Report

**Status**: 🔒 **ALL 3 HYPOTHESES EVALUATED & REJECTED UNDER PRE-REGISTERED PROTOCOL**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Canonical Baseline**: `Trend_EMA_50` Equal-Weighted (Net Sharpe: 1.484, Net Return: +93.41%, Max DD: -56.5%)  
**Canonical Friction**: 30.0 bps Round-Trip Transaction Friction  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples per cell)

---

## 1. Executive Summary Table

| Hypothesis ID | Mechanism Tested | Best Candidate | Net Return vs Base | Net Sharpe vs Base | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | Turnover Drag | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H18-B** | Relative Strength (Top-K Momentum) | `Top3_L14d_R7d` | **+165.27%** vs +93.41% | **1.639** vs 1.484 | +0.155 | `[-0.352, +0.661]` ($p=0.276$) | 4.68% | **-73.90%** vs -56.5% | ❌ **REJECTED** (Drawdown/CI) |
| **H18-C** | Pre-Entry Volatility Compression | `ATR_Ratio_L30d` | **+31.49%** vs +93.41% | **1.167** vs 1.484 | -0.317 | `[-0.971, +0.519]` ($p=0.751$) | 0.76% | **-23.09%** vs -56.5% | ❌ **REJECTED** (Opportunity Lag) |
| **H18-A** | Continuous Funding Rate Overlay | `Momentum_L30d_W50%` | **+88.61%** vs +93.41% | **1.529** vs 1.484 | +0.045 | `[-0.140, +0.220]` ($p=0.317$) | 7.67% | **-58.91%** vs -56.5% | ❌ **REJECTED** (Friction Drag/CI) |

---

## 2. Core Quantitative Discoveries

1. **Momentum Concentration vs Diversification (H18-B)**:
   Concentrating into top 2 or 3 momentum coins amplified headline nominal return during bull runs, but increased maximum drawdown from $-56.5\%$ to $-73.9\%$ to $-88.9\%$. The paired bootstrap confidence interval spans zero ($[-0.352, +0.661]$), failing to prove statistically distinguishable alpha.
2. **Volatility Compression Pitfall (H18-C)**:
   Requiring volatility compression prior to entry reduced drawdown ($-23.1\%$), but at the cost of missing the majority of major crypto trend expansions, collapsing annualized return by $-62\%$ to $-89\%$.
3. **Funding Rate Market Dynamics (H18-A)**:
   - **Contrarian mode** (cutting exposure when funding is high) was decisively refuted ($p > 0.98$), as persistent positive funding is characteristic of strong bull runs.
   - **Momentum mode** yielded minor gross lift, but dynamic rebalancing friction (6–8% per year) neutralized the edge.

---

## 3. Governance State

- **Phase 17 Forward Baseline**: Remains 100% frozen, untouched, and scheduled to record prospective observations starting October 6, 2026.
- **Phase 18 Candidate Promotion**: **None promoted.** All 3 hypotheses are formally archived without unconstrained parameter tuning.
- **Real Money**: **OFF.**
