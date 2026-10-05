# Phase 18 Alpha Research: Hypothesis H18-C (Pre-Entry Volatility Compression)

**Status**: ❌ **ALL 4 PARAMETER CONFIGURATIONS FAILED TO PASS GATES**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Friction**: Canonical 30.0 bps round-trip transaction friction  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples)

---

## 1. Executive Summary

Hypothesis H18-C tested whether filtering `Trend_EMA_50` entries to bars following a volatility compression (historical Bollinger Bandwidth or ATR ratio $< 20\text{th}$ percentile over lookbacks of 30 or 60 days) improves risk-adjusted returns and drawdown.

### Empirical Verdict
- **Gate 1 (Incremental Net Return $> 0$)**: ❌ **FAILED**. Net return dropped drastically across all variants (+4.5% to +31.5% vs +93.4% baseline).
- **Gate 2 (Incremental Sharpe $\ge +0.20$)**: ❌ **FAILED**. Net Sharpe deteriorated across all 4 variants ($\Delta\text{Sharpe} = -0.317$ to $-1.123$).
- **Gate 3 (Paired Bootstrap 95% CI Lower Bound $> 0$)**: ❌ **FAILED**. All paired CIs were negative ($p \ge 0.750$).
- **Gate 4 (Friction Robustness @ 45 bps)**: ✅ **PASSED**. Net Sharpe remained positive due to very low trading frequency.
- **Gate 5 (Turnover Efficiency)**: ❌ **FAILED**. Since gross excess alpha was negative, friction drag to gross alpha ratio failed.
- **Gate 6 (Regime Stability)**: ❌ **FAILED**. Severe multi-year underperformance relative to baseline.

---

## 2. Parameter Grid Results Table

| Candidate Configuration | Net Annualized Return | Baseline Return | Net Sharpe | Baseline Sharpe | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | One-Tailed $p$-value | Max Drawdown | All Gates Passed? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Bandwidth_L30d_Cutoff20pct` | +17.34% | +93.41% | 1.011 | 1.484 | -0.473 | `[-1.170, +0.230]` | 0.9095 | -32.24% | ❌ **FAIL** |
| `Bandwidth_L60d_Cutoff20pct` | +4.54% | +93.41% | 0.361 | 1.484 | -1.123 | `[-1.930, -0.303]` | 0.9971 | -28.87% | ❌ **FAIL** |
| `ATR_Ratio_L30d_Cutoff20pct` | +31.49% | +93.41% | 1.167 | 1.484 | -0.317 | `[-0.971, +0.519]` | 0.7506 | -23.09% | ❌ **FAIL** |
| `ATR_Ratio_L60d_Cutoff20pct` | +11.45% | +93.41% | 0.864 | 1.484 | -0.620 | `[-1.459, +0.212]` | 0.9287 | -16.20% | ❌ **FAIL** |

---

## 3. Core Insights & Why Volatility Compression Failed

1. **Massive Opportunity Cost (Missed Trend Breakouts)**:
   In cryptocurrency markets, the most explosive trend legs often initiate following high-volatility capitulation bottoms rather than prolonged low-volatility compressions. Restricting entry only to periods where volatility rank was in the bottom 20% caused the strategy to miss ~60–80% of major bull market expansions.
2. **Drawdown Reduction is an Artifact of Cash Drag**:
   While max drawdown decreased (e.g. $-16.2\%$ vs $-56.5\%$), this was caused by remaining in cash for the vast majority of the 4.4-year sample, not timing alpha.
3. **Statistically Significant Deterioration**:
   For the 60-day bandwidth compression variant, the 95% paired bootstrap CI was entirely negative ($[-1.930, -0.303]$ with $p=0.9971$), proving that the filter actively degraded performance.

---

## 4. Archival Decision

Per pre-registration rules, **Hypothesis H18-C is formally rejected and archived**. It will NOT advance to Holdout evaluation or live execution.
