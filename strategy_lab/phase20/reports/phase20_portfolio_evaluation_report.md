# Phase 20 Research: Portfolio Construction & Risk Management Report

**Status**: ❌ **ALL 3 CANDIDATE MECHANISMS FAILED ACCEPTANCE GATES**  
**Dataset Window**: Development Window (2020-08-11 to 2024-12-31, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Canonical Friction**: 30.0 bps round-trip transaction friction (swept 0 to 50 bps)  
**Pre-Registered Seed**: 42 (10,000 paired bootstrap resamples per candidate)

---

## 1. Executive Summary Table

| Candidate Mechanism | Net Annual Return | Baseline Return | Net Sharpe (30 bps) | Baseline Sharpe | $\Delta\text{Sharpe}$ | Paired 95% Bootstrap CI | One-Tailed $p$-val | Friction Sweep (0 / 30 / 50 bps) | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Risk_Parity_Sizing` ($1/\text{ATR}$) | +78.75% | +93.41% | 1.295 | 1.484 | -0.189 | `[-0.352, -0.010]` | 0.9802 | 1.405 / 1.295 / 1.221 | -56.37% | ❌ **FAIL** (CI Negative) |
| `Breadth_Scaled_Exposure` ($E_t$) | +94.38% | +93.41% | 1.509 | 1.484 | +0.025 | `[-0.069, +0.123]` | 0.2979 | 1.591 / 1.509 / 1.454 | -54.01% | ❌ **FAIL** (CI spans 0) |
| `Composite_Risk_Portfolio` | +80.59% | +93.41% | 1.324 | 1.484 | -0.160 | `[-0.346, +0.046]` | 0.9331 | 1.428 / 1.324 / 1.254 | -53.29% | ❌ **FAIL** (CI/G2) |

---

## 2. Core Quantitative Discoveries

1. **The Altcoin Convexity Suppression Effect**:
   In equity markets, inverse-volatility weighting often improves Sharpe by penalizing high-volatility tail risk. In cryptocurrency markets, however, high-beta altcoins (e.g. SOL, DOGE) provide disproportionate right-tail convexity during bull market expansions. Penalizing them proportional to their historical ATR systematically reduced net return by $-14.7\%$ and degraded Net Sharpe by $-0.189$ with a **statistically significant negative confidence interval** (`[-0.352, -0.010]`, $p=0.9802$).
2. **Equal Weighting Remains the Most Robust Construction**:
   Equal weighting ($1/9$ per trending asset) maintains maximum exposure to multi-coin dispersion and right-tail explosive moves without suffering from volatility estimation noise or turnover drag.
3. **Continuous Breadth Scaling**:
   While continuous breadth scaling yielded a minor positive point-estimate lift ($\Delta\text{Sharpe} = +0.025$), the paired bootstrap CI spans zero (`[-0.069, +0.123]`, $p=0.2979$), failing to demonstrate statistically distinguishable alpha.

---

## 3. Archival Decision

Per pre-registration rules:
- All 3 Phase 20 portfolio mechanisms are **formally archived**.
- None will replace the canonical equal-weighted `Trend_EMA_50` baseline.
- **Phase 17 Forward Baseline remains the sole active benchmark** accumulating prospective evidence.
