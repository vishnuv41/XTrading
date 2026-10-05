# Pre-Registration: Out-of-Time Secondary Robustness Evaluation (2018–2020)

**Timestamp**: 2026-10-05T03:40:00Z
**Status**: COMMITTED PRIOR TO EXECUTION ON 2018-2020 DATA

---

## 1. Motivation & Window Definition
The primary development (2020–2025) and holdout (2025–2026) panels were truncated to August 11, 2020 due to the listing date of SOL/USDT.
However, 6 major universe assets (BTC, ETH, BNB, LTC, XRP, ADA) possess clean, verified historical data dating back to May 4, 2018.
The period **2018-05-04 to 2020-08-11** (830 daily bars, ~2.27 years) was **never used** to select, tune, or optimize any strategy candidate. It represents an untouched out-of-time stress test encompassing:
1. The 2018 Crypto Winter (BTC dropped ~84%, ETH dropped ~94%).
2. The 2019 rally and subsequent consolidation.
3. The March 2020 liquidity shock / flash crash.

---

## 2. Pre-Registered Panel & Strategies
* **Universe**: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `LTC/USDT`, `XRP/USDT`, `ADA/USDT` (6 assets, equal-weighted).
* **Friction**: Canonical 30.0 bps round-trip friction (15.0 bps per leg).
* **Execution**: Next-day open ($t+1$) execution.
* **Strategies to Evaluate**:
  1. `Buy_and_Hold` (Benchmark)
  2. `Trend_EMA_20` (Advancing candidate 1)
  3. `Trend_EMA_50` (Advancing candidate 2)
  4. `TSMOM_30d` (Advancing candidate 3)
  5. `TSMOM_60d` (Advancing candidate 4)

---

## 3. Pre-Declared Pass/Evaluation Criteria
* **Primary Criterion**: Risk mitigation — Drawdown reduction relative to Buy & Hold ($\text{MaxDD}_{\text{strat}} < \text{MaxDD}_{\text{B\&H}}$).
* **Secondary Criterion**: Positive Net Sharpe ratio over the 2018–2020 bear/recovery cycle ($Net Sharpe > 0.00$).
* **Reporting**: Point estimate Net Sharpe, paired $\Delta \text{Sharpe}$ vs Buy & Hold with 95% Date-Cluster Bootstrap CI, Max Drawdown, and Calmar Ratio.
* **Strict Commitment**: Zero post-hoc parameter adjustments. Single evaluation run.
