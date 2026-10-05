# PHASE 13 FORWARD SHADOW RUN PRE-REGISTRATION & METHODOLOGICAL AUDIT

**Repository Commitment Date**: October 3, 2026 (Updated: October 5, 2026)  
**Status**: FROZEN PRE-REGISTRATION & CAUSAL AUDIT COMPLETE  
**Target Repository Path**: `strategy_lab/PHASE13_FORWARD_SHADOW_PREREGISTRATION.md`  
**Look Ledger File**: [`strategy_lab/LOOK_LEDGER.jsonl`](file:///d:/all/XTrading_combined%20%281%29/XTrading/strategy_lab/LOOK_LEDGER.jsonl)

---

## 1. Exhaustive Null Benchmark on May 25 – Sept 11, 2026 Window

Evaluated against the **Exhaustive Long-Only Null** (all $N=1,297$ High Volatility bars in the window, rather than a subsample):

| Strategy / Benchmark | $N$ Trades / Bars | TP Rate (%) | Mean Net True R | **Net True R Lift vs Null** | 95% Paired Lift CI |
| :--- | --: | --: | --: | --: | :--- |
| **Model Top 5% Signal** | 78 | **33.33%** | **-0.200R** | **+0.123R** ($Z = +0.18$) | **[-0.187R, +0.439R]** |
| **Exhaustive Long-Only Null** | 1,297 | **32.38%** | **-0.323R** | Baseline (0.0R) | N/A |

### Definitive Methodological Conclusions:
1. **Absolute Return Failed to Replicate**: The model's absolute net return on the clean untouched window is $\bar{R}_{\text{net}} = -0.200\text{R}$ [95% CI: $-0.508\text{R}, +0.146\text{R}$], decisively ruling out in-sample profitability ($+0.385\text{R}$, $> 3.5\sigma$ away).
2. **Relative Skill is Unresolved**: The model's TP rate ($33.33\%$) is within $+0.95\%$ of the Exhaustive Long Null ($32.38\%$), yielding a Net True R Lift of **+0.123R** [95% Paired CI: $-0.187\text{R}, +0.439\text{R}$], with $Z = +0.18$ ($p = 0.44$). Relative lift cannot be statistically resolved at $N=78$.
3. **Execution Viability**: Because capital trades absolute P&L and cannot capture relative lift alone, the strategy is unviable under retail taker friction ($30\text{ bps}$).

---

## 2. Dataset Provenance & Date Range Verification

- `oos_predictions.csv` ($N=11,139$ rows) is a multi-symbol stacked dataset across 3 assets from Dec 9, 2025 to May 25, 2026 (~166 days / 5.5 months per asset):
  - `BTC/USDT`: 3,704 rows
  - `ETH/USDT`: 3,674 rows
  - `SOL/USDT`: 3,761 rows
- The Discovery split ($N=8,911$ rows) represents the first 80% chronological slice across these stacked symbols.

---

## 3. Protocol Safeguards for Low-Turnover Pivot (4H / 1D)

For all upcoming research under **Bottleneck #2**:
1. **Protocol Template**: All hypotheses must follow [`strategy_lab/HYPOTHESIS_PROTOCOL_TEMPLATE.md`](file:///d:/all/XTrading_combined%20%281%29/XTrading/strategy_lab/HYPOTHESIS_PROTOCOL_TEMPLATE.md).
2. **Sample Size & Multi-Year History**: Power-based minimum $N$ ($N \ge 250$–$350$) must be satisfied using multi-year history (2018–2026 across major liquid symbols) before test execution.
3. **Persistent Look Ledger**: All runs and trial partitions must be logged in [`strategy_lab/LOOK_LEDGER.jsonl`](file:///d:/all/XTrading_combined%20%281%29/XTrading/strategy_lab/LOOK_LEDGER.jsonl).
