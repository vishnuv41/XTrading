# STRATEGY LAB HYPOTHESIS PROTOCOL TEMPLATE

**Protocol Identifier**: `HYPOTHESIS_<ID>_<NAME>`  
**Creation Date**: YYYY-MM-DD  
**Status**: DRAFT / PRE-REGISTERED / EXECUTED  
**Author**: Quantitative Research Lab

---

## 1. Hypothesis & Economic Mechanism

- **Hypothesis Statement**: (Explicit, testable statement of alpha mechanism).
- **Underlying Market Inefficiency**: (Why this edge should exist and who is on the losing side).
- **Timeframe & Target Asset Universe**: (e.g., 4H or 1D on BTC/USDT, ETH/USDT, SOL/USDT).

---

## 2. Statistical Power & Independent Cluster Architecture

- **Estimated Per-Trade Return Standard Deviation ($\sigma$)**: e.g., $1.5\text{R}$.
- **Minimum Target Net Edge ($\Delta$)**: e.g., $+0.20\text{R}$ ($+30.0\text{ bps}$).
- **Target Statistical Power ($1 - \beta$)**: $80\%$ at $\alpha = 0.05$ (two-sided).
- **Independent Cluster Sample Size Formula**:
  $$N_{\text{clusters}} = \left(\frac{Z_{\alpha/2} + Z_{\beta}}{\Delta / \sigma}\right)^2 = \left(\frac{1.96 + 0.84}{\Delta / \sigma}\right)^2$$
- **Pre-Registered Minimum $N$**: Minimum $N = 250\text{--}350$ independent time clusters (e.g., multi-year weekly/daily blocks across 2018–2026).
- **Cross-Sectional Clustered Bootstrap**: All confidence intervals must be computed via **Date-Cluster Bootstrap** (resampling entire date/week blocks containing all pooled assets simultaneously) to preserve cross-asset co-movement.

---

## 3. Causal Timing & Feature Integrity

- **Strict Causal Boundary**: Every feature $F_t$ MUST use only data up to and including bar close $t-1$:
  $$\text{Feature}_t = f(\{O_i, H_i, L_i, C_i, V_i\}_{i \le t-1})$$
- **Automated Causal Assertion**: Code must execute an automated test asserting zero future bar access before any model fitting.

---

## 4. Cost Model & Scale-Free Position Geometry

- **Execution Cost Baseline**: **30.0 BPS Round-Trip** ($10\text{ bps}$ fee + $5\text{ bps}$ slippage per leg).
- **Position Barriers**: $+3.0\text{ ATR}$ Take Profit / $-1.5\text{ ATR}$ Stop Loss (14-bar Wilder's ATR).
- **Scale-Free Per-Trade R-Multiple**:
  $$\text{Risk Distance (bps)}_i = \frac{1.5 \times \text{ATR}_i}{\text{Price}_i} \times 10,000$$
  $$\text{Friction (R)}_i = \frac{30.0\text{ bps}}{\text{Risk Distance (bps)}_i}$$
  $$\text{Take Profit Net R}_i = +2.0\text{R} - \text{Friction (R)}_i$$
  $$\text{Stop Loss Net R}_i = -1.0\text{R} - \text{Friction (R)}_i$$

---

## 5. Null Benchmark, Look Ledger & Single-Test Rule

- **Exhaustive Empirical Null**: Computed across all eligible bars in the evaluation window using identical per-trade friction.
- **Look Ledger Registration**: Every exploratory partition, quantile sweep, and validation run must be logged in [`strategy_lab/LOOK_LEDGER.jsonl`](file:///d:/all/XTrading_combined%20%281%29/XTrading/strategy_lab/LOOK_LEDGER.jsonl).
- **Single-Test Reserved Holdout Rule**: The final reserved holdout dataset (e.g., chronological final 25% or post-Sept 11, 2026 data) is evaluated ONCE. No iterative parameter tuning.
- **Success Criteria**:
  1. Mean Net True R $\bar{R}_{\text{net}} > 0.0\text{R}$ on reserved holdout.
  2. Date-Cluster Bootstrap 95% CI lower bound $> 0.0\text{R}$.
  3. Net True R Lift over Exhaustive Null $\Delta R > +0.20\text{R}$ ($Z_{\text{cluster}} > 2.0$, $p < 0.05$).
