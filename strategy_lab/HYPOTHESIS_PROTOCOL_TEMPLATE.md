# STRATEGY LAB HYPOTHESIS PROTOCOL TEMPLATE

**Protocol Name**: `HYPOTHESIS_<ID>_<NAME>`  
**Creation Date**: YYYY-MM-DD  
**Status**: DRAFT / PRE-REGISTERED / EXECUTED  
**Author**: Quantitative Research Lab

---

## 1. Hypothesis & Economic Mechanism

- **Hypothesis Statement**: (Explicit, testable statement of alpha mechanism).
- **Underlying Market Inefficiency**: (Why this edge should exist and who is on the losing side).
- **Timeframe & Target Asset**: (e.g., 4H or 1D on BTC/USDT, ETH/USDT, SOL/USDT).

---

## 2. Statistical Power & Required Sample Size ($N$)

- **Estimated Per-Trade Return Standard Deviation ($\sigma$)**: e.g., $250.0\text{ bps}$ ($1.5\text{R}$).
- **Minimum Target Net Edge ($\Delta$)**: e.g., $+30.0\text{ bps}$ ($+0.20\text{R}$).
- **Target Statistical Power ($1 - \beta$)**: $80\%$ at $\alpha = 0.05$ (two-sided).
- **Required Sample Size Formula**:
  $$N = \left(\frac{Z_{\alpha/2} + Z_{\beta}}{\Delta / \sigma}\right)^2 = \left(\frac{1.96 + 0.84}{\Delta / \sigma}\right)^2$$
- **Pre-Registered Minimum $N$**: (Must reach required $N$ before declaring victory or defeat).
- **Data Universe**: Multi-year historical data (e.g., 2018–2026 across major liquid symbols) to satisfy required $N$ on 4H/1D timeframes.

---

## 3. Causal Timing & Feature Integrity

- **Causal Constraint**: Every feature $F_t$ MUST be computed strictly using data available at or before bar close $t-1$:
  $$\text{Feature}_t = f(\{O_i, H_i, L_i, C_i, V_i\}_{i \le t-1})$$
- **Code Assertion**: Pipeline must execute automated assertion verifying zero future bar access.

---

## 4. Cost Model & Position Geometry

- **Execution Cost Baseline**: **30.0 BPS Round-Trip** ($10\text{ bps}$ fee + $5\text{ bps}$ slippage per leg).
- **Position Barriers**: $+3.0\text{ ATR}$ Take Profit / $-1.5\text{ ATR}$ Stop Loss (14-bar Wilder's ATR).
- **Scale-Free R-Multiple**:
  - $\text{Risk Distance (bps)} = (1.5 \times \text{ATR} / \text{Price}) \times 10,000$
  - $\text{Friction (R)} = 30.0 / \text{Risk Distance (bps)}$
  - $\text{Take Profit Net R} = +2.0\text{R} - \text{Friction (R)}$
  - $\text{Stop Loss Net R} = -1.0\text{R} - \text{Friction (R)}$

---

## 5. Null Benchmark & Success Criteria

- **Exhaustive Empirical Null**: Computed across all eligible bars in the evaluation window with time-matching.
- **Success Criteria**:
  1. $\bar{R}_{\text{net}} > 0.0\text{R}$ on reserved holdout.
  2. 5-bar block bootstrap 95% Confidence Interval lower bound $> 0.0\text{R}$.
  3. Net True R Lift over Exhaustive Null $\Delta R > +0.20\text{R}$ ($Z > 2.0$, $p < 0.05$).
- **Single Test Rule**: The reserved holdout window is evaluated ONCE. No iterative parameter tuning.
