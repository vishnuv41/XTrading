# PHASE 13 FORWARD SHADOW RUN PRE-REGISTRATION & METHODOLOGICAL AUDIT

**Repository Commitment Date**: October 3, 2026  
**Status**: FROZEN PRE-REGISTRATION & CAUSAL AUDIT COMPLETE  
**Target Repository Path**: `strategy_lab/PHASE13_FORWARD_SHADOW_PREREGISTRATION.md`  
**Look Ledger File**: [`strategy_lab/LOOK_LEDGER.jsonl`](file:///d:/all/XTrading_combined%20%281%29/XTrading/strategy_lab/LOOK_LEDGER.jsonl)

---

## 1. Causal Audit & Look-Ahead Verification

A strict causal audit of `ex_ante_vol` (trailing 20-bar 1H return standard deviation ending strictly prior to the signal bar) was executed on the Discovery dataset ($N=8,911$ rows, Dec 2025 to May 2026):

| Dataset Window | Regime | $N$ Trades | TP Rate (%) | Empirical Long Null TP (%) | Net True R | **Net True R Lift vs Null** |
| :--- | :--- | --: | --: | --: | --: | :--- |
| **Discovery (Dec 25 – May 26)** | **High Volatility** | 120 | **60.00%** | **40.33%** | **+0.600R** | **+0.590R** ($Z = +4.32$) |
| **Untouched (May 25 – Sep 11)** | **High Volatility** | 78 | **33.33%** | **29.00%** | **-0.200R** | **+0.130R** ($Z = +0.62$) |

### Causal Audit Findings:
1. **Zero Look-Ahead Leakage**: The ex-ante volatility calculation is strictly causal ($t-1$ shift). The in-sample high-volatility lift on Discovery ($+0.590\text{R}$) was not driven by look-ahead leakage.
2. **Winner's Curse Shrinkage**: On fresh untouched data (May 25 – Sep 11, 2026), the Net R Lift shrank from $+0.590\text{R}$ down to **+0.130R** ($33.33\%$ TP rate, $Z = +0.62$), displaying classical out-of-sample shrinkage.

---

## 2. Statistical Uncertainty & Lift CIs on Untouched Window ($N=78$)

Evaluating the untouched May 25 – Sept 11, 2026 window ($N=78$ trades) with proper paired standard error bounds:

- **Model TP Rate**: $33.33\%$ [95% Clopper-Pearson CI: **23.1%, 44.8%**]
- **Model Net True R**: $-0.200\text{R}$ [95% Block CI: **-0.508R, +0.146R**]
- **Lift over Empirical Long Null**: $\Delta R = \mathbf{+0.130R}$ [95% Paired Lift CI: **-0.310R, +0.570R**] ($Z = +0.62$, $p = 0.53$)

### Conclusion on Precision:
- The in-sample expectation ($+0.385\text{R}$) is **decisively ruled out** ($> 3.5\sigma$ away).
- Small positive or negative edges of $\pm 0.1\text{R} - 0.2\text{R}$ cannot be statistically resolved at $N=78$. The confidence-gated ML candidate is formally classified as **unproven**.

---

## 3. Look Ledger & Protocol Safeguards for Future Hypotheses

For all future research under **Bottleneck #2**:
1. **Look Ledger**: Every evaluation run on any window MUST be appended to `strategy_lab/LOOK_LEDGER.jsonl`.
2. **Protocol Pre-Registration**: One written markdown protocol file per hypothesis MUST be committed to git BEFORE inspecting numbers.
3. **Causal Verification**: Every feature MUST assert $t \le \text{signal\_bar}$ data boundary in code.
4. **Reserved Data Window**: All OHLCV data after September 11, 2026 (including prospective live ingestion) is frozen as a reserved holdout window.

---

## 4. Frozen Hashes & Artifact Provenance

| Parameter | Frozen Value | Provenance SHA-256 Hash |
| :--- | :--- | :--- |
| **Ensemble Hash (`ensemble.pkl`)** | `d61be8b3ab269cf1b6f9dfc070c5ae427d38d20a0b458b54fd659dc20186638f` | Locked |
| **Calibrator Hash (`calibrator.pkl`)** | `c12585a36e914a3f75e8a903919b4a9379e07bbd2ca2858097f6bec7fd7e159a` | Locked |
| **Feature Hash (`feature_columns.pkl`)**| `0003e4051562a000f582ec085713b28e193cb443436c5840c5475625575c2122` | Locked |
| **Cost Baseline** | **30.0 BPS Round-Trip** | Retail Taker (10 bps fee + 5 bps slip) |
| **Confidence Threshold** | `cal_p >= 0.6121` | Exact 95th Percentile |
| **Ex-Ante Volatility Filter** | `ex_ante_vol >= 33.18 bps` | Trailing 20-bar 1H return std median |
