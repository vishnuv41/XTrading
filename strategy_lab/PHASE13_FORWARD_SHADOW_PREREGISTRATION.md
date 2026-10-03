# PHASE 13 FORWARD SHADOW RUN PRE-REGISTRATION & UNTOUCHED WINDOW AUDIT

**Repository Commitment Date**: October 3, 2026  
**Status**: FROZEN PRE-REGISTRATION & UNTOUCHED WINDOW AUDIT COMPLETE  
**Target Repository Path**: `strategy_lab/PHASE13_FORWARD_SHADOW_PREREGISTRATION.md`

---

## 1. Empirical Long-Only Random Null Benchmark (May 25 – Sept 11, 2026 Window)

Evaluating $N=100$ Random Long-Only entries across High Volatility bars (`ex_ante_vol >= 33.18 bps`) on the exact same 2,613-row window (May 25 – Sept 11, 2026):

| Strategy / Benchmark | $N$ Trades | TP Rate (%) | Mean Net Ret (bps) | Mean Net True R | **Net True R Lift vs Null** | 95% Bootstrap CI (Net R) |
| :--- | --: | --: | --: | --: | --: | :--- |
| **Model Top 5% Signal** | 78 | **33.33%** | **-27.43 bps** | **-0.200R** | **+0.130R** | [-0.508R, +0.146R] |
| **Empirical Long-Only Random Null** | 100 | **29.00%** | **-45.25 bps** | **-0.330R** | Baseline (0.0R) | [-0.610R, -0.050R] |

### Statistical Findings:
1. **Shrinkage from In-Sample Edge**: The in-sample Net R Lift of $+0.375\text{R}$ shrank to **+0.130R** on the May 25 – Sept 11, 2026 window.
2. **Confidence Intervals**: With $N=78$, the 95% Bootstrap CI for Model Net Return is **[-0.508R, +0.146R]**. The in-sample expectation ($+0.385\text{R}$) is decisively ruled out ($> 3.5\sigma$ away), while small positive or negative edges cannot be statistically resolved at $N=78$.

---

## 2. Code Audit & Execution History Log

`evaluate_unused_window.py` run history and parameters audit:
- **Runs 1–4**: Resolved code integration & dataframe indexing errors (`prepare_model_input` missing MTF columns, `ATR14` column name, pandas timestamp indexing).
- **Units Audit**: Identified that `oos_predictions.csv` measured trade-level volatility (~185 bps), whereas 1H candle return std has a median of **33.18 bps**. The threshold was fixed to 33.18 bps to align units without altering rules post-hoc.

---

## 3. Frozen Parameters & Repository Commitment

| Parameter | Frozen Value | Source / Provenance Hash |
| :--- | :--- | :--- |
| **Model Candidate** | `models_artifacts/BTCUSDT_1h_v2` | Isotonic-calibrated MTF Ensemble |
| **Ensemble Hash (`ensemble.pkl`)** | `d61be8b3ab269cf1b6f9dfc070c5ae427d38d20a0b458b54fd659dc20186638f` | SHA-256 Provenance Lock |
| **Calibrator Hash (`calibrator.pkl`)** | `c12585a36e914a3f75e8a903919b4a9379e07bbd2ca2858097f6bec7fd7e159a` | SHA-256 Provenance Lock |
| **Feature Hash (`feature_columns.pkl`)**| `0003e4051562a000f582ec085713b28e193cb443436c5840c5475625575c2122` | SHA-256 Provenance Lock |
| **Timeframe / Asset** | `1h` timeframe, `BTC/USDT` | Binance Futures Market Data |
| **Execution Cost Baseline** | **30.0 BPS Round-Trip** | Retail Taker: 10 bps fee + 5 bps slippage per leg |
| **Confidence Threshold** | `cal_p >= 0.6121` | **Exact 95th Percentile Cutoff** |
| **Ex-Ante Volatility Filter** | `ex_ante_vol >= 33.18 bps` | Trailing 20-bar 1H return std $\ge$ **Median Volatility** |
| **Position Barriers** | $+3.0\text{ ATR}$ Take Profit / $-1.5\text{ ATR}$ Stop Loss | $2.0 \times \text{Risk-Reward Ratio}$ |
| **ATR Window** | 14 bars (`atr_14`) | Standard Wilder's Smoothing |
| **Position Rule** | **Single Position Only** | Max 1 position active per symbol, zero pyramiding |
| **Append-Only Trade Log** | `strategy_lab/shadow_trades.jsonl` | Persistent local & remote trade logging |
