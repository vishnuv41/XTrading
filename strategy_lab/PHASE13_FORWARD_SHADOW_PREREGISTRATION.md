# PHASE 13 FORWARD SHADOW RUN PRE-REGISTRATION & UNTOUCHED HOLDOUT REPORT

**Repository Commitment Date**: October 3, 2026  
**Status**: FROZEN PRE-REGISTRATION & UNTOUCHED EVALUATION COMPLETE  
**Target Repository Path**: `strategy_lab/PHASE13_FORWARD_SHADOW_PREREGISTRATION.md`

---

## 1. Frozen Parameters & Artifact Hashes

The strategy evaluates the calibrated ML model combined with ex-ante volatility filtering under 30.0 BPS retail taker cost baseline:

| Parameter | Frozen Value | Source / Hash |
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

---

## 2. Once-Only Evaluation on Untouched Window (May 25 – Sept 11, 2026)

Applied the frozen rule ONCE to the completely untouched OHLCV data window ($N=2,613$ hourly rows from May 25, 2026 to September 11, 2026):

| Metric | Empirical Result | Null Baseline Benchmark | Status |
| :--- | --: | --: | :--- |
| **Total Closed Trades ($N$)** | **78 trades** | N/A | Completed Window Run |
| **Take Profit Rate (%)** | **33.33%** (26 / 78) | 33.33% Theoretical Null | **Matches Fair Null Exactly** |
| **Mean Net Return (bps)** | **-27.43 bps** | -30.00 bps (Friction Drag) | **Negative Net Expectancy** |
| **Mean Net True R-Multiple** | **-0.200R** | -0.200R | **Zero Excess Edge** |

### Definitive Conclusion:
On fresh untouched out-of-sample data, the in-sample high-volatility lift suffered complete **winner's curse shrinkage**. The strategy's TP hit rate ($33.33\%$) matched the random fair-game null exactly, producing a net loss of **-27.43 bps (-0.200R)** per trade under retail friction ($30\text{ bps}$).

---

## 3. Stopping Rules & Pre-Registered Criteria

1. **Target Sample Size**: Stop upon reaching $N=100$ forward closed trades.
2. **Pre-Registered Success Criteria**:
   - Mean Net True R-Multiple $\bar{R}_{\text{net}} > 0.0\text{R}$ AND 5-bar block bootstrap 95% CI lower bound $> 0.0\text{R}$.
   - Net R-Multiple Lift over Time-Matched Random Null $\Delta R > +0.20\text{R}$.
3. **Current Status**: Strategy fails prospective edge validation on untouched data. Phase 13 prospective paper trading continues independently as a live prospective control.
