# PHASE 13 FORWARD SHADOW RUN PRE-REGISTRATION SPECIFICATION

**Repository Commitment Date**: October 3, 2026  
**Status**: FROZEN PRE-REGISTRATION (BINDING RECORD)  
**Target Repository Path**: `strategy_lab/PHASE13_FORWARD_SHADOW_PREREGISTRATION.md`

---

## 1. Frozen Parameters & Environment

The forward shadow run evaluates the calibrated ML model combined with ex-ante volatility filtering under prospective live market conditions starting **October 3, 2026**.

| Parameter | Frozen Value | Source / Verification |
| :--- | :--- | :--- |
| **Model Candidate** | `models_artifacts/BTCUSDT_1h_v2` | Isotonic-calibrated Ensemble |
| **Timeframe / Asset** | `1h` timeframe, `BTC/USDT` | Binance Futures Market Data |
| **Execution Cost Baseline** | **30.0 BPS Round-Trip** | Retail Taker: 10 bps fee + 5 bps slippage per leg |
| **Confidence Threshold** | `cal_p >= 0.6121` | **Exact 95th Percentile Cutoff** (Discovery Set) |
| **Ex-Ante Volatility Filter** | `ex_ante_vol >= 185.4 bps` | Trailing 20-bar return std $\ge$ **Discovery Median** |
| **Position Barriers** | $+3.0\text{ ATR}$ Take Profit / $-1.5\text{ ATR}$ Stop Loss | $2.0 \times \text{Risk-Reward Ratio}$ |
| **ATR Window** | 14 bars (`atr_14`) | Standard Wilder's Smoothing |
| **Position Rule** | **Single Position Only** | Max 1 position active per symbol, zero pyramiding |
| **Start Date** | **October 3, 2026** | Prospective live candle ingestion |

---

## 2. Stopping Rules & Success Criteria

1. **Target Sample Size**:
   - The run automatically stops upon reaching **$N = 100$ completed closed trades**.

2. **Pre-Registered Success Criteria**:
   - **Primary Metric**: Mean Net R-Multiple ($\bar{R}_{\text{net}}$) across $N=100$ trades under 30.0 bps cost.
   - **Success Gate**: $\bar{R}_{\text{net}} > 0.0\text{R}$ AND 5-bar block bootstrap 95% Confidence Interval lower bound $> 0.0\text{R}$.
   - **Alpha Lift Gate**: Net R-Multiple Lift over Random Null $\Delta R > +0.20\text{R}$.

3. **Failure Criterion**:
   - If $\bar{R}_{\text{net}} \le 0.0\text{R}$ or 95% block CI lower bound $\le 0.0\text{R}$ at $N=100$, the strategy is declared **unprofitable under retail taker friction** and retired.

---

## 3. Governance Isolation

- The Forward Shadow Run operates in **read-only shadow mode** (`persist_to_db=False`) inside `strategy_lab/`.
- Zero write access to production PostgreSQL tables (`trade_log`, `prediction_log`).
- Phase 13 prospective paper trading daemon (`:8000`) continues running independently as the unpolluted pre-registered prospective control.
