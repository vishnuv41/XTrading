# Phase 13 Canonical Operational Protocol & Parameter Freeze

## 🔒 Parameter Freeze Declaration

Phase 13 candidate formulation:
> **50/50 XGBoost + CatBoost continuous-return regression → 66 causal MTF features → 250-prediction rolling Top-1% rank gate → 1.5 ATR SL / 3.0 ATR TP → 48-bar timeout → 4-bar cooldown → 10 BPS fees/slippage**

The following parameters are **100% frozen** and must not be altered for the duration of Phase 13 prospective evaluation:

| Parameter | Frozen Value |
| :--- | :--- |
| **Model Candidate** | 50/50 XGBoost + CatBoost continuous-return regression |
| **Feature Set** | 66 Multi-Timeframe (MTF) causal features |
| **Percentile Gate** | Top 1.0% percentile threshold ($0.9900$) over 250 rolling predictions |
| **Stop Loss (SL)** | $1.5 \times \text{ATR}_{14}$ |
| **Take Profit (TP)** | $3.0 \times \text{ATR}_{14}$ |
| **Timeout Limit** | 48 1-hour bars |
| **Trade Cooldown** | 4 1-hour bars |
| **Trading Frictions** | 10 BPS (0.10%) fees + fill slippage |
| **Position Heat** | 1.0% account risk heat per trade |
| **Assets** | BTC/USDT and ETH/USDT |
| **Mode** | Paper-only append-only PostgreSQL ledger |

---

## 🛑 Definition of a Phase 13 Completed Trade

Only completed trades count toward $N$:

$$\text{OPEN} \longrightarrow (\text{SL / TP / Timeout Exit}) \longrightarrow \text{CLOSE}$$

**Strict Exclusions (DO NOT count toward $N$)**:
- Signal generated without entry (e.g. cash limit)
- `HOLD` decisions
- Rejected or duplicate signals
- Active floating positions ($N$ increments strictly on position `CLOSE`)

---

## 📊 Pre-Registered Decision Criteria ($N \ge 30$)

No statistical decision gate evaluation is valid before completing **$N \ge 30$ completed trades**.

When $N \ge 30$, the candidate will be formally evaluated against these exact pre-registered thresholds:

| Metric | Target Acceptance Gate |
| :--- | :--- |
| **Completed Trades ($N$)** | $\ge 30$ |
| **Profit Factor ($PF$)** | $\ge 1.30$ |
| **Sharpe Ratio** | $\ge 1.00$ |
| **Subgroup Consistency** | $\ge 60\%$ positive sub-blocks |
| **Bootstrap Loss Probability** | $< 15\%$ |
| **Bootstrap 95% CI Lower Bound** | $> -20\%$ net return |

---

## 🚫 Rule of Non-Interference

1. **Zero Peeking / Early Interventions**: No statistical acceptance or rejection decision is valid before completing **$N \ge 30$ completed trades**.
2. **Read-Only Analytics**: All monitoring scripts, watchdogs, and dashboards must remain strictly read-only with ZERO feedback loop into model execution or position management.
3. **Current Baseline State**:
   - Prospective trades: **$N = 4 / 30$** (2 Wins / 2 Losses)
   - Realized P&L: **$+\$4.18$ (+0.04% return)**
   - Open Positions: **Flat (0 active positions)**
