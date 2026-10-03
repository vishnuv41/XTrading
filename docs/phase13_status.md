# Phase 13 Master Status File

```text
STATUS: FROZEN / PROSPECTIVE

Candidate Specification:
50/50 XGBoost + CatBoost continuous-return regression

Features:
66 causal MTF features

Gate:
Top 1% percentile threshold (0.9900) over rolling 250 predictions

Risk:
1.5 ATR Stop Loss
3.0 ATR Take Profit

Timeout:
48 bars (1-hour candles)

Cooldown:
4 bars

Friction:
10 BPS (0.10%) + slippage

Start:
2026-09-06 14:00 UTC

Engineering boundary:
2026-09-09 07:00 UTC

Current:
N = 4 / 30 completed trades
2 Wins / 2 Losses
Net realized = +$4.18 (+0.04% return)

Decision gate:
N >= 30
PF >= 1.30
Sharpe >= 1.00
Consistency >= 60%
Loss probability < 15%
CI lower > -20%

NO INTERIM OPTIMIZATION.
```

---

## Current Baseline Summary

| Parameter | Value |
| :--- | :--- |
| **Completed Trades ($N$)** | **4 / 30** |
| **Wins / Losses** | **2 / 2 (50.0% Win Rate)** |
| **Realized Net P&L** | **$+\$4.18 (+0.04%)** |
| **Portfolio Position** | **Flat (0 active positions)** |
| **Model Candidate State** | **🔒 100% Frozen & Untouched** |
