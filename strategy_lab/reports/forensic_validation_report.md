# Strategy Lab V2 — Forensic Validation Report

*Purely offline research — Zero Phase 13 live execution contamination.*

## 1. Candidate B: Donchian Volatility Breakout (BTC/USDT 4H)

### Test 1 — Full History & Causal Parity Audit
- **Period**: `2026-06-19 00:00:00+00:00` to `2026-09-10 04:00:00+00:00`
- **Trades ($N$)**: `12`
- **Win Rate**: `41.7%`
- **Profit Factor**: `1.01`
- **Net Return**: `+0.12%` ($+11.55)
- **Max Drawdown**: `6.78%`

### Test 2 — Walk-Forward Out-of-Sample (OOS) Validation

| Window | Start | End | Trades (N) | Win Rate % | Profit Factor | Net P&L % | Max DD % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Window 1 | 2026-06-19 | 2026-07-30 | 6 | 16.7% | 0.24 | -4.32% | 4.58% |
| Window 2 | 2026-07-09 | 2026-08-20 | 7 | 42.9% | 1.22 | +1.01% | 4.57% |
| Window 3 | 2026-07-30 | 2026-09-10 | 6 | 83.3% | 7.07 | +7.40% | 1.12% |


### Test 3 — Market Regime Stratification

| Regime | Trades (N) | Wins | Losses | Win Rate % | Net P&L ($) |
| --- | --- | --- | --- | --- | --- |
| BULLISH_TREND | 5 | 1 | 4 | 20.0% | $-256.47 |
| BEARISH_TREND | 2 | 0 | 2 | 0.0% | $-220.04 |
| SIDEWAYS_RANGING | 1 | 0 | 1 | 0.0% | $-113.17 |
| HIGH_VOLATILITY | 2 | 2 | 0 | 100.0% | $+258.44 |
| LOW_VOLATILITY | 2 | 2 | 0 | 100.0% | $+342.78 |


### Test 4 — Cost & Friction Sensitivity Analysis

| Cost Scenario | RT Friction BPS | Trades (N) | Win Rate % | Profit Factor | Net P&L % |
| --- | --- | --- | --- | --- | --- |
| 0 BPS (Zero Fee) | 0 BPS | 12 | 41.7% | 1.40 | +2.79% |
| 10 BPS (Taker Only) | 10 BPS | 12 | 41.7% | 1.26 | +1.92% |
| 20 BPS (Fee + Minor Slip) | 20 BPS | 12 | 41.7% | 1.13 | +1.00% |
| 30 BPS (Standard) | 30 BPS | 12 | 41.7% | 1.01 | +0.12% |
| 50 BPS (High Friction) | 50 BPS | 12 | 41.7% | 0.82 | -1.51% |


### Test 5 — Parameter Perturbation & Stability

| Channel Period | Trades (N) | Win Rate % | Profit Factor | Net P&L % | Max DD % |
| --- | --- | --- | --- | --- | --- |
| 16 | 14 | 42.9% | 1.05 | +0.45% | 7.91% |
| 18 | 13 | 38.5% | 0.87 | -1.14% | 7.91% |
| 20 | 12 | 41.7% | 1.01 | +0.12% | 6.78% |
| 22 | 12 | 41.7% | 1.01 | +0.12% | 6.78% |
| 24 | 11 | 45.5% | 1.19 | +1.30% | 5.68% |


---

## 2. Candidate D: ML Regime Router (ETH/USDT 4H)

### Test 1 — Full History & Causal Parity Audit
- **Period**: `2026-06-19 00:00:00+00:00` to `2026-09-10 04:00:00+00:00`
- **Trades ($N$)**: `12`
- **Win Rate**: `50.0%`
- **Profit Factor**: `1.19`
- **Net Return**: `+1.27%` ($+126.77)
- **Max Drawdown**: `4.83%`

### Test 2 — Walk-Forward Out-of-Sample (OOS) Validation

| Window | Start | End | Trades (N) | Win Rate % | Profit Factor | Net P&L % | Max DD % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Window 1 | 2026-06-19 | 2026-07-30 | 5 | 60.0% | 2.44 | +3.21% | 1.10% |
| Window 2 | 2026-07-09 | 2026-08-20 | 6 | 16.7% | 0.12 | -4.83% | 4.83% |
| Window 3 | 2026-07-30 | 2026-09-10 | 4 | 50.0% | 0.71 | -0.77% | 2.69% |


### Test 3 — Market Regime Stratification

| Regime | Trades (N) | Wins | Losses | Win Rate % | Net P&L ($) |
| --- | --- | --- | --- | --- | --- |
| BULLISH_TREND | 2 | 1 | 1 | 50.0% | $+68.66 |
| BEARISH_TREND | 1 | 0 | 1 | 0.0% | $-113.62 |
| SIDEWAYS_RANGING | 5 | 4 | 1 | 80.0% | $+501.59 |
| HIGH_VOLATILITY | 3 | 1 | 2 | 33.3% | $-212.59 |
| LOW_VOLATILITY | 1 | 0 | 1 | 0.0% | $-117.27 |


### Test 4 — Cost & Friction Sensitivity Analysis

| Cost Scenario | RT Friction BPS | Trades (N) | Win Rate % | Profit Factor | Net P&L % |
| --- | --- | --- | --- | --- | --- |
| 0 BPS (Zero Fee) | 0 BPS | 12 | 50.0% | 1.49 | +3.00% |
| 10 BPS (Taker Only) | 10 BPS | 12 | 50.0% | 1.38 | +2.43% |
| 20 BPS (Fee + Minor Slip) | 20 BPS | 12 | 50.0% | 1.28 | +1.84% |
| 30 BPS (Standard) | 30 BPS | 12 | 50.0% | 1.19 | +1.27% |
| 50 BPS (High Friction) | 50 BPS | 12 | 50.0% | 1.02 | +0.17% |


### Test 5 — Parameter Perturbation & Stability

| SL ATR Mult | Trades (N) | Win Rate % | Profit Factor | Net P&L % | Max DD % |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 17 | 23.5% | 0.67 | -5.24% | 12.69% |
| 1.25 | 15 | 33.3% | 0.81 | -2.25% | 8.16% |
| 1.5 | 12 | 50.0% | 1.19 | +1.27% | 4.83% |
| 1.75 | 9 | 77.8% | 3.93 | +6.45% | 1.08% |
| 2.0 | 9 | 77.8% | 3.48 | +5.38% | 1.07% |


