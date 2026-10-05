# Pre-Registration: Forward Paper-Trading Protocol for Baseline `Trend_EMA_50`

**Protocol Creation Timestamp**: 2026-10-05T03:55:00Z  
**Status**: COMMITTED & FROZEN FOR 12-MONTH FORWARD EVALUATION  
**Pre-Registered Review Date**: 2027-10-05 (12 months forward, zero interim peeking/tuning)  

---

## 1. Objective
Establish an unpolluted, forward paper-trading track record for the canonical low-turnover baseline (`Trend_EMA_50`, Daily) across the 9-asset crypto universe, evaluated net of canonical 30.0 bps round-trip friction and compared concurrently against two control benchmarks.

---

## 2. Universe & Data Specifications
* **Universe (9 assets)**: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `XRP/USDT`, `ADA/USDT`, `LTC/USDT`, `SOL/USDT`, `DOGE/USDT`, `LINK/USDT`.
* **Execution Frequency**: Daily (00:00 UTC) at closed candle boundaries.
* **Execution Lag**: Signals generated at close $t$; rebalanced at open $t+1$.
* **Friction Model**: Canonical 30.0 bps round-trip (15.0 bps per entry/exit leg).

---

## 3. Parallel Tracking Arms (Starting Notional: $10,000 USD each)
1. **Arm A (Candidate Baseline)**: `Trend_EMA_50`
   - Weight per asset $i$: $w_{i, t} = \frac{1}{9} \cdot \mathbf{1}_{\{ \text{Close}_{i, t} > \text{EMA50}_{i, t} \}}$.
2. **Arm B (100% Crypto Beta Control)**: `Buy_and_Hold`
   - Weight per asset $i$: $w_{i, t} = \frac{1}{9}$ (always fully invested).
3. **Arm C (Static De-leveraging Control)**: `Static_Exposure_45pct`
   - Weight per asset $i$: $w_{i, t} = \frac{0.45}{9}$ (45% total crypto exposure, 55% cash).

---

## 4. Evaluation Criteria (To Be Scored on 2027-10-05)
1. **Net Sharpe Ratio**: Annualized Sharpe net of 30.0 bps friction.
2. **Paired Difference**: $\Delta \text{Sharpe} = \text{Sharpe}_{\text{EMA50}} - \text{Sharpe}_{\text{B\&H}}$ and $\Delta \text{Sharpe} = \text{Sharpe}_{\text{EMA50}} - \text{Sharpe}_{\text{Static45}}$.
3. **Maximum Drawdown**: Realized peak-to-trough drawdown across the 12-month forward period.
4. **Statistical Test**: Date-clustered calendar-week bootstrap CI on paired difference.

---

## 5. Frozen Execution Rules
* **No Interim Tuning**: Strategy parameters (EMA span = 50, universe = 9 symbols, friction = 30 bps) are strictly immutable.
* **Process Isolation**: Operates in independent persistence state (`strategy_lab/paper/state/`), completely segregated from Phase 13 single-asset engines.
