# Pre-Registration: Forward Paper-Trading Protocol for Baseline `Trend_EMA_50`

**Protocol Creation Timestamp**: 2026-10-05T04:15:00Z  
**Status**: COMMITTED & FROZEN FOR 12-MONTH FORWARD TRACKING  
**Pre-Registered Review Date**: 2027-10-05 (12 months forward, zero interim parameter tuning)  

---

## 1. Objective
Establish an unpolluted, forward paper-trading track record for the canonical low-turnover baseline (`Trend_EMA_50`, Daily) across the 9-asset crypto universe, evaluated net of canonical 30.0 bps round-trip friction and compared concurrently against two control benchmarks.

---

## 2. Universe & Execution Conventions
* **Universe (9 assets)**: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `XRP/USDT`, `ADA/USDT`, `LTC/USDT`, `SOL/USDT`, `DOGE/USDT`, `LINK/USDT`.
* **Execution Frequency**: Daily at 00:00:00 UTC closed candle boundary.
* **Execution Lag**: Signals calculated strictly on data available at daily close $t$; virtual fills executed at bar open $t+1$.
* **Friction Model**: Canonical 30.0 bps round-trip (15.0 bps per entry/exit leg).
* **Scheduling & Catch-Up**: The tracking harness checks `last_updated` against available closed candles. If the execution environment is offline for multiple days, the harness chronologically iterates through all missed calendar bars upon next execution, applying daily transitions idempotently.
* **Missing Data / Outage Fallback**: If an exchange outage prevents fetching candle $t$, previous day position weights are held until the next verified closed candle.

---

## 3. Parallel Tracking Arms (Starting Notional: $10,000 USD each)

1. **Arm A (Candidate Baseline)**: `Trend_EMA_50`
   - Weight per asset $i$: $w_{i, t} = \frac{1}{9} \cdot \mathbf{1}_{\{ \text{Close}_{i, t} > \text{EMA50}_{i, t} \}}$.
   - Rebalanced daily based on EMA 50 status; friction charged on daily weight deltas.
2. **Arm B (100% Crypto Beta Control)**: `Buy_and_Hold`
   - Weight per asset $i$: $w_{i, t} = \frac{1}{9}$ (always fully invested).
3. **Arm C (Static De-leveraging Control)**: `Static_Exposure_45pct`
   - Target weight per asset $i$: $w_{i} = \frac{0.45}{9} = 5.0\%$ (45% total crypto exposure, 55% cash).
   - **Rebalancing Rule**: Rebalanced back to target 45% on the **1st day of every calendar month**, charging 15.0 bps friction per leg on the rebalancing turnover. Drift is permitted during the month.

---

## 4. Decision Rule & Statistical Expectations for 2027-10-05

### Pre-Declared Success Criteria
1. **Sharpe Comparison**: Arm A Net Sharpe $> \text{Arm C}$ Net Sharpe over the 12-month forward period.
2. **Drawdown Comparison**: Arm A Max Drawdown not worse than Arm C Max Drawdown by more than 5.0 percentage points.

### Epistemic / Statistical Qualification
* Over a 1-year sample ($N=365$ daily bars), the standard error of the annualized Sharpe ratio is $\text{SE}(S) \approx \sqrt{\frac{1 + 0.5 \cdot S^2}{1}} \approx 1.0$.
* Therefore, a 12-month review cannot establish statistical significance at standard confidence levels ($\alpha = 0.05$). The 2027-10-05 review is pre-declared as a **descriptive forward operational milestone**, not a conclusive hypothesis test.

---

## 5. Persistence, State Isolation & Audit Trail
* **Process Isolation**: Completely segregated from Phase 13 single-position engines and live databases.
* **State Management**: Ephemeral state stored in `strategy_lab/paper/state/` (gitignored to prevent repository bloat and merge collisions).
* **Audit Trail**: Every daily bar transition is recorded in an immutable, append-only log: `strategy_lab/paper/forward_track_ledger.jsonl`, committed to the git repository during periodic audit reviews.
