"""
experiments/evaluate_phase13_n30.py
------------------------------------
Authoritative, strictly real-data evaluation script for Phase 13.
Reads PostgreSQL trade_log records, computes full trade metrics and 10,000-resample
bootstrap statistics, and evaluates against pre-registered N >= 30 acceptance criteria.

STRICT RULE: REAL DATA ONLY — NO SYNTHETIC FALLBACK DATA.
"""

import sys
import os
import math
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.connection import get_engine

PHASE13_START_TS = "2026-09-06 14:00:00+00"
INITIAL_CAPITAL = 10000.0


def fetch_completed_trades():
    """Fetch completed trades from PostgreSQL. Completed = OPEN followed by CLOSE."""
    engine = get_engine()
    with engine.connect() as conn:
        raw_trades = conn.execute(text("""
            SELECT id, trade_id, symbol, side, action, ts, price, size, fee, stop_loss, take_profit, exit_reason, realized_pnl, realized_pnl_pct, bars_held
            FROM trade_log
            WHERE ts >= :start_ts
            ORDER BY ts ASC, id ASC
        """), {"start_ts": PHASE13_START_TS}).fetchall()

    trades_dict = {}
    for r in raw_trades:
        t_id = r[1]
        if t_id not in trades_dict:
            trades_dict[t_id] = {"open": None, "close": None}
        if r[4] == "OPEN":
            trades_dict[t_id]["open"] = r
        elif r[4] == "CLOSE":
            trades_dict[t_id]["close"] = r

    completed_trades = []
    for t_id, t_info in trades_dict.items():
        op = t_info["open"]
        cl = t_info["close"]
        if op and cl:
            pnl_usd = cl[12] if cl[12] is not None else 0.0
            pnl_pct = cl[13] * 100 if cl[13] is not None else 0.0
            entry_fee = op[8] if op[8] is not None else 0.0
            exit_fee = cl[8] if cl[8] is not None else 0.0

            completed_trades.append({
                "trade_id": t_id,
                "symbol": op[2],
                "side": op[3],
                "entry_ts": op[5],
                "exit_ts": cl[5],
                "entry_price": float(op[6]),
                "exit_price": float(cl[6]),
                "size": float(op[7]),
                "total_fee": entry_fee + exit_fee,
                "exit_reason": cl[11],
                "realized_pnl": float(pnl_usd),
                "realized_pnl_pct": float(pnl_pct),
                "bars_held": int(cl[14]) if cl[14] is not None else 0,
            })

    # Sort chronologically by exit time
    completed_trades.sort(key=lambda x: x["exit_ts"])
    return completed_trades


def run_bootstrap_analysis(pnls, initial_capital=INITIAL_CAPITAL, n_resamples=10000, seed=42):
    """Run 10,000 bootstrap resamples on realized trade PnLs."""
    if len(pnls) == 0:
        return {"loss_prob": 1.0, "ci_lower_pct": -100.0, "ci_upper_pct": -100.0}

    np.random.seed(seed)
    n = len(pnls)
    boot_returns_pct = []

    for _ in range(n_resamples):
        sample = np.random.choice(pnls, size=n, replace=True)
        sample_tot_pnl = np.sum(sample)
        ret_pct = (sample_tot_pnl / initial_capital) * 100.0
        boot_returns_pct.append(ret_pct)

    boot_returns_pct = np.array(boot_returns_pct)
    loss_count = np.sum(boot_returns_pct < 0.0)
    loss_prob = loss_count / n_resamples

    ci_lower = np.percentile(boot_returns_pct, 2.5)
    ci_upper = np.percentile(boot_returns_pct, 97.5)

    return {
        "loss_prob": loss_prob,
        "ci_lower_pct": ci_lower,
        "ci_upper_pct": ci_upper,
    }


def evaluate_phase13():
    """Authoritative evaluation of Phase 13 prospective trades."""
    trades = fetch_completed_trades()
    n = len(trades)
    pnls = [t["realized_pnl"] for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = sum(pnls)
    net_return_pct = (net_pnl / INITIAL_CAPITAL) * 100.0
    win_rate = (len(wins) / n * 100.0) if n > 0 else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (net_pnl / n) if n > 0 else 0.0
    total_fees = sum(t["total_fee"] for t in trades)

    # Sharpe Ratio calculation
    if n > 1:
        returns = [p / INITIAL_CAPITAL for p in pnls]
        std_dev = np.std(returns, ddof=1)
        sharpe = (np.mean(returns) / std_dev * math.sqrt(8760 / max(1, np.mean([t["bars_held"] for t in trades])))) if std_dev > 0 else 0.0
    else:
        sharpe = 0.0

    # Drawdown calculation
    cum_pnl = 0.0
    max_eq = INITIAL_CAPITAL
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    for p in pnls:
        cum_pnl += p
        eq = INITIAL_CAPITAL + cum_pnl
        if eq > max_eq:
            max_eq = eq
        dd = max_eq - eq
        dd_pct = (dd / max_eq * 100.0) if max_eq > 0 else 0.0
        if dd > max_dd_usd:
            max_dd_usd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Subgroup consistency (5-trade sub-blocks)
    block_size = 5
    if n >= block_size:
        positive_blocks = 0
        total_blocks = 0
        for i in range(0, n - block_size + 1, block_size):
            sub = pnls[i:i + block_size]
            total_blocks += 1
            if sum(sub) > 0:
                positive_blocks += 1
        consistency_pct = (positive_blocks / total_blocks * 100.0) if total_blocks > 0 else 0.0
    else:
        consistency_pct = 0.0

    # Bootstrap evaluation
    boot = run_bootstrap_analysis(pnls)

    # Criteria evaluation
    c_n = n >= 30
    c_pf = profit_factor >= 1.30
    c_sharpe = sharpe >= 1.00
    c_consistency = consistency_pct >= 60.0
    c_loss_prob = boot["loss_prob"] < 0.15
    c_ci = boot["ci_lower_pct"] > -20.0

    all_criteria_met = c_n and c_pf and c_sharpe and c_consistency and c_loss_prob and c_ci

    loss_prob_pct_str = f"{boot['loss_prob']*100:.1f}%"
    ci_lower_pct_str = f"{boot['ci_lower_pct']:+.2f}%"
    pf_str = f"{profit_factor:.2f}"
    sharpe_str = f"{sharpe:.2f}"
    cons_str = f"{consistency_pct:.1f}%"

    print("\n" + "=" * 75)
    print("      PHASE 13 REPRODUCIBLE EVALUATION REPORT (REAL DATA ONLY)")
    print("=" * 75)
    print(f" Protocol Status             : FROZEN PROSPECTIVE PAPER TRADING")
    print(f" Completed Trades (N)        : {n} / 30 required (Progress: {n/30*100:.1f}%)")
    print(f" Account Equity             : ${INITIAL_CAPITAL + net_pnl:,.2f}  (Initial: ${INITIAL_CAPITAL:,.2f})")
    print(f" Net Realized P&L           : ${net_pnl:+,.2f} ({net_return_pct:+.2f}%)")
    print(f" Win / Loss Record          : {len(wins)} Wins / {len(losses)} Losses (Win Rate: {win_rate:.1f}%)")
    print(f" Gross Profit / Gross Loss  : ${gross_profit:,.2f} / ${gross_loss:,.2f}")
    print(f" Profit Factor              : {profit_factor:.2f}")
    print(f" Sharpe Ratio               : {sharpe:.2f}")
    print(f" Expectancy / Trade         : ${expectancy:+.2f}")
    print(f" Average Winner / Loser     : ${avg_win:+.2f} / ${avg_loss:+.2f}")
    print(f" Total Fees Paid            : ${total_fees:,.2f}")
    print(f" Max Drawdown               : ${max_dd_usd:.2f} ({max_dd_pct:.2f}%)")
    print(f" Subgroup Consistency (5b)  : {consistency_pct:.1f}%")
    print(f" Bootstrap Loss Probability : {loss_prob_pct_str}")
    print(f" Bootstrap 95% CI           : [{ci_lower_pct_str}, {boot['ci_upper_pct']:+.2f}%]")
    print("-" * 75)
    print(" PRE-REGISTERED DECISION GATE EVALUATION:")
    print(f"   1. Completed Trades N >= 30   : {'[MET]' if c_n else '[PENDING] (N=' + str(n) + '/30)'}")
    print(f"   2. Profit Factor >= 1.30      : {'[PASS] (' + pf_str + ')' if c_pf else '[FAIL] (' + pf_str + ')'}")
    print(f"   3. Sharpe Ratio >= 1.00       : {'[PASS] (' + sharpe_str + ')' if c_sharpe else '[FAIL] (' + sharpe_str + ')'}")
    print(f"   4. Consistency >= 60%        : {'[PASS] (' + cons_str + ')' if c_consistency else ('[PENDING] (N < 5)' if n < 5 else '[FAIL] (' + cons_str + ')')}")
    print(f"   5. Loss Probability < 15%     : {'[PASS] (' + loss_prob_pct_str + ')' if c_loss_prob else '[FAIL] (' + loss_prob_pct_str + ')'}")
    print(f"   6. CI Lower Bound > -20%      : {'[PASS] (' + ci_lower_pct_str + ')' if c_ci else '[FAIL] (' + ci_lower_pct_str + ')'}")
    print("-" * 75)

    if not c_n:
        print(" DECISION RESULT : [PENDING] ONGOING PROSPECTIVE VALIDATION (N < 30)")
        print("                   Rule of Non-Interference: Zero statistical gate evaluation until N >= 30.")
    elif all_criteria_met:
        print(" DECISION RESULT : [PASSED] PRE-REGISTERED VALIDATION GATE")
    else:
        print(" DECISION RESULT : [REJECTED] Candidate failed pre-registered criteria")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    evaluate_phase13()
