"""
strategy_lab/scripts/evaluate_out_of_time_2018_2020.py
------------------------------------------------------
Evaluates the 4 pre-registered advancing candidates + Buy & Hold benchmark on the
untouched 2018-2020 out-of-time dataset (2018-05-04 to 2020-08-11).
Universe: BTC, ETH, BNB, LTC, XRP, ADA (6 assets).
Friction: Canonical 30.0 bps round-trip friction.
"""

import sys
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.lowturn.panel_loader import load_panel_universe, load_symbol_ohlcv
from strategy_lab.lowturn.strategies import build_frozen_strategy_grid
from strategy_lab.lowturn.execution import simulate_portfolio_strategy
from strategy_lab.lowturn.bootstrap import calendar_week_block_bootstrap_sharpe_ci, compute_annualized_sharpe
from strategy_lab.lowturn.metrics import compute_max_drawdown, compute_annual_turnover

OOT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LTC/USDT", "XRP/USDT", "ADA/USDT"]
START_TS = datetime(2018, 5, 4, tzinfo=timezone.utc)
END_TS = datetime(2020, 8, 11, tzinfo=timezone.utc)

EVAL_STRATEGIES = [
    "Buy_and_Hold",
    "Trend_EMA_20",
    "Trend_EMA_50",
    "TSMOM_30d",
    "TSMOM_60d",
]


def run_oot_evaluation():
    print(f"=========================================================================")
    print(f"RUNNING PRE-REGISTERED OUT-OF-TIME EVALUATION (2018-05-04 to 2020-08-11)")
    print(f"Universe: {OOT_SYMBOLS} ({len(OOT_SYMBOLS)} assets)")
    print(f"Friction: 30.0 bps canonical round-trip | Execution: t+1")
    print(f"=========================================================================\n")

    panel = load_panel_universe(timeframe="1d", symbols=OOT_SYMBOLS, start_ts=START_TS, end_ts=END_TS)
    btc_df = load_symbol_ohlcv("BTC/USDT", timeframe="1d", start_ts=START_TS, end_ts=END_TS)
    grid = build_frozen_strategy_grid(timeframe="1d")

    # 1. Base Buy & Hold
    bh_weights = {sym: grid["Buy_and_Hold"](df) for sym, df in panel.items()}
    bh_res = simulate_portfolio_strategy(panel, bh_weights, round_trip_bps=30.0)
    bh_ret = bh_res["net_return"]
    bh_sharpe = compute_annualized_sharpe(bh_ret)
    bh_max_dd = compute_max_drawdown(bh_res["net_equity"])
    bh_ann_ret = float(bh_ret.mean() * 365)

    # Setup calendar week clustering for paired bootstrap
    df_clust = pd.DataFrame({"bh": bh_ret})
    naive_idx = df_clust.index.tz_localize(None) if df_clust.index.tz is not None else df_clust.index
    df_clust["year_week"] = naive_idx.to_period("W")
    unique_weeks = df_clust["year_week"].unique()
    n_weeks = len(unique_weeks)
    week_indices = [np.where(df_clust["year_week"] == w)[0] for w in unique_weeks]

    rng = np.random.default_rng(42)
    n_bootstraps = 2000

    results = []

    for strat_id in EVAL_STRATEGIES:
        weight_fn = grid[strat_id]
        weights = {sym: weight_fn(df) for sym, df in panel.items()}
        res = simulate_portfolio_strategy(panel, weights, round_trip_bps=30.0)
        net_ret = res["net_return"]
        net_equity = res["net_equity"]
        turnover = res["turnover"]

        ann_ret = float(net_ret.mean() * 365)
        ann_vol = float(net_ret.std() * np.sqrt(365))
        strat_sharpe, ci_lo, ci_hi = calendar_week_block_bootstrap_sharpe_ci(net_ret, n_bootstraps=2000, seed=42)
        max_dd = compute_max_drawdown(net_equity)
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0
        ann_turnover = compute_annual_turnover(turnover)

        # Paired Delta Sharpe vs Buy & Hold
        strat_vals = net_ret.values
        bh_vals = bh_ret.values
        strat_groups = [strat_vals[idx] for idx in week_indices]
        bh_groups = [bh_vals[idx] for idx in week_indices]

        boot_deltas = np.zeros(n_bootstraps)
        for b in range(n_bootstraps):
            sampled_w = rng.integers(0, n_weeks, size=n_weeks)
            s_s = np.concatenate([strat_groups[k] for k in sampled_w])
            s_b = np.concatenate([bh_groups[k] for k in sampled_w])
            
            s_std = np.std(s_s)
            b_std = np.std(s_b)
            
            sh_s = np.sqrt(365) * np.mean(s_s) / s_std if s_std > 1e-8 else 0.0
            sh_b = np.sqrt(365) * np.mean(s_b) / b_std if b_std > 1e-8 else 0.0
            boot_deltas[b] = sh_s - sh_b

        paired_lo = np.percentile(boot_deltas, 2.5)
        paired_hi = np.percentile(boot_deltas, 97.5)
        delta_sharpe = strat_sharpe - bh_sharpe

        results.append({
            "strategy": strat_id,
            "ann_net_return": ann_ret,
            "ann_vol": ann_vol,
            "net_sharpe": strat_sharpe,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "delta_sharpe": delta_sharpe,
            "paired_ci_lower": paired_lo,
            "paired_ci_upper": paired_hi,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "annual_turnover": ann_turnover,
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return results


if __name__ == "__main__":
    run_oot_evaluation()
