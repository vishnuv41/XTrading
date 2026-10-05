"""
strategy_lab/scripts/evaluate_static_exposure_and_oot_nulls.py
--------------------------------------------------------------
Computes:
1. Static Exposure-Matched Benchmark (matching trend rule's empirical time-in-market)
   across Development (2020-2025), Holdout (2025-2026), and Out-of-Time (2018-2020) windows.
2. Exposure-Matched Null p-values on Holdout and Out-of-Time windows.
"""

import sys
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.lowturn.panel_loader import load_dev_and_holdout_panels, load_panel_universe, load_symbol_ohlcv
from strategy_lab.lowturn.strategies import build_frozen_strategy_grid
from strategy_lab.lowturn.execution import simulate_portfolio_strategy, simulate_single_asset_strategy
from strategy_lab.lowturn.bootstrap import calendar_week_block_bootstrap_sharpe_ci, compute_annualized_sharpe
from strategy_lab.lowturn.metrics import compute_max_drawdown, compute_annual_turnover
from strategy_lab.lowturn.exposure_null import evaluate_exposure_matched_null

# Windows
dev_panel, holdout_panel, split_meta = load_dev_and_holdout_panels("1d")
grid = build_frozen_strategy_grid("1d")

OOT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LTC/USDT", "XRP/USDT", "ADA/USDT"]
oot_start = datetime(2018, 5, 4, tzinfo=timezone.utc)
oot_end = datetime(2020, 8, 11, tzinfo=timezone.utc)
oot_panel = load_panel_universe("1d", symbols=OOT_SYMBOLS, start_ts=oot_start, end_ts=oot_end)

windows = {
    "Development (2020-2025, 9sym)": dev_panel,
    "Holdout (2025-2026, 9sym)": holdout_panel,
    "Out-of-Time (2018-2020, 6sym)": oot_panel,
}

advancing_strats = ["Trend_EMA_50", "Trend_EMA_20", "TSMOM_30d", "TSMOM_60d"]

print("=========================================================================================")
print("1. STATIC EXPOSURE-MATCHED BENCHMARK VS DYNAMIC TREND TIMING")
print("=========================================================================================\n")

static_results = []

for win_name, panel in windows.items():
    # Evaluate Trend_EMA_50
    ema50_weights = {sym: grid["Trend_EMA_50"](df) for sym, df in panel.items()}
    ema50_res = simulate_portfolio_strategy(panel, ema50_weights, round_trip_bps=30.0)
    ema50_exposure = float(ema50_res["exposure"].mean())
    ema50_sharpe = compute_annualized_sharpe(ema50_res["net_return"])
    ema50_max_dd = compute_max_drawdown(ema50_res["net_equity"])
    ema50_ann_ret = float(ema50_res["net_return"].mean() * 365)

    # Evaluate 100% Buy & Hold
    bh_weights = {sym: grid["Buy_and_Hold"](df) for sym, df in panel.items()}
    bh_res = simulate_portfolio_strategy(panel, bh_weights, round_trip_bps=30.0)
    bh_sharpe = compute_annualized_sharpe(bh_res["net_return"])
    bh_max_dd = compute_max_drawdown(bh_res["net_equity"])
    bh_ann_ret = float(bh_res["net_return"].mean() * 365)

    # Evaluate Static Exposure-Matched Buy & Hold (fixed weight = ema50_exposure)
    static_weights = {sym: pd.Series(ema50_exposure, index=df.index) for sym, df in panel.items()}
    static_res = simulate_portfolio_strategy(panel, static_weights, round_trip_bps=30.0)
    static_sharpe = compute_annualized_sharpe(static_res["net_return"])
    static_max_dd = compute_max_drawdown(static_res["net_equity"])
    static_ann_ret = float(static_res["net_return"].mean() * 365)

    static_results.append({
        "window": win_name,
        "empirical_exposure": f"{ema50_exposure*100:.1f}%",
        "trend_ann_ret": f"{ema50_ann_ret*100:.1f}%",
        "trend_sharpe": f"{ema50_sharpe:.3f}",
        "trend_max_dd": f"{ema50_max_dd*100:.1f}%",
        "static_exp_ann_ret": f"{static_ann_ret*100:.1f}%",
        "static_exp_sharpe": f"{static_sharpe:.3f}",
        "static_exp_max_dd": f"{static_max_dd*100:.1f}%",
        "bh_full_sharpe": f"{bh_sharpe:.3f}",
        "bh_full_max_dd": f"{bh_max_dd*100:.1f}%",
    })

print(pd.DataFrame(static_results).to_string(index=False))

print("\n=========================================================================================")
print("2. EXPOSURE-MATCHED NULL P-VALUES ACROSS ALL THREE WINDOWS")
print("=========================================================================================\n")

null_results = []

for strat_id in advancing_strats:
    weight_fn = grid[strat_id]
    row = {"strategy": strat_id}
    
    for win_name, panel in windows.items():
        weights = {sym: weight_fn(df) for sym, df in panel.items()}
        res = simulate_portfolio_strategy(panel, weights, round_trip_bps=30.0)
        strat_sharpe = compute_annualized_sharpe(res["net_return"])
        
        p_val, null_mean, null_std = evaluate_exposure_matched_null(
            panel, weights, strat_sharpe, n_simulations=1000, round_trip_bps=30.0, seed=42
        )
        short_name = win_name.split(" ")[0]
        row[f"{short_name}_Sharpe"] = f"{strat_sharpe:.3f}"
        row[f"{short_name}_p_val"] = f"{p_val:.4f}"
        row[f"{short_name}_null_mean"] = f"{null_mean:.3f}"

    null_results.append(row)

print(pd.DataFrame(null_results).to_string(index=False))
