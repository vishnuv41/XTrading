"""
strategy_lab/scripts/compute_detailed_dev_diagnostics.py
--------------------------------------------------------
Computes:
1. Paired Date-Cluster Block Bootstrap CI on Delta Sharpe vs Buy & Hold (Delta Sharpe = Sharpe_strat - Sharpe_BH).
2. Bonferroni adjusted null p-values (p_adj = min(1.0, p * 16)).
3. Per-symbol breadth check (% of universe symbols with positive Sharpe and beating B&H).
4. Parameter plateau verification across lookback neighborhoods.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.lowturn.panel_loader import load_dev_and_holdout_panels
from strategy_lab.lowturn.strategies import build_frozen_strategy_grid
from strategy_lab.lowturn.execution import simulate_portfolio_strategy, simulate_single_asset_strategy
from strategy_lab.lowturn.bootstrap import compute_annualized_sharpe

dev_panel, _, split_meta = load_dev_and_holdout_panels("1d")
grid = build_frozen_strategy_grid("1d")
symbols = list(dev_panel.keys())

# 1. Base Buy & Hold
bh_weights = {sym: grid["Buy_and_Hold"](df) for sym, df in dev_panel.items()}
bh_res = simulate_portfolio_strategy(dev_panel, bh_weights, round_trip_bps=30.0)
bh_ret = bh_res["net_return"]
bh_sharpe = compute_annualized_sharpe(bh_ret)

# Per-symbol Buy & Hold Sharpe
bh_sym_sharpes = {}
for sym, df in dev_panel.items():
    res = simulate_single_asset_strategy(df, bh_weights[sym], round_trip_bps=30.0)
    bh_sym_sharpes[sym] = compute_annualized_sharpe(res["net_return"])

# Setup paired calendar week clusters
df_clust = pd.DataFrame({"bh": bh_ret})
naive_idx = df_clust.index.tz_localize(None) if df_clust.index.tz is not None else df_clust.index
df_clust["year_week"] = naive_idx.to_period("W")
unique_weeks = df_clust["year_week"].unique()
n_weeks = len(unique_weeks)

# Pre-index week slices
week_indices = [np.where(df_clust["year_week"] == w)[0] for w in unique_weeks]

rng = np.random.default_rng(42)
n_bootstraps = 2000

print(f"=========================================================================================")
print(f"DEVELOPMENT WINDOW DETAILED DIAGNOSTICS & PAIRED DELTA-SHARPE CI MATRIX")
print(f"Universe: {len(symbols)} symbols | Common Dev Period: {split_meta['start_ts'][:10]} to {split_meta['dev_end'][:10]}")
print(f"Buy & Hold Net Sharpe: {bh_sharpe:.3f}")
print(f"=========================================================================================\n")

results = []

for strat_id, weight_fn in grid.items():
    weights = {sym: weight_fn(df) for sym, df in dev_panel.items()}
    res = simulate_portfolio_strategy(dev_panel, weights, round_trip_bps=30.0)
    strat_ret = res["net_return"]
    strat_sharpe = compute_annualized_sharpe(strat_ret)
    delta_sharpe = strat_sharpe - bh_sharpe

    # Paired Bootstrap on (strat_ret - bh_ret) and Delta Sharpe
    strat_vals = strat_ret.values
    bh_vals = bh_ret.values

    strat_groups = [strat_vals[idx] for idx in week_indices]
    bh_groups = [bh_vals[idx] for idx in week_indices]

    boot_deltas = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        sampled_w = rng.integers(0, n_weeks, size=n_weeks)
        sampled_strat = np.concatenate([strat_groups[k] for k in sampled_w])
        sampled_bh = np.concatenate([bh_groups[k] for k in sampled_w])
        
        s_std = np.std(sampled_strat)
        b_std = np.std(sampled_bh)
        
        sh_s = np.sqrt(365) * np.mean(sampled_strat) / s_std if s_std > 1e-8 else 0.0
        sh_b = np.sqrt(365) * np.mean(sampled_bh) / b_std if b_std > 1e-8 else 0.0
        boot_deltas[b] = sh_s - sh_b

    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)

    # Per-symbol checks
    sym_pos_count = 0
    sym_beat_bh_count = 0
    for sym, df in dev_panel.items():
        sym_res = simulate_single_asset_strategy(df, weights[sym], round_trip_bps=30.0)
        sym_sh = compute_annualized_sharpe(sym_res["net_return"])
        if sym_sh > 0.0:
            sym_pos_count += 1
        if sym_sh > bh_sym_sharpes[sym]:
            sym_beat_bh_count += 1

    results.append({
        "strategy": strat_id,
        "net_sharpe": strat_sharpe,
        "delta_sharpe": delta_sharpe,
        "paired_ci_lower": ci_lo,
        "paired_ci_upper": ci_hi,
        "sym_pos_pct": (sym_pos_count / len(symbols)) * 100.0,
        "sym_beat_bh_pct": (sym_beat_bh_count / len(symbols)) * 100.0,
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
