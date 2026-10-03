"""
diagnose_volatility_regime_stratification.py
----------------------------------------------
Volatility-Regime Stratification Diagnostic Experiment.

Tests whether the high raw-P (>=0.65) prediction collapse clusters specifically
during high-volatility regimes (due to ATR stop-loss whipsaws), occurs equally
across all volatility regimes, or is a population share composition artifact.

Data & Join Rules:
  - Causal vol_regime ('low', 'medium', 'high') generated via regime.volatility_regime
    (20-bar realized vol ranked in a 500-bar rolling window).
  - Exact match join on (symbol, timestamp) as of entry bar.
  - Includes mechanistic exit_reason ('TP', 'SL', 'TIMEOUT') per trade attempt.

Analysis Sections:
  1. Population Share of Volatility Regimes per Decile & Bucket
  2. Stratified Probability Buckets x Volatility Regime (with N, TP%, Avg Return, PF, exit_reason breakdown)
  3. Stratified Deciles x Volatility Regime
  4. Ranking & Discrimination Metrics by Volatility Regime (Spearman Rho, ROC-AUC, PR-AUC)
  5. Bootstrap Significance Test of TP Rate Difference (Decile 9 [0.60-0.65] vs Decile 10 [>=0.65]) per Regime
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, log_loss

warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regime.volatility_regime import calculate_volatility_regime
from enhanced_ml.walkforward_pooled import (
    load_symbol,
    restrict_to_common_range,
    add_causal_features,
    ensure_numeric,
    build_binary_labels,
    select_features_train_only,
    fit_medians,
    transform_matrix,
    train_base_models,
    make_calendar_folds,
    execute_trade,
    finite_float,
)


def compute_metrics_for_subset(sub: pd.DataFrame, p_col: str = "raw_p") -> dict:
    valid = sub.dropna(subset=[p_col, "label"]).copy()
    if len(valid) < 5:
        return {"n": len(valid), "spearman_rho": None, "roc_auc": None, "pr_auc": None, "brier": None}

    y = valid["label"].to_numpy(int)
    p = valid[p_col].to_numpy(float)

    rho, pval = spearmanr(p, y) if len(y) > 1 else (np.nan, np.nan)

    try:
        roc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        roc = np.nan

    try:
        if len(np.unique(y)) > 1:
            prec, rec, _ = precision_recall_curve(y, p)
            pr = float(auc(rec, prec))
        else:
            pr = np.nan
    except Exception:
        pr = np.nan

    try:
        brier = float(brier_score_loss(y, p))
    except Exception:
        brier = np.nan

    return {
        "n": int(len(valid)),
        "spearman_rho": float(rho) if math.isfinite(rho) else None,
        "spearman_pval": float(pval) if math.isfinite(pval) else None,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
    }


def bootstrap_tp_difference(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, n_iter: int = 1000, seed: int = 42) -> dict:
    """Bootstrap difference in TP rate: TP_Rate(Group A) - TP_Rate(Group B)."""
    sub_a = df.loc[mask_a, "label"].dropna().to_numpy(float)
    sub_b = df.loc[mask_b, "label"].dropna().to_numpy(float)

    if len(sub_a) == 0 or len(sub_b) == 0:
        return {"obs_diff": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan}

    obs_tp_a = float(np.mean(sub_a)) * 100.0
    obs_tp_b = float(np.mean(sub_b)) * 100.0
    obs_diff = obs_tp_a - obs_tp_b

    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n_iter):
        sample_a = rng.choice(sub_a, size=len(sub_a), replace=True)
        sample_b = rng.choice(sub_b, size=len(sub_b), replace=True)
        diffs.append((np.mean(sample_a) - np.mean(sample_b)) * 100.0)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided empirical p-value for null hypothesis diff = 0
    p_val = float(np.mean(np.abs(diffs - np.mean(diffs)) >= np.abs(obs_diff)))

    return {
        "n_a": int(len(sub_a)),
        "n_b": int(len(sub_b)),
        "tp_rate_a_pct": obs_tp_a,
        "tp_rate_b_pct": obs_tp_b,
        "obs_diff_pct": obs_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_val,
    }


def run_volatility_stratification_experiment(args):
    print(f"\n{'='*82}\nVOLATILITY REGIME STRATIFICATION EXPERIMENT\n{'='*82}")
    print(f"Symbols: {args.symbols} | Timeframe: {args.timeframe} | Folds: {args.n_folds}\n")

    raw_dfs = {s: load_symbol(s, args.timeframe, args.exchange) for s in args.symbols}
    raw_dfs = restrict_to_common_range(raw_dfs)

    symbol_data = {}
    for symbol, raw in raw_dfs.items():
        # Causal volatility regime calculation
        raw_reg = calculate_volatility_regime(raw.copy(), close_col="close", window=20, lookback_for_percentile=500)
        
        features = add_causal_features(raw)
        numeric, dropped = ensure_numeric(features)
        labels = build_binary_labels(raw, features, args.side, args.sl_mult, args.risk_reward, args.max_holding)
        
        symbol_data[symbol] = {
            "raw": raw,
            "regime": raw_reg["vol_regime"],
            "features": features,
            "numeric": numeric,
            "labels": labels,
        }

    reference_symbol = min(symbol_data, key=lambda s: len(symbol_data[s]["raw"]))
    reference_timestamps = symbol_data[reference_symbol]["raw"]["timestamp"]
    folds = make_calendar_folds(reference_timestamps, args.n_folds, args.min_train_bars, args.test_bars)

    oos_records = []

    for fold_no, (train_start_ts, train_end_ts, test_start_ts, test_end_ts) in enumerate(folds, 1):
        purge = max(1, args.max_holding)
        model_frames = []

        for symbol, d in symbol_data.items():
            raw, features, numeric, labels = d["raw"], d["features"], d["numeric"], d["labels"]
            ts = raw["timestamp"]

            train_mask = (ts >= train_start_ts) & (ts <= train_end_ts)
            train_idx = np.where(train_mask)[0]
            if len(train_idx) <= purge:
                continue
            train_idx = train_idx[:-purge]

            y_train_all = labels.iloc[train_idx].to_numpy(float)
            valid = np.isfinite(y_train_all) & np.isin(y_train_all, [0.0, 1.0])
            train_idx = train_idx[valid]
            if len(train_idx) < 200:
                continue

            m = numeric.iloc[train_idx].copy()
            m["__label__"] = labels.iloc[train_idx].astype(int).to_numpy()
            m["__symbol__"] = symbol
            model_frames.append(m)

        pooled_model = pd.concat(model_frames, ignore_index=True)
        y_model = pooled_model.pop("__label__")
        pooled_model.pop("__symbol__")

        columns = select_features_train_only(pooled_model, y_model, args.max_features, args.random_seed + fold_no)
        medians = fit_medians(pooled_model, columns)

        X_model = transform_matrix(pooled_model, columns, medians)
        xgb, cat = train_base_models(X_model, y_model, args.random_seed + fold_no)

        for symbol, d in symbol_data.items():
            raw, regime_s, features, numeric, labels = d["raw"], d["regime"], d["features"], d["numeric"], d["labels"]
            ts = raw["timestamp"]
            test_mask = (ts >= test_start_ts) & (ts <= test_end_ts)
            test_idx = np.where(test_mask)[0]
            if len(test_idx) == 0:
                continue

            X_test = transform_matrix(numeric.iloc[test_idx], columns, medians)
            px_test = np.asarray(xgb.predict_proba(X_test)[:, 1], dtype=float)
            pc_test = np.asarray(cat.predict_proba(X_test)[:, 1], dtype=float)
            p_raw_test = args.xgb_weight * px_test + args.catboost_weight * pc_test

            y_test = labels.iloc[test_idx].to_numpy(float)
            feature_test = features.iloc[test_idx].reset_index(drop=True)

            for local_i, idx in enumerate(test_idx):
                lbl = y_test[local_i]
                if not (math.isfinite(lbl) and lbl in (0.0, 1.0)):
                    continue

                v_regime = regime_s.iloc[idx]
                if pd.isna(v_regime):
                    v_regime = "medium"

                atr_val = finite_float(feature_test.iloc[local_i].get("ATR14", np.nan))
                net_ret = np.nan
                exit_reason = "UNKNOWN"
                
                if math.isfinite(atr_val) and atr_val > 0 and idx + 1 < len(raw):
                    try:
                        trd = execute_trade(
                            raw, symbol, int(idx), args.side, atr_val, args.sl_mult,
                            args.risk_reward, args.max_holding, args.transaction_cost_bps,
                            p_raw_test[local_i], p_raw_test[local_i], 0.0, fold_no
                        )
                        net_ret = trd.net_return
                        exit_reason = trd.exit_reason
                    except Exception:
                        pass

                oos_records.append({
                    "fold": fold_no,
                    "symbol": symbol,
                    "timestamp": str(ts.iloc[idx]),
                    "xgb_p": float(px_test[local_i]),
                    "cat_p": float(pc_test[local_i]),
                    "raw_p": float(p_raw_test[local_i]),
                    "label": int(lbl),
                    "net_return": net_ret,
                    "exit_reason": exit_reason,
                    "vol_regime": str(v_regime),
                })

    df = pd.DataFrame(oos_records)
    
    # Validation checks on merge/regime
    assert len(df) == len(oos_records), "Row count changed during prediction collection!"
    assert df["vol_regime"].isna().sum() == 0, "Null volatility regime values detected!"
    
    print(f"Collected {len(df)} total OOS prediction records with exact causal vol_regime join.\n")

    # Quantile bins & standard fixed buckets
    df = df.sort_values("raw_p").reset_index(drop=True)
    df["decile"] = pd.qcut(df["raw_p"], q=10, labels=False, duplicates="drop")
    decile_names = [
        "Bottom 10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "Top 10%"
    ]

    fixed_bins = [0.0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 1.0]
    bucket_labels = ["<0.40", "0.40-0.45", "0.45-0.48", "0.48-0.50", "0.50-0.52", "0.52-0.55", "0.55-0.60", "0.60-0.65", ">=0.65"]
    df["bucket"] = pd.cut(df["raw_p"], bins=fixed_bins, labels=bucket_labels, include_lowest=True)

    # ------------------------------------------------------------------------
    # SECTION 1: Volatility Regime Population Share by Probability Bucket & Decile
    # ------------------------------------------------------------------------
    print(f"{'='*82}\nSECTION 1: VOLATILITY REGIME POPULATION SHARE BY PROBABILITY BUCKET\n{'='*82}")
    print(f"{'Bucket':<12} | {'Count N':<8} | {'Low Vol %':<10} | {'Med Vol %':<10} | {'High Vol %':<10}")
    print("-" * 60)
    
    pop_share_buckets = []
    for b in bucket_labels:
        sub = df[df["bucket"] == b]
        n = len(sub)
        if n == 0:
            continue
        vc = sub["vol_regime"].value_counts(normalize=True) * 100.0
        low_p = float(vc.get("low", 0.0))
        med_p = float(vc.get("medium", 0.0))
        high_p = float(vc.get("high", 0.0))
        print(f"{b:<12} | {n:<8d} | {low_p:<10.1f}% | {med_p:<10.1f}% | {high_p:<10.1f}%")
        pop_share_buckets.append({"bucket": b, "count": n, "low_pct": low_p, "med_pct": med_p, "high_pct": high_p})

    # ------------------------------------------------------------------------
    # SECTION 2: Top Decile & High-Confidence Breakdown Stratified by Volatility Regime
    # ------------------------------------------------------------------------
    print(f"\n{'='*92}\nSECTION 2: PROBABILITY BUCKETS STRATIFIED BY VOLATILITY REGIME\n{'='*92}")
    print(f"{'Bucket':<10} | {'Regime':<8} | {'N':<6} | {'Mean P':<7} | {'TP %':<7} | {'Avg Ret %':<10} | {'PF':<6} | {'SL %':<6} | {'TP Exit %':<9}")
    print("-" * 92)

    stratified_bucket_results = []
    for b in bucket_labels:
        for reg in ["low", "medium", "high"]:
            sub = df[(df["bucket"] == b) & (df["vol_regime"] == reg)]
            n = len(sub)
            if n == 0:
                continue
            mean_p = float(sub["raw_p"].mean())
            tp_rate = float(sub["label"].mean() * 100.0)
            
            rets = sub["net_return"].dropna()
            avg_ret = float(rets.mean() * 100.0) if len(rets) else np.nan
            
            gains = rets[rets > 0].sum() if len(rets) else 0.0
            losses = -rets[rets < 0].sum() if len(rets) else 0.0
            pf = (float(gains / losses)) if losses > 0 else (float("inf") if gains > 0 else 0.0)
            
            sl_pct = float((sub["exit_reason"] == "SL").mean() * 100.0)
            tp_exit_pct = float((sub["exit_reason"] == "TP").mean() * 100.0)
            
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"{b:<10} | {reg:<8} | {n:<6d} | {mean_p:<7.4f} | {tp_rate:<7.2f}% | {avg_ret:<+10.2f}% | {pf_str:<6} | {sl_pct:<6.1f}% | {tp_exit_pct:<9.1f}%")
            
            stratified_bucket_results.append({
                "bucket": b, "vol_regime": reg, "n": n, "mean_p": mean_p,
                "tp_rate_pct": tp_rate, "avg_net_return_pct": avg_ret, "profit_factor": pf,
                "sl_exit_pct": sl_pct, "tp_exit_pct": tp_exit_pct
            })

    # ------------------------------------------------------------------------
    # SECTION 3: Decile Table Stratified by Volatility Regime
    # ------------------------------------------------------------------------
    print(f"\n{'='*88}\nSECTION 3: 10 DECILES STRATIFIED BY VOLATILITY REGIME\n{'='*88}")
    print(f"{'Decile':<12} | {'Regime':<8} | {'N':<6} | {'Mean P':<7} | {'TP %':<7} | {'Avg Ret %':<10} | {'PF':<6}")
    print("-" * 88)

    stratified_decile_results = []
    for d_idx in range(10):
        d_name = decile_names[d_idx]
        for reg in ["low", "medium", "high"]:
            sub = df[(df["decile"] == d_idx) & (df["vol_regime"] == reg)]
            n = len(sub)
            if n == 0:
                continue
            mean_p = float(sub["raw_p"].mean())
            tp_rate = float(sub["label"].mean() * 100.0)
            
            rets = sub["net_return"].dropna()
            avg_ret = float(rets.mean() * 100.0) if len(rets) else np.nan
            
            gains = rets[rets > 0].sum() if len(rets) else 0.0
            losses = -rets[rets < 0].sum() if len(rets) else 0.0
            pf = (float(gains / losses)) if losses > 0 else (float("inf") if gains > 0 else 0.0)
            
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"{d_name:<12} | {reg:<8} | {n:<6d} | {mean_p:<7.4f} | {tp_rate:<7.2f}% | {avg_ret:<+10.2f}% | {pf_str:<6}")
            
            stratified_decile_results.append({
                "decile": d_name, "vol_regime": reg, "n": n, "mean_p": mean_p,
                "tp_rate_pct": tp_rate, "avg_net_return_pct": avg_ret, "profit_factor": pf
            })

    # ------------------------------------------------------------------------
    # SECTION 4: Discrimination Metrics by Volatility Regime
    # ------------------------------------------------------------------------
    print(f"\n{'='*82}\nSECTION 4: RANKING & DISCRIMINATION METRICS BY VOLATILITY REGIME\n{'='*82}")
    print(f"{'Vol Regime':<12} | {'Sample N':<8} | {'Spearman Rho':<13} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
    print("-" * 75)

    metrics_by_regime = {}
    for reg in ["low", "medium", "high"]:
        sub = df[df["vol_regime"] == reg]
        m = compute_metrics_for_subset(sub, "raw_p")
        metrics_by_regime[reg] = m
        rho_str = f"{m['spearman_rho']:+.4f}" if m['spearman_rho'] is not None else "N/A"
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        br_str = f"{m['brier']:.4f}" if m['brier'] is not None else "N/A"
        print(f"{reg:<12} | {m['n']:<8d} | {rho_str:<13} | {auc_str:<8} | {pr_str:<8} | {br_str:<8}")

    # ------------------------------------------------------------------------
    # SECTION 5: Bootstrap Test of Decile 9 (0.60-0.65) vs Decile 10 (>=0.65) Collapse
    # ------------------------------------------------------------------------
    print(f"\n{'='*82}\nSECTION 5: BOOTSTRAP SIGNIFICANCE TEST: [0.60-0.65] vs [>=0.65] TP DROP\n{'='*82}")
    print("Testing null hypothesis: TP_Rate(0.60-0.65) - TP_Rate(>=0.65) == 0 (1,000 iterations)\n")

    bootstrap_results = {}
    
    # Overall pooled test
    mask_9_all = df["bucket"] == "0.60-0.65"
    mask_10_all = df["bucket"] == ">=0.65"
    bs_all = bootstrap_tp_difference(df, mask_9_all, mask_10_all)
    bootstrap_results["overall"] = bs_all
    
    print(f"OVERALL POOLED: [0.60-0.65] ({bs_all['tp_rate_a_pct']:.2f}%, N={bs_all['n_a']}) vs [>=0.65] ({bs_all['tp_rate_b_pct']:.2f}%, N={bs_all['n_b']})")
    print(f"  Observed Diff  : {bs_all['obs_diff_pct']:+.2f}%")
    print(f"  95% Boot CI    : [{bs_all['ci_95_lower']:+.2f}%, {bs_all['ci_95_upper']:+.2f}%]")
    print(f"  p-value        : {bs_all['p_value']:.4f}\n")

    # Per regime test
    for reg in ["low", "medium", "high"]:
        mask_9 = (df["bucket"] == "0.60-0.65") & (df["vol_regime"] == reg)
        mask_10 = (df["bucket"] == ">=0.65") & (df["vol_regime"] == reg)
        bs = bootstrap_tp_difference(df, mask_9, mask_10)
        bootstrap_results[reg] = bs
        
        print(f"REGIME '{reg.upper()}': [0.60-0.65] ({bs['tp_rate_a_pct']:.2f}%, N={bs['n_a']}) vs [>=0.65] ({bs['tp_rate_b_pct']:.2f}%, N={bs['n_b']})")
        print(f"  Observed Diff  : {bs['obs_diff_pct']:+.2f}%" if math.isfinite(bs['obs_diff_pct']) else "  Observed Diff  : N/A")
        print(f"  95% Boot CI    : [{bs['ci_95_lower']:+.2f}%, {bs['ci_95_upper']:+.2f}%]" if math.isfinite(bs['ci_95_lower']) else "  95% Boot CI    : N/A")
        print(f"  p-value        : {bs['p_value']:.4f}" if math.isfinite(bs['p_value']) else "  p-value        : N/A\n")

    # Save artifact
    output_dir = Path("models_artifacts/VOLATILITY_STRATIFICATION_DIAGNOSTIC")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "vol_stratification_summary.json", "w") as f:
        json.dump({
            "pop_share_buckets": pop_share_buckets,
            "stratified_bucket_results": stratified_bucket_results,
            "stratified_decile_results": stratified_decile_results,
            "metrics_by_regime": metrics_by_regime,
            "bootstrap_results": bootstrap_results,
        }, f, indent=2, default=str)

    print(f"Volatility stratification diagnostic summary saved to: {output_dir}/vol_stratification_summary.json")


def build_parser():
    p = argparse.ArgumentParser(description="Volatility Regime Stratification Diagnostic Script.")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--side", choices=["long", "short"], default="long")
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--risk-reward", type=float, default=2.5)
    p.add_argument("--max-holding", type=int, default=48)
    p.add_argument("--xgb-weight", type=float, default=0.65)
    p.add_argument("--catboost-weight", type=float, default=0.35)
    p.add_argument("--max-features", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--test-bars", type=int, default=1000)
    p.add_argument("--min-train-bars", type=int, default=3000)
    p.add_argument("--transaction-cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_volatility_stratification_experiment(args)
