"""
raw_probability_decile_experiment.py
------------------------------------
Phase 6 Quantile/Decile Ranking Experiment.
Computes 10 raw-probability deciles for OOS predictions across:
  - Per Fold x Symbol
  - Per Fold (Pooled)
  - Per Symbol (Pooled)
  - Overall Pooled OOS

For each decile:
  - N
  - Mean raw probability
  - Median raw probability
  - TP / Win rate (%)
  - Average net return (%)
  - Profit Factor (PF)
  - Total return (%)

Summary ranking metrics:
  - Top 10% vs Bottom 10% (TP rate diff, return diff)
  - Top 20% vs Bottom 20% (TP rate diff, return diff)
  - Spearman rank correlation (P_raw vs True Label & P_raw vs Net Return)
  - ROC-AUC
  - PR-AUC
  - Brier score
  - Log loss
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


def compute_ranking_metrics(df: pd.DataFrame, p_col: str = "raw_p") -> dict:
    valid = df.dropna(subset=[p_col, "label"]).copy()
    if len(valid) == 0:
        return {}

    y_true = valid["label"].to_numpy(int)
    p_raw = valid[p_col].to_numpy(float)
    returns = valid["net_return"].dropna().to_numpy(float) if "net_return" in valid.columns else np.array([])

    # Spearman rank correlation
    rho_label, pval_label = spearmanr(p_raw, y_true) if len(y_true) > 1 else (np.nan, np.nan)
    rho_ret, pval_ret = (spearmanr(p_raw[:len(returns)], returns) if len(returns) > 1 else (np.nan, np.nan))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, p_raw)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        roc_auc = np.nan

    # PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, p_raw)
            pr_auc = float(auc(rec, prec))
        else:
            pr_auc = np.nan
    except Exception:
        pr_auc = np.nan

    # Brier & Log loss
    try:
        brier = float(brier_score_loss(y_true, p_raw))
        lloss = float(log_loss(y_true, np.column_stack([1.0 - p_raw, p_raw])))
    except Exception:
        brier, lloss = np.nan, np.nan

    return {
        "n_samples": int(len(valid)),
        "spearman_label_rho": float(rho_label) if math.isfinite(rho_label) else None,
        "spearman_label_pval": float(pval_label) if math.isfinite(pval_label) else None,
        "spearman_return_rho": float(rho_ret) if math.isfinite(rho_ret) else None,
        "spearman_return_pval": float(pval_ret) if math.isfinite(pval_ret) else None,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": lloss,
    }


def compute_decile_table(df: pd.DataFrame, p_col: str = "raw_p") -> list[dict]:
    valid = df.dropna(subset=[p_col, "label"]).copy()
    if len(valid) < 10:
        return []

    # Sort ascending by raw probability
    valid = valid.sort_values(p_col).reset_index(drop=True)
    
    # Assign 10 deciles (qcut with duplicates='drop')
    try:
        valid["decile"] = pd.qcut(valid[p_col], q=10, labels=False, duplicates="drop")
    except Exception:
        valid["decile"] = pd.cut(valid[p_col], bins=10, labels=False)

    decile_ranks = sorted(valid["decile"].unique())
    decile_labels = [
        "Bottom 10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "Top 10%"
    ]

    results = []
    for idx, d_idx in enumerate(decile_ranks):
        sub = valid[valid["decile"] == d_idx]
        n = len(sub)
        mean_p = float(sub[p_col].mean())
        median_p = float(sub[p_col].median())
        tp_rate = float(sub["label"].mean() * 100.0)
        
        rets = sub["net_return"].dropna()
        avg_ret = float(rets.mean() * 100.0) if len(rets) else np.nan
        tot_ret = float(((1.0 + rets).prod() - 1.0) * 100.0) if len(rets) else np.nan

        gains = rets[rets > 0].sum() if len(rets) else 0.0
        losses = -rets[rets < 0].sum() if len(rets) else 0.0
        pf = (float(gains / losses)) if losses > 0 else (float("inf") if gains > 0 else 0.0)

        label_name = decile_labels[idx] if idx < len(decile_labels) else f"Decile {d_idx+1}"

        results.append({
            "decile_num": int(idx + 1),
            "decile_name": label_name,
            "count": int(n),
            "mean_p": mean_p,
            "median_p": median_p,
            "tp_rate_pct": tp_rate,
            "avg_net_return_pct": avg_ret,
            "total_return_pct": tot_ret,
            "profit_factor": pf,
        })

    return results


def format_table(deciles: list[dict], title: str):
    print(f"\n{'='*82}\n{title}\n{'='*82}")
    print(f"{'Decile':<12} | {'Count':<6} | {'Mean P':<7} | {'Med P':<7} | {'TP Rate %':<10} | {'Avg Ret %':<10} | {'Tot Ret %':<10} | {'PF':<6}")
    print("-" * 82)
    for d in deciles:
        pf_str = f"{d['profit_factor']:.2f}" if d['profit_factor'] != float("inf") else "inf"
        print(f"{d['decile_name']:<12} | {d['count']:<6d} | {d['mean_p']:<7.4f} | {d['median_p']:<7.4f} | {d['tp_rate_pct']:<10.2f}% | {d['avg_net_return_pct']:<+10.2f}% | {d['total_return_pct']:<+10.2f}% | {pf_str:<6}")


def run_decile_experiment(args):
    print(f"\n{'='*82}\nPHASE 6 — RAW PROBABILITY QUANTILE / RANKING EXPERIMENT\n{'='*82}")
    print(f"Symbols: {args.symbols} | Timeframe: {args.timeframe} | Folds: {args.n_folds}\n")

    raw_dfs = {s: load_symbol(s, args.timeframe, args.exchange) for s in args.symbols}
    raw_dfs = restrict_to_common_range(raw_dfs)

    symbol_data = {}
    for symbol, raw in raw_dfs.items():
        features = add_causal_features(raw)
        numeric, dropped = ensure_numeric(features)
        labels = build_binary_labels(raw, features, args.side, args.sl_mult, args.risk_reward, args.max_holding)
        symbol_data[symbol] = {"raw": raw, "features": features, "numeric": numeric, "labels": labels}

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
            raw, features, numeric, labels = d["raw"], d["features"], d["numeric"], d["labels"]
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

                atr_val = finite_float(feature_test.iloc[local_i].get("ATR14", np.nan))
                net_ret = np.nan
                if math.isfinite(atr_val) and atr_val > 0 and idx + 1 < len(raw):
                    try:
                        trd = execute_trade(
                            raw, symbol, int(idx), args.side, atr_val, args.sl_mult,
                            args.risk_reward, args.max_holding, args.transaction_cost_bps,
                            p_raw_test[local_i], p_raw_test[local_i], 0.0, fold_no
                        )
                        net_ret = trd.net_return
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
                })

    oos_df = pd.DataFrame(oos_records)
    print(f"Collected {len(oos_df)} total OOS predictions across all folds and symbols.")

    # ------------------------------------------------------------------------
    # 1. Overall Pooled OOS Decile Table & Metrics
    # ------------------------------------------------------------------------
    pooled_deciles = compute_decile_table(oos_df, "raw_p")
    pooled_metrics = compute_ranking_metrics(oos_df, "raw_p")
    format_table(pooled_deciles, "OVERALL POOLED OOS RAW-PROBABILITY DECILES")

    # Top vs Bottom comparisons
    top10 = pooled_deciles[-1] if len(pooled_deciles) >= 10 else None
    bot10 = pooled_deciles[0] if len(pooled_deciles) >= 10 else None
    top20_tp = np.mean([d["tp_rate_pct"] for d in pooled_deciles[-2:]]) if len(pooled_deciles) >= 10 else None
    bot20_tp = np.mean([d["tp_rate_pct"] for d in pooled_deciles[:2]]) if len(pooled_deciles) >= 10 else None

    print("\n--- OVERALL POOLED RANKING METRICS ---")
    if top10 and bot10:
        print(f"Top 10% TP Rate ({top10['tp_rate_pct']:.2f}%) vs Bottom 10% TP Rate ({bot10['tp_rate_pct']:.2f}%): Diff = {top10['tp_rate_pct'] - bot10['tp_rate_pct']:+.2f}%")
        print(f"Top 10% Avg Ret ({top10['avg_net_return_pct']:+.2f}%) vs Bottom 10% Avg Ret ({bot10['avg_net_return_pct']:+.2f}%): Diff = {top10['avg_net_return_pct'] - bot10['avg_net_return_pct']:+.2f}%")
    if top20_tp and bot20_tp:
        print(f"Top 20% Avg TP Rate ({top20_tp:.2f}%) vs Bottom 20% Avg TP Rate ({bot20_tp:.2f}%): Diff = {top20_tp - bot20_tp:+.2f}%")
    print(f"Spearman Rho (Raw P vs Label): {pooled_metrics.get('spearman_label_rho'):+.4f} (p-val = {pooled_metrics.get('spearman_label_pval')})")
    print(f"Spearman Rho (Raw P vs Return): {pooled_metrics.get('spearman_return_rho'):+.4f} (p-val = {pooled_metrics.get('spearman_return_pval')})")
    print(f"ROC-AUC: {pooled_metrics.get('roc_auc'):.4f} | PR-AUC: {pooled_metrics.get('pr_auc'):.4f}")
    print(f"Brier Score: {pooled_metrics.get('brier_score'):.4f} | Log Loss: {pooled_metrics.get('log_loss'):.4f}")

    # ------------------------------------------------------------------------
    # 2. Per-Fold Breakdown
    # ------------------------------------------------------------------------
    per_fold_results = {}
    for f_no in range(1, args.n_folds + 1):
        f_df = oos_df[oos_df["fold"] == f_no]
        if len(f_df) == 0:
            continue
        f_deciles = compute_decile_table(f_df, "raw_p")
        f_metrics = compute_ranking_metrics(f_df, "raw_p")
        format_table(f_deciles, f"FOLD {f_no} POOLED RAW-PROBABILITY DECILES")
        print(f"Fold {f_no} Spearman Rho (Label): {f_metrics.get('spearman_label_rho'):+.4f} | ROC-AUC: {f_metrics.get('roc_auc'):.4f}")
        per_fold_results[f"fold_{f_no}"] = {"deciles": f_deciles, "metrics": f_metrics}

    # ------------------------------------------------------------------------
    # 3. Per-Symbol Breakdown
    # ------------------------------------------------------------------------
    per_symbol_results = {}
    for sym in args.symbols:
        s_df = oos_df[oos_df["symbol"] == sym]
        if len(s_df) == 0:
            continue
        s_deciles = compute_decile_table(s_df, "raw_p")
        s_metrics = compute_ranking_metrics(s_df, "raw_p")
        format_table(s_deciles, f"SYMBOL {sym} POOLED RAW-PROBABILITY DECILES")
        print(f"Symbol {sym} Spearman Rho (Label): {s_metrics.get('spearman_label_rho'):+.4f} | ROC-AUC: {s_metrics.get('roc_auc'):.4f}")
        per_symbol_results[sym] = {"deciles": s_deciles, "metrics": s_metrics}

    # Save artifact
    output_dir = Path("models_artifacts/DECILEN_RANKING_EXPERIMENT")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "decile_summary.json", "w") as f:
        json.dump({
            "pooled_deciles": pooled_deciles,
            "pooled_metrics": pooled_metrics,
            "per_fold": per_fold_results,
            "per_symbol": per_symbol_results,
        }, f, indent=2, default=str)

    print(f"\nDecile experiment summary saved to: {output_dir}/decile_summary.json")


def build_parser():
    p = argparse.ArgumentParser(description="Raw Probability Decile Ranking Experiment.")
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
    run_decile_experiment(args)
