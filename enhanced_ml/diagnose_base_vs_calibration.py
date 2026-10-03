"""
diagnose_base_vs_calibration.py
---------------------------------
Diagnostic script to separate base model collapse (Possibility 1) from
calibration layer destruction (Possibility 2) and evaluate probability bucket
reliability across walk-forward folds.

Outputs per fold:
  - XGBoost raw probability distribution (min, 25%, median, 75%, max, mean, std)
  - CatBoost raw probability distribution
  - Raw Blend probability distribution
  - Calibration parameters (slope/coef and intercept)
  - Calibrated probability distribution

Outputs pooled OOS:
  - Probability bucket vs Actual TP rate vs Expected return vs Profit Factor
    (for both Raw Blend probabilities and Calibrated probabilities)
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

warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to sys.path if needed
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
    fit_probability_calibrator,
    apply_probability_calibrator,
    make_calendar_folds,
    execute_trade,
    finite_float,
)


def get_stats(arr: np.ndarray) -> dict:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None, "std": None}
    return {
        "n": int(len(valid)),
        "min": float(np.min(valid)),
        "p25": float(np.percentile(valid, 25)),
        "median": float(np.median(valid)),
        "p75": float(np.percentile(valid, 75)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def analyze_probability_buckets(df: pd.DataFrame, prob_col: str, title: str):
    print(f"\n{'='*78}\nPROBABILITY BUCKET DIAGNOSTIC: {title} ({prob_col})\n{'='*78}")
    
    # Custom fixed bins
    fixed_bins = [0.0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 1.0]
    labels = ["<0.40", "0.40-0.45", "0.45-0.48", "0.48-0.50", "0.50-0.52", "0.52-0.55", "0.55-0.60", "0.60-0.65", ">=0.65"]
    
    df["bucket"] = pd.cut(df[prob_col], bins=fixed_bins, labels=labels, include_lowest=True)
    
    results = []
    print(f"{'Bucket':<12} | {'Count':<7} | {'Pct %':<6} | {'Mean P':<7} | {'Actual TP %':<11} | {'Avg Return':<10} | {'PF':<6}")
    print("-" * 75)
    
    for bucket in labels:
        sub = df[df["bucket"] == bucket]
        n = len(sub)
        pct = (n / len(df)) * 100.0 if len(df) > 0 else 0.0
        if n == 0:
            print(f"{bucket:<12} | {0:<7d} | {0.0:<6.1f} | {'N/A':<7} | {'N/A':<11} | {'N/A':<10} | {'N/A':<6}")
            continue
            
        mean_p = sub[prob_col].mean()
        actual_tp = sub["label"].mean() * 100.0  # label is 1 for TP, 0 for SL
        returns = sub["net_return"].dropna()
        avg_ret = returns.mean() * 100.0 if len(returns) > 0 else np.nan
        
        gains = returns[returns > 0].sum() if len(returns) else 0.0
        losses = -returns[returns < 0].sum() if len(returns) else 0.0
        pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"{bucket:<12} | {n:<7d} | {pct:<6.1f} | {mean_p:<7.4f} | {actual_tp:<11.2f}% | {avg_ret:<+10.2f}% | {pf_str:<6}")
        
        results.append({
            "bucket": bucket,
            "count": n,
            "pct": pct,
            "mean_p": mean_p,
            "actual_tp_rate": actual_tp,
            "avg_net_return": avg_ret,
            "profit_factor": pf,
        })
        
    return results


def run_diagnostic(args):
    print(f"\n{'='*78}\nBASE MODEL VS CALIBRATION DIAGNOSTIC EXPERIMENT\n{'='*78}")
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
    
    fold_diagnostics = []
    oos_records = []
    
    for fold_no, (train_start_ts, train_end_ts, test_start_ts, test_end_ts) in enumerate(folds, 1):
        print(f"\n{'='*78}\nDIAGNOSTIC FOLD {fold_no}\n{'='*78}")
        print(f"TRAIN: {train_start_ts} -> {train_end_ts}")
        print(f"TEST : {test_start_ts} -> {test_end_ts}")
        
        purge = max(1, args.max_holding)
        model_frames, cal_frames = [], []
        
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
                
            cal_size = max(args.min_calibration_bars, int(len(train_idx) * args.calibration_fraction))
            cal_size = min(cal_size, len(train_idx) - 100)
            if cal_size <= 0:
                continue
            model_end = len(train_idx) - cal_size
            model_idx, cal_idx = train_idx[:model_end], train_idx[model_end:]
            
            if labels.iloc[model_idx].nunique() < 2 or labels.iloc[cal_idx].nunique() < 2:
                continue
                
            m = numeric.iloc[model_idx].copy(); m["__label__"] = labels.iloc[model_idx].astype(int).to_numpy(); m["__symbol__"] = symbol
            c = numeric.iloc[cal_idx].copy(); c["__label__"] = labels.iloc[cal_idx].astype(int).to_numpy(); c["__symbol__"] = symbol
            model_frames.append(m)
            cal_frames.append(c)
            
        pooled_model = pd.concat(model_frames, ignore_index=True)
        pooled_cal = pd.concat(cal_frames, ignore_index=True)
        y_model = pooled_model.pop("__label__"); pooled_model.pop("__symbol__")
        y_cal = pooled_cal.pop("__label__"); pooled_cal.pop("__symbol__")
        
        columns = select_features_train_only(pooled_model, y_model, args.max_features, args.random_seed + fold_no)
        medians = fit_medians(pooled_model, columns)
        
        X_model = transform_matrix(pooled_model, columns, medians)
        X_cal = transform_matrix(pooled_cal, columns, medians)
        
        xgb, cat = train_base_models(X_model, y_model, args.random_seed + fold_no)
        
        # Inner calibration fit
        px_cal = np.asarray(xgb.predict_proba(X_cal)[:, 1], dtype=float)
        pc_cal = np.asarray(cat.predict_proba(X_cal)[:, 1], dtype=float)
        p_cal_raw = args.xgb_weight * px_cal + args.catboost_weight * pc_cal
        
        calibrator = fit_probability_calibrator(p_cal_raw, y_cal.to_numpy(int))
        
        calib_coef = float(calibrator.coef_[0][0]) if (calibrator is not None and hasattr(calibrator, "coef_")) else None
        calib_intercept = float(calibrator.intercept_[0]) if (calibrator is not None and hasattr(calibrator, "intercept_")) else None
        
        print(f"Calibration model parameters:")
        print(f"  Slope (coef)   : {calib_coef:.4f}" if calib_coef is not None else "  Calibrator is None")
        print(f"  Intercept      : {calib_intercept:.4f}" if calib_intercept is not None else "")
        
        # Collect OOS predictions across all symbols for this fold
        fold_oos_xgb, fold_oos_cat, fold_oos_raw, fold_oos_cal = [], [], [], []
        
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
            p_cal_test = apply_probability_calibrator(calibrator, p_raw_test)
            
            fold_oos_xgb.extend(px_test)
            fold_oos_cat.extend(pc_test)
            fold_oos_raw.extend(p_raw_test)
            fold_oos_cal.extend(p_cal_test)
            
            # Store row-level diagnostic record for valid test labels
            y_test = labels.iloc[test_idx].to_numpy(float)
            feature_test = features.iloc[test_idx].reset_index(drop=True)
            
            for local_i, idx in enumerate(test_idx):
                lbl = y_test[local_i]
                if not (math.isfinite(lbl) and lbl in (0.0, 1.0)):
                    continue
                    
                atr_val = finite_float(feature_test.iloc[local_i].get("ATR14", np.nan))
                
                # Compute hypothetical trade outcome net return if executed at this bar
                # (long side trade simulation)
                net_ret = np.nan
                if math.isfinite(atr_val) and atr_val > 0 and idx + 1 < len(raw):
                    try:
                        trd = execute_trade(
                            raw, symbol, int(idx), args.side, atr_val, args.sl_mult,
                            args.risk_reward, args.max_holding, args.transaction_cost_bps,
                            p_raw_test[local_i], p_cal_test[local_i], 0.0, fold_no
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
                    "cal_p": float(p_cal_test[local_i]),
                    "label": int(lbl),
                    "net_return": net_ret,
                })

        # Calculate fold probability statistics overall and per-symbol
        stats_xgb = get_stats(np.array(fold_oos_xgb))
        stats_cat = get_stats(np.array(fold_oos_cat))
        stats_raw = get_stats(np.array(fold_oos_raw))
        stats_cal = get_stats(np.array(fold_oos_cal))
        
        print("\n--- POOLED FOLD OOS PROBABILITY STATISTICS ---")
        print(f"Calibrator: Coef = {calib_coef:.4f} | Intercept = {calib_intercept:.4f}" if calib_coef is not None else "Calibrator: None")
        print(f"{'Model':<12} | {'Min':<6} | {'P25':<6} | {'Median':<6} | {'P75':<6} | {'Max':<6} | {'StdDev':<6}")
        print("-" * 60)
        for name, st in [("XGBoost", stats_xgb), ("CatBoost", stats_cat), ("Raw Blend", stats_raw), ("Calibrated", stats_cal)]:
            print(f"{name:<12} | {st['min']:<6.3f} | {st['p25']:<6.3f} | {st['median']:<6.3f} | {st['p75']:<6.3f} | {st['max']:<6.3f} | {st['std']:<6.3f}")
            
        per_symbol_stats = {}
        print("\n--- PER-SYMBOL OOS PROBABILITY BREAKDOWN ---")
        for sym in args.symbols:
            sym_records = [r for r in oos_records if r["fold"] == fold_no and r["symbol"] == sym]
            if not sym_records:
                continue
            sxgb = get_stats(np.array([r["xgb_p"] for r in sym_records]))
            scat = get_stats(np.array([r["cat_p"] for r in sym_records]))
            sraw = get_stats(np.array([r["raw_p"] for r in sym_records]))
            scal = get_stats(np.array([r["cal_p"] for r in sym_records]))
            
            per_symbol_stats[sym] = {"xgb": sxgb, "cat": scat, "raw": sraw, "cal": scal}
            print(f"\nSymbol: {sym}")
            print(f"  XGBoost   : min={sxgb['min']:.3f} / p25={sxgb['p25']:.3f} / median={sxgb['median']:.3f} / p75={sxgb['p75']:.3f} / max={sxgb['max']:.3f} / std={sxgb['std']:.3f}")
            print(f"  CatBoost  : min={scat['min']:.3f} / p25={scat['p25']:.3f} / median={scat['median']:.3f} / p75={scat['p75']:.3f} / max={scat['max']:.3f} / std={scat['std']:.3f}")
            print(f"  Raw Blend : min={sraw['min']:.3f} / p25={sraw['p25']:.3f} / median={sraw['median']:.3f} / p75={sraw['p75']:.3f} / max={sraw['max']:.3f} / std={sraw['std']:.3f}")
            print(f"  Calibrated: min={scal['min']:.3f} / p25={scal['p25']:.3f} / median={scal['median']:.3f} / p75={scal['p75']:.3f} / max={scal['max']:.3f} / std={scal['std']:.3f}")

        fold_diag = {
            "fold": fold_no,
            "calib_coef": calib_coef,
            "calib_intercept": calib_intercept,
            "xgb_stats": stats_xgb,
            "cat_stats": stats_cat,
            "raw_stats": stats_raw,
            "cal_stats": stats_cal,
            "per_symbol_stats": per_symbol_stats,
        }
        fold_diagnostics.append(fold_diag)

    oos_df = pd.DataFrame(oos_records)
    print(f"\nTotal OOS labeled samples collected across all folds: {len(oos_df)}")
    
    # Bucket Analysis 1: Raw Blend Probabilities
    raw_bucket_results = analyze_probability_buckets(oos_df, "raw_p", "RAW BLEND PROBABILITIES (XGB + CatBoost)")
    
    # Bucket Analysis 2: Calibrated Probabilities
    cal_bucket_results = analyze_probability_buckets(oos_df, "cal_p", "CALIBRATED PROBABILITIES (Logistic Platt)")

    # Save artifact
    output_dir = Path("models_artifacts/DIAGNOSTIC_BASE_VS_CALIBRATION")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    oos_df.to_csv(output_dir / "oos_predictions.csv", index=False)
    
    with open(output_dir / "diagnostic_summary.json", "w") as f:
        json.dump({
            "fold_diagnostics": fold_diagnostics,
            "raw_bucket_results": raw_bucket_results,
            "cal_bucket_results": cal_bucket_results,
        }, f, indent=2, default=str)
        
    print(f"\nDiagnostic results saved to: {output_dir}/")


def build_parser():
    p = argparse.ArgumentParser(description="Base model vs Calibration diagnostic script.")
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
    p.add_argument("--calibration-fraction", type=float, default=0.20)
    p.add_argument("--min-calibration-bars", type=int, default=400)
    p.add_argument("--transaction-cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_diagnostic(args)
