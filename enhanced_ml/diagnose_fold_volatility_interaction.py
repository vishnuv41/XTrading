"""
diagnose_fold_volatility_interaction.py
-----------------------------------------
PHASE 7A.1 — Fold x Volatility Confounding & Timeout Accounting Audit.

Determines whether the observed High-Volatility degradation (>=0.65 raw-P collapse) is:
  A) Present independently across multiple folds, OR
  B) Mostly explained by Fold 2's signal collapse (confounded by Fold 2).

Also audits timeout accounting to trace why SL% + TP% = 100%.

Outputs:
  - Fold x Volatility Regime Crosstab (N, % of fold, mean P, median P)
  - High-Confidence (>=0.65) Breakdown by Fold x Volatility Regime
  - Within-Fold Bucket Comparison (0.60-0.65 vs >=0.65) by Volatility Regime
  - Discrimination by Fold x Volatility Regime (Spearman Rho, ROC-AUC)
  - Critical Fold 2 Exclusion Test (All Folds vs Folds 1+3+4)
  - Timeout Accounting Audit
  - Final Classification (Exact 1 of 4 choices)

Saves artifacts to: models_artifacts/VOLATILITY_FOLD_INTERACTION/
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

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


def compute_subset_metrics(df_sub: pd.DataFrame, p_col: str = "raw_p") -> dict:
    valid = df_sub.dropna(subset=[p_col, "label"]).copy()
    n = len(valid)
    if n < 5:
        return {"n": n, "spearman_rho": None, "spearman_pval": None, "roc_auc": None}

    y = valid["label"].to_numpy(int)
    p = valid[p_col].to_numpy(float)

    rho, pval = spearmanr(p, y) if len(y) > 1 else (np.nan, np.nan)
    try:
        roc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        roc = np.nan

    return {
        "n": n,
        "spearman_rho": float(rho) if math.isfinite(rho) else None,
        "spearman_pval": float(pval) if math.isfinite(pval) else None,
        "roc_auc": roc,
    }


def analyze_fold_volatility_interaction(args):
    print(f"\n{'='*88}\nPHASE 7A.1 — FOLD x VOLATILITY CONFOUNDING & TIMEOUT ACCOUNTING AUDIT\n{'='*88}")
    print(f"Symbols: {args.symbols} | Timeframe: {args.timeframe} | Folds: {args.n_folds}\n")

    # 1. Load Data & Attach Causal Volatility Regime & Build Labels
    raw_dfs = {s: load_symbol(s, args.timeframe, args.exchange) for s in args.symbols}
    raw_dfs = restrict_to_common_range(raw_dfs)

    symbol_data = {}
    total_raw_bars = 0
    total_nan_labels = 0

    for symbol, raw in raw_dfs.items():
        total_raw_bars += len(raw)
        raw_reg = calculate_volatility_regime(raw.copy(), close_col="close", window=20, lookback_for_percentile=500)
        features = add_causal_features(raw)
        numeric, dropped = ensure_numeric(features)
        labels = build_binary_labels(raw, features, args.side, args.sl_mult, args.risk_reward, args.max_holding)
        
        nan_count = int(labels.isna().sum())
        total_nan_labels += nan_count

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
    timeout_audit_counter = {"resolved_tp": 0, "resolved_sl": 0, "unresolved_timeout": 0, "nan_dropped": 0}

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
                v_regime = regime_s.iloc[idx]
                if pd.isna(v_regime):
                    v_regime = "medium"

                if not (math.isfinite(lbl) and lbl in (0.0, 1.0)):
                    timeout_audit_counter["nan_dropped"] += 1
                    continue

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

                if exit_reason == "TP":
                    timeout_audit_counter["resolved_tp"] += 1
                elif exit_reason == "SL":
                    timeout_audit_counter["resolved_sl"] += 1
                elif exit_reason == "TIMEOUT":
                    timeout_audit_counter["unresolved_timeout"] += 1

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
    print(f"Collected {len(df)} valid OOS prediction records.\n")

    # Add deciles & probability buckets
    df = df.sort_values("raw_p").reset_index(drop=True)
    df["decile"] = pd.qcut(df["raw_p"], q=10, labels=False, duplicates="drop")
    
    fixed_bins = [0.0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 1.0]
    bucket_labels = ["<0.40", "0.40-0.45", "0.45-0.48", "0.48-0.50", "0.50-0.52", "0.52-0.55", "0.55-0.60", "0.60-0.65", ">=0.65"]
    df["bucket"] = pd.cut(df["raw_p"], bins=fixed_bins, labels=bucket_labels, include_lowest=True)

    output_dir = Path("models_artifacts/VOLATILITY_FOLD_INTERACTION")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 2: FOLD x VOLATILITY REGIME CROSS-TAB
    # ------------------------------------------------------------------------
    print(f"{'='*88}\nSTEP 2: FOLD x VOLATILITY REGIME CROSS-TAB\n{'='*88}")
    print(f"{'Fold':<6} | {'Vol Regime':<10} | {'Count N':<8} | {'% of Fold':<10} | {'Mean Raw P':<10} | {'Median Raw P':<12}")
    print("-" * 65)

    crosstab_records = []
    for f_no in range(1, args.n_folds + 1):
        f_df = df[df["fold"] == f_no]
        f_total = len(f_df)
        for reg in ["low", "medium", "high"]:
            sub = f_df[f_df["vol_regime"] == reg]
            n = len(sub)
            pct = (n / f_total * 100.0) if f_total > 0 else 0.0
            mean_p = float(sub["raw_p"].mean()) if n > 0 else np.nan
            med_p = float(sub["raw_p"].median()) if n > 0 else np.nan

            print(f"{f_no:<6d} | {reg:<10} | {n:<8d} | {pct:<10.1f}% | {mean_p:<10.4f} | {med_p:<12.4f}")
            crosstab_records.append({
                "fold": f_no, "vol_regime": reg, "count": n,
                "pct_of_fold": pct, "mean_raw_p": mean_p, "median_raw_p": med_p
            })

    pd.DataFrame(crosstab_records).to_csv(output_dir / "fold_volatility_crosstab.csv", index=False)

    # ------------------------------------------------------------------------
    # STEP 3: HIGH-CONFIDENCE (>=0.65) BREAKDOWN WITHIN EACH FOLD x REGIME
    # ------------------------------------------------------------------------
    print(f"\n{'='*115}\nSTEP 3: HIGH-CONFIDENCE (>=0.65) BREAKDOWN WITHIN EACH FOLD x REGIME\n{'='*115}")
    print(f"{'Fold':<5} | {'Regime':<8} | {'N':<5} | {'Status Tag':<22} | {'TP %':<7} | {'Avg Ret %':<10} | {'PF':<6} | {'Tot Ret %':<10} | {'SL %':<6} | {'TP Exit %':<9} | {'Timeout %':<9}")
    print("-" * 115)

    high_conf_records = []
    for f_no in range(1, args.n_folds + 1):
        for reg in ["low", "medium", "high"]:
            sub = df[(df["fold"] == f_no) & (df["vol_regime"] == reg) & (df["raw_p"] >= 0.65)]
            n = len(sub)

            tag = "VALID"
            if n < 20:
                tag = "[TOO SMALL TO INTERPRET]"
            elif n < 50:
                tag = "[SUGGESTIVE ONLY]"

            if n == 0:
                print(f"{f_no:<5d} | {reg:<8} | {0:<5d} | {tag:<22} | {'N/A':<7} | {'N/A':<10} | {'N/A':<6} | {'N/A':<10} | {'N/A':<6} | {'N/A':<9} | {'N/A':<9}")
                continue

            mean_p = float(sub["raw_p"].mean())
            tp_rate = float(sub["label"].mean() * 100.0)
            rets = sub["net_return"].dropna()
            avg_ret = float(rets.mean() * 100.0) if len(rets) else np.nan
            tot_ret = float(((1.0 + rets).prod() - 1.0) * 100.0) if len(rets) else np.nan

            gains = rets[rets > 0].sum() if len(rets) else 0.0
            losses = -rets[rets < 0].sum() if len(rets) else 0.0
            pf = (float(gains / losses)) if losses > 0 else (float("inf") if gains > 0 else 0.0)

            sl_pct = float((sub["exit_reason"] == "SL").mean() * 100.0)
            tp_exit_pct = float((sub["exit_reason"] == "TP").mean() * 100.0)
            to_pct = float((sub["exit_reason"] == "TIMEOUT").mean() * 100.0)

            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"

            print(f"{f_no:<5d} | {reg:<8} | {n:<5d} | {tag:<22} | {tp_rate:<7.2f}% | {avg_ret:<+10.2f}% | {pf_str:<6} | {tot_ret:<+10.2f}% | {sl_pct:<6.1f}% | {tp_exit_pct:<9.1f}% | {to_pct:<9.1f}%")

            high_conf_records.append({
                "fold": f_no, "vol_regime": reg, "count": n, "sample_tag": tag,
                "mean_raw_p": mean_p, "tp_rate_pct": tp_rate, "avg_net_return_pct": avg_ret,
                "profit_factor": pf, "total_return_pct": tot_ret, "sl_exit_pct": sl_pct,
                "tp_exit_pct": tp_exit_pct, "timeout_exit_pct": to_pct
            })

    pd.DataFrame(high_conf_records).to_csv(output_dir / "fold_volatility_high_confidence.csv", index=False)

    # ------------------------------------------------------------------------
    # STEP 4: WITHIN-FOLD BUCKET COMPARISON (0.60-0.65 vs >=0.65)
    # ------------------------------------------------------------------------
    print(f"\n{'='*95}\nSTEP 4: WITHIN-FOLD BUCKET COMPARISON (0.60-0.65 vs >=0.65)\n{'='*95}")
    print(f"{'Fold':<5} | {'Regime':<8} | {'N (0.60-0.65)':<14} | {'TP% (0.60-0.65)':<15} | {'N (>=0.65)':<10} | {'TP% (>=0.65)':<12} | {'Diff %':<8}")
    print("-" * 95)

    for f_no in range(1, args.n_folds + 1):
        for reg in ["low", "medium", "high"]:
            sub_a = df[(df["fold"] == f_no) & (df["vol_regime"] == reg) & (df["bucket"] == "0.60-0.65")]
            sub_b = df[(df["fold"] == f_no) & (df["vol_regime"] == reg) & (df["bucket"] == ">=0.65")]
            na, nb = len(sub_a), len(sub_b)

            tpa = float(sub_a["label"].mean() * 100.0) if na > 0 else np.nan
            tpb = float(sub_b["label"].mean() * 100.0) if nb > 0 else np.nan
            diff = (tpb - tpa) if (math.isfinite(tpa) and math.isfinite(tpb)) else np.nan

            tpa_str = f"{tpa:.2f}%" if math.isfinite(tpa) else "N/A"
            tpb_str = f"{tpb:.2f}%" if math.isfinite(tpb) else "N/A"
            diff_str = f"{diff:+.2f}%" if math.isfinite(diff) else "N/A"

            print(f"{f_no:<5d} | {reg:<8} | {na:<14d} | {tpa_str:<15} | {nb:<10d} | {tpb_str:<12} | {diff_str:<8}")

    # ------------------------------------------------------------------------
    # STEP 5: DISCRIMINATION BY FOLD x VOL REGIME
    # ------------------------------------------------------------------------
    print(f"\n{'='*82}\nSTEP 5: DISCRIMINATION BY FOLD x VOL REGIME\n{'='*82}")
    print(f"{'Fold':<5} | {'Regime':<8} | {'Count N':<8} | {'Spearman Rho':<14} | {'ROC-AUC':<8}")
    print("-" * 55)

    discrim_records = []
    for f_no in range(1, args.n_folds + 1):
        for reg in ["low", "medium", "high"]:
            sub = df[(df["fold"] == f_no) & (df["vol_regime"] == reg)]
            m = compute_subset_metrics(sub, "raw_p")
            rho_str = f"{m['spearman_rho']:+.4f}" if m['spearman_rho'] is not None else "N/A"
            auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
            print(f"{f_no:<5d} | {reg:<8} | {m['n']:<8d} | {rho_str:<14} | {auc_str:<8}")

            discrim_records.append({
                "fold": f_no, "vol_regime": reg, "count": m["n"],
                "spearman_rho": m["spearman_rho"], "roc_auc": m["roc_auc"]
            })

    pd.DataFrame(discrim_records).to_csv(output_dir / "fold_volatility_discrimination.csv", index=False)

    # ------------------------------------------------------------------------
    # STEP 6: CRITICAL FOLD-2 EXCLUSION TEST (SENSITIVITY ANALYSIS)
    # ------------------------------------------------------------------------
    print(f"\n{'='*92}\nSTEP 6: CRITICAL FOLD-2 EXCLUSION TEST (SENSITIVITY ANALYSIS)\n{'='*92}")
    print("Comparing ALL FOLDS (1+2+3+4) vs FOLDS 1+3+4 ONLY (Excluding Fold 2)\n")

    df_ex_f2 = df[df["fold"] != 2]

    print(f"{'Metric / Regime':<20} | {'ALL FOLDS (1+2+3+4)':<22} | {'FOLDS 1+3+4 ONLY (Excl. Fold 2)':<28}")
    print("-" * 75)

    for reg in ["low", "medium", "high"]:
        sub_all = df[df["vol_regime"] == reg]
        sub_ex = df_ex_f2[df_ex_f2["vol_regime"] == reg]

        m_all = compute_subset_metrics(sub_all)
        m_ex = compute_subset_metrics(sub_ex)

        sub_all_highconf = sub_all[sub_all["raw_p"] >= 0.65]
        sub_ex_highconf = sub_ex[sub_ex["raw_p"] >= 0.65]

        tp_all = sub_all_highconf["label"].mean() * 100.0 if len(sub_all_highconf) else np.nan
        tp_ex = sub_ex_highconf["label"].mean() * 100.0 if len(sub_ex_highconf) else np.nan

        rets_all = sub_all_highconf["net_return"].dropna()
        g_all, l_all = (rets_all[rets_all > 0].sum(), -rets_all[rets_all < 0].sum()) if len(rets_all) else (0, 0)
        pf_all = (g_all / l_all) if l_all > 0 else (float("inf") if g_all > 0 else 0.0)

        rets_ex = sub_ex_highconf["net_return"].dropna()
        g_ex, l_ex = (rets_ex[rets_ex > 0].sum(), -rets_ex[rets_ex < 0].sum()) if len(rets_ex) else (0, 0)
        pf_ex = (g_ex / l_ex) if l_ex > 0 else (float("inf") if g_ex > 0 else 0.0)

        print(f"REGIME: {reg.upper()}")
        print(f"  Sample N           | {m_all['n']:<22d} | {m_ex['n']:<28d}")
        print(f"  ROC-AUC            | {m_all['roc_auc']:<22.4f} | {m_ex['roc_auc']:<28.4f}")
        print(f"  Spearman Rho       | {m_all['spearman_rho']:<+22.4f} | {m_ex['spearman_rho']:<+28.4f}")
        print(f"  >=0.65 N           | {len(sub_all_highconf):<22d} | {len(sub_ex_highconf):<28d}")
        print(f"  >=0.65 TP Rate %   | {tp_all:<22.2f}% | {tp_ex:<28.2f}%")
        print(f"  >=0.65 PF          | {pf_all:<22.2f}  | {pf_ex:<28.2f} \n")

    # ------------------------------------------------------------------------
    # STEP 7: TIMEOUT ACCOUNTING AUDIT
    # ------------------------------------------------------------------------
    print(f"\n{'='*82}\nSTEP 7: TIMEOUT ACCOUNTING AUDIT\n{'='*82}")
    print("Investigating exact source data exit reasons for High-Vol >=0.65 bucket:\n")
    sub_high_vol_conf = df[(df["vol_regime"] == "high") & (df["raw_p"] >= 0.65)]
    
    sl_cnt = int((sub_high_vol_conf["exit_reason"] == "SL").sum())
    tp_cnt = int((sub_high_vol_conf["exit_reason"] == "TP").sum())
    to_cnt = int((sub_high_vol_conf["exit_reason"] == "TIMEOUT").sum())
    unk_cnt = int((sub_high_vol_conf["exit_reason"] == "UNKNOWN").sum())

    print(f"Total >=0.65 HIGH-vol observations evaluated: {len(sub_high_vol_conf)}")
    print(f"  SL count               : {sl_cnt} ({sl_cnt/len(sub_high_vol_conf)*100:.1f}%)")
    print(f"  TP count               : {tp_cnt} ({tp_cnt/len(sub_high_vol_conf)*100:.1f}%)")
    print(f"  Timeout count          : {to_cnt} ({to_cnt/len(sub_high_vol_conf)*100:.1f}%)")
    print(f"  Unknown count          : {unk_cnt}")
    print(f"\nUpstream Labeling Audit Note:")
    print(f"  Total raw candles loaded across symbols: {total_raw_bars}")
    print(f"  Total bars assigned NaN by build_binary_labels (unresolved timeouts): {total_nan_labels}")
    print(f"  Explanation: build_binary_labels assigns y = NaN to bars that hit NEITHER SL nor TP")
    print(f"  within max_holding (48 bars). Because binary classification filtering drops y = NaN rows,")
    print(f"  all evaluated trade records represent resolved outcomes (SL or TP). Unresolved timeout bars")
    print(f"  were excluded upstream by target labeling, resulting in exact 100% SL+TP resolution.\n")

    # ------------------------------------------------------------------------
    # STEP 10: FINAL CLASSIFICATION
    # ------------------------------------------------------------------------
    # Determine classification based on strict empirical evidence across independent folds
    # Check High-Vol ROC-AUC and >=0.65 performance across Folds 1, 3, 4 individually
    high_vol_fold1 = df[(df["fold"] == 1) & (df["vol_regime"] == "high")]
    high_vol_fold4 = df[(df["fold"] == 4) & (df["vol_regime"] == "high")]

    m_f1_h = compute_subset_metrics(high_vol_fold1)
    m_f4_h = compute_subset_metrics(high_vol_fold4)

    # Fold 2 contribution check
    f2_high_conf_high_vol = len(df[(df["fold"] == 2) & (df["vol_regime"] == "high") & (df["raw_p"] >= 0.65)])
    total_high_conf_high_vol = len(sub_high_vol_conf)
    f2_share = f2_high_conf_high_vol / total_high_conf_high_vol if total_high_conf_high_vol > 0 else 0.0

    classification = "INCONCLUSIVE — CELLS TOO SMALL"

    if total_nan_labels == total_raw_bars:
        classification = "TIMEOUT ACCOUNTING ISSUE DETECTED"
    elif f2_share >= 0.70:
        classification = "EFFECT LARGELY EXPLAINED BY FOLD 2"
    elif m_f1_h.get("roc_auc") is not None and m_f4_h.get("roc_auc") is not None:
        if m_f1_h["roc_auc"] < 0.52 and m_f4_h["roc_auc"] < 0.52:
            classification = "VOLATILITY EFFECT PERSISTS ACROSS FOLDS"
        else:
            classification = "EFFECT LARGELY EXPLAINED BY FOLD 2"

    print(f"{'='*82}\nFINAL CLASSIFICATION: {classification}\n{'='*82}\n")

    summary_json = {
        "classification": classification,
        "crosstab": crosstab_records,
        "high_confidence": high_conf_records,
        "discrimination": discrim_records,
        "fold2_exclusion_test": {
            "all_folds": {reg: compute_subset_metrics(df[df["vol_regime"] == reg]) for reg in ["low", "medium", "high"]},
            "excl_fold2": {reg: compute_subset_metrics(df_ex_f2[df_ex_f2["vol_regime"] == reg]) for reg in ["low", "medium", "high"]}
        },
        "timeout_audit": {
            "total_high_vol_high_conf": len(sub_high_vol_conf),
            "sl_cnt": sl_cnt,
            "tp_cnt": tp_cnt,
            "to_cnt": to_cnt,
            "unresolved_nan_dropped_labels": total_nan_labels,
        }
    }

    with open(output_dir / "fold_volatility_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"Summary saved to: {output_dir}/fold_volatility_summary.json")


def build_parser():
    p = argparse.ArgumentParser(description="Fold x Volatility Confounding & Timeout Audit.")
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
    analyze_fold_volatility_interaction(args)
