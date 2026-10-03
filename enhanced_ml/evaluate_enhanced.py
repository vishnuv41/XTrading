"""
enhanced_ml/evaluate_enhanced.py
------------------------------------
The V2 analog of confidence_accuracy.py: how does the V2 (TP-before-SL,
MTF-featured) model's confidence relate to its actual binary outcome, on
the untouched final test window.

confidence_accuracy.py cannot be reused as-is: it decodes 3-class
BUY/SELL/HOLD labels via ml.train's LABEL_MAP, and rebuilds V1's base
features + V1's triple-barrier labels — none of which apply to a binary
TP-before-SL target with MTF features. Rather than bend that script to
cover both, this is a parallel, purpose-built script — same structure
and reasoning, different label/feature source (enhanced_ml.predict_enhanced
+ enhanced_ml.labeling.tp_before_sl instead of ml.predict + ml.train).

IMPORTANT — matching training params: --side / --sl-mult / --risk-reward
/ --max-holding / --higher-timeframes / --exchange here MUST match
exactly what was passed to enhanced_ml.train_enhanced when this model_dir
was trained (these aren't currently persisted in metadata.json — same
known gap noted in predict_enhanced.py). Defaults below match
train_enhanced.py's own CLI defaults, so a plain re-run with no extra
flags is correct as long as you didn't override anything at train time
either. If you did, pass the same overrides here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.predict import load_training_artifacts
from enhanced_ml.predict_enhanced import predict_enhanced
from enhanced_ml.train_enhanced import build_tp_before_sl_label_fn, default_v2_output_dir

DEFAULT_SYMBOL = settings.symbols.symbols[0]
DEFAULT_TIMEFRAME = "1h"


def parse_args():
    parser = argparse.ArgumentParser(description="Confidence vs actual outcome for a V2 TP-before-SL model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--exchange", default=settings.exchange.default_exchange)
    parser.add_argument("--model-dir", default=None,
                         help="Defaults to models_artifacts/<SYMBOL>_<timeframe>_v2")
    parser.add_argument("--higher-timeframes", nargs="+", default=["4h", "1d"])
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--risk-reward", type=float, default=2.0)
    parser.add_argument("--max-holding", type=int, default=48)
    return parser.parse_args()


LABEL_TO_NAME = {0: "SL_FIRST", 1: "TP_FIRST"}

CONFIDENCE_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    args = parse_args()
    model_dir = Path(args.model_dir or default_v2_output_dir(args.symbol, args.timeframe))

    if not model_dir.exists():
        print("ERROR: model directory not found:")
        print(model_dir)
        return 1

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        print("ERROR: metadata.json not found:")
        print(metadata_path)
        return 1

    import json
    with open(metadata_path) as f:
        metadata = json.load(f)

    print("=" * 80)
    print("V2 (TP-before-SL) CONFIDENCE vs ACTUAL OUTCOME")
    print("=" * 80)
    print(f"Symbol:            {args.symbol}")
    print(f"Timeframe:         {args.timeframe}")
    print(f"Model:             {model_dir}")
    print(f"Side:              {args.side}")
    print(f"SL mult / R:R:     {args.sl_mult} / {args.risk_reward}  "
          f"(production: {settings.risk.sl_atr_multiplier} / {settings.risk.tp_risk_reward} "
          f"{'-- MATCHES' if (args.sl_mult, args.risk_reward) == (settings.risk.sl_atr_multiplier, settings.risk.tp_risk_reward) else '-- DOES NOT MATCH, check your training params'})")
    print()

    print("Loading OHLCV...")
    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
    print(f"OHLCV rows: {len(df)}")

    print("Loading model artifacts...")
    artifacts = load_training_artifacts(str(model_dir))

    print("Rebuilding V2 predictions (MTF features)...")
    predictions = predict_enhanced(
        df,
        ensemble=artifacts["ensemble"],
        feature_columns=artifacts["feature_columns"],
        symbol=args.symbol,
        higher_timeframes=args.higher_timeframes,
        exchange=args.exchange,
        calibrator=None,  # raw confidence — same reasoning as confidence_accuracy.py
    )

    print("Rebuilding TP-before-SL labels...")
    label_fn = build_tp_before_sl_label_fn(
        side=args.side, sl_multiplier=args.sl_mult,
        risk_reward_ratio=args.risk_reward, max_holding=args.max_holding,
    )
    # label_fn needs the same MTF feature_df predict_enhanced already built
    # (touch_idx/label logic only reads OHLC+ATR columns, MTF columns are
    # irrelevant to it, but reusing predict_enhanced's rebuild keeps this
    # single-pass rather than recomputing features twice).
    from enhanced_ml.train_enhanced import build_mtf_feature_fn
    feat_df = build_mtf_feature_fn(args.symbol, args.higher_timeframes, exchange=args.exchange)(df, has_volume=True)
    labels = label_fn(feat_df)

    work = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "pred_label": predictions["pred_label"].values,
        "confidence": predictions["confidence"].values,
        "label": labels["label"].values,
    })
    work = work.dropna(subset=["pred_label", "confidence", "label"]).copy()
    work["pred_label"] = work["pred_label"].astype(int)
    work["label"] = work["label"].astype(int)

    final_test_start = metadata.get("final_test_start")
    final_test_end = metadata.get("final_test_end")
    if not final_test_start or not final_test_end:
        print("\nERROR: metadata.json does not contain final_test_start / final_test_end")
        return 1

    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    start = pd.to_datetime(final_test_start, utc=True)
    end = pd.to_datetime(final_test_end, utc=True)
    work = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)].copy()

    if work.empty:
        print("ERROR: final test window contains no usable rows.")
        return 1

    print()
    print("=" * 80)
    print("FINAL TEST WINDOW")
    print("=" * 80)
    print(f"Start: {start}")
    print(f"End:   {end}")
    print(f"Rows:  {len(work)}")

    y_true = work["label"]
    y_pred = work["pred_label"]
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    print()
    print("=" * 80)
    print("OVERALL FINAL TEST")
    print("=" * 80)
    print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Balanced accuracy: {balanced:.4f} ({balanced * 100:.2f}%)")

    print()
    print("=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)
    pred_dist = work["pred_label"].value_counts().sort_index()
    for cls, count in pred_dist.items():
        print(f"{LABEL_TO_NAME[cls]:10s} {count:5d} ({count / len(work) * 100:6.2f}%)")

    print()
    print("=" * 80)
    print("ACTUAL OUTCOME DISTRIBUTION")
    print("=" * 80)
    actual_dist = work["label"].value_counts().sort_index()
    for cls, count in actual_dist.items():
        print(f"{LABEL_TO_NAME[cls]:10s} {count:5d} ({count / len(work) * 100:6.2f}%)")

    print()
    print("=" * 80)
    print("RAW CONFIDENCE")
    print("=" * 80)
    print(work["confidence"].describe())

    print()
    print("=" * 100)
    print("ACCURACY BY CONFIDENCE BUCKET")
    print("=" * 100)
    bins = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
    work["confidence_bucket"] = pd.cut(work["confidence"], bins=bins, include_lowest=True)
    print(f"{'Confidence':<20}{'Samples':>10}{'Accuracy':>12}{'Balanced':>12}")
    print("-" * 100)
    for bucket, group in work.groupby("confidence_bucket", observed=True):
        if len(group) == 0:
            continue
        acc = accuracy_score(group["label"], group["pred_label"])
        bal = balanced_accuracy_score(group["label"], group["pred_label"]) if group["label"].nunique() > 1 else float("nan")
        print(f"{str(bucket):<20}{len(group):>10}{acc*100:>11.2f}%{bal*100:>11.2f}%")

    print()
    print("=" * 100)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 100)
    print(f"{'Threshold':<12}{'Samples':>10}{'% Kept':>10}{'Accuracy':>12}{'Balanced':>12}"
          f"{'TP_FIRST':>10}{'SL_FIRST':>10}")
    print("-" * 100)
    threshold_results = []
    for threshold in CONFIDENCE_THRESHOLDS:
        subset = work[work["confidence"] >= threshold]
        if len(subset) == 0:
            print(f"{threshold:<12.2f}{0:>10}{0:>9.2f}%{'N/A':>12}{'N/A':>12}{0:>10}{0:>10}")
            continue
        acc = accuracy_score(subset["label"], subset["pred_label"])
        bal = balanced_accuracy_score(subset["label"], subset["pred_label"]) if subset["label"].nunique() > 1 else float("nan")
        pct_kept = len(subset) / len(work) * 100
        n_tp = (subset["pred_label"] == 1).sum()
        n_sl = (subset["pred_label"] == 0).sum()
        print(f"{threshold:<12.2f}{len(subset):>10}{pct_kept:>9.2f}%{acc*100:>11.2f}%{bal*100:>11.2f}%{n_tp:>10}{n_sl:>10}")
        threshold_results.append({"threshold": threshold, "accuracy": acc, "balanced": bal, "n": len(subset)})

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    eligible = [r for r in threshold_results if r["n"] >= 100]
    if eligible:
        best = max(eligible, key=lambda r: r["accuracy"])
        print(f"Best accuracy threshold with >=100 samples: {best['threshold']:.2f}")
        print(f"Accuracy: {best['accuracy']*100:.2f}%")
        print(f"Samples: {best['n']} ({best['n']/len(work)*100:.2f}% of final test)")
    else:
        print("No confidence threshold retained at least 100 samples.")

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)
    print("This evaluates V2 against the SAME TP-before-SL configuration used to train it")
    print("(side/sl_mult/risk_reward/max_holding above) — re-check those match your actual")
    print("training run if this model_dir wasn't trained with train_enhanced.py's defaults.")
    print("Raw (uncalibrated) confidence is used — this model has no calibrator fit yet.")
    print("Do not select a threshold solely from this table; validate with an actual backtest next.")
    print("=" * 80)


if __name__ == "__main__":
    main()