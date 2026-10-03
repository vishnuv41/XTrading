"""
confidence_accuracy.py

Analyze how prediction confidence relates to actual
triple-barrier outcomes on the untouched final test set.

Uses the EXACT labeling configuration from ml.train.build_labels():

    pt_mult      = 2.0
    sl_mult      = 2.0
    max_holding  = 20
    volatility   = ATR14 / close

Usage:

    python confidence_accuracy.py

Or:

    python confidence_accuracy.py --symbol BTC/USDT --timeframe 1h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pipeline.data_loader import load_ohlcv
from ml.predict import load_training_artifacts, predict
from ml.train import build_labels, LABEL_MAP


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"

DEFAULT_MODEL_DIR = Path("models_artifacts/BTCUSDT_1h")

CONFIDENCE_THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
    )

    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    return parser.parse_args()


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_TO_NAME = {
    -1: "SELL",
    0: "HOLD",
    1: "BUY",
}


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"ERROR: model directory not found:")
        print(model_dir)
        return 1

    metadata_path = model_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found:")
        print(metadata_path)
        return 1

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 80)
    print("CONFIDENCE vs ACTUAL TRIPLE-BARRIER OUTCOME")
    print("=" * 80)

    print(f"Symbol:      {args.symbol}")
    print(f"Timeframe:   {args.timeframe}")
    print(f"Model:       {model_dir}")
    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading OHLCV...")

    df = load_ohlcv(
        args.symbol,
        args.timeframe,
    )

    print(f"OHLCV rows: {len(df)}")

    # ========================================================
    # REBUILD FEATURES
    # ========================================================

    print("Loading model artifacts...")

    artifacts = load_training_artifacts(
        str(model_dir)
    )

    print("Rebuilding predictions...")

    # IMPORTANT:
    # We deliberately use calibrator=None here.
    #
    # This analyzes RAW ensemble confidence, because your
    # calibration experiments showed isotonic/sigmoid were
    # destroying HOLD predictions.
    predictions = predict(
        df=df,
        ensemble=artifacts["ensemble"],
        feature_columns=artifacts["feature_columns"],
        calibrator=None,
    )

    # ========================================================
    # REBUILD EXACT TRAINING LABELS
    # ========================================================

    print("Rebuilding triple-barrier labels...")

    labels = build_labels(
        df,
        pt_mult=2.0,
        sl_mult=2.0,
        max_holding=20,
    )

    # ========================================================
    # ALIGN DATA
    # ========================================================

    work = pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "pred_label": predictions["pred_label"].values,
            "confidence": predictions["confidence"].values,
            "label": labels["label"].values,
            "touch_idx": labels["touch_idx"].values,
        }
    )

    # Remove rows where prediction or label isn't available
    work = work.dropna(
        subset=[
            "pred_label",
            "confidence",
            "label",
        ]
    ).copy()

    work["pred_label"] = work["pred_label"].astype(int)
    work["label"] = work["label"].astype(int)

    # ========================================================
    # FINAL TEST WINDOW
    # ========================================================

    final_test_start = metadata.get("final_test_start")
    final_test_end = metadata.get("final_test_end")

    if not final_test_start or not final_test_end:
        print()
        print("ERROR: metadata.json does not contain")
        print("final_test_start / final_test_end")
        return 1

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        utc=True,
    )

    start = pd.to_datetime(
        final_test_start,
        utc=True,
    )

    end = pd.to_datetime(
        final_test_end,
        utc=True,
    )

    work = work[
        (work["timestamp"] >= start)
        & (work["timestamp"] <= end)
    ].copy()

    if work.empty:
        print("ERROR: final test window contains no usable rows.")
        return 1

    # ========================================================
    # DISPLAY TEST WINDOW
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL TEST WINDOW")
    print("=" * 80)

    print(f"Start: {start}")
    print(f"End:   {end}")
    print(f"Rows:  {len(work)}")

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    y_true = work["label"]
    y_pred = work["pred_label"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    print()
    print("=" * 80)
    print("OVERALL FINAL TEST")
    print("=" * 80)

    print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Balanced accuracy: {balanced:.4f} ({balanced * 100:.2f}%)")

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)

    pred_counts = work["pred_label"].value_counts()

    for label in [-1, 0, 1]:

        count = int(pred_counts.get(label, 0))

        pct = (
            count / len(work) * 100
            if len(work)
            else 0
        )

        print(
            f"{LABEL_TO_NAME[label]:<6} "
            f"{count:>6} "
            f"({pct:>6.2f}%)"
        )

    # ========================================================
    # ACTUAL LABEL DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print("ACTUAL TRIPLE-BARRIER OUTCOME DISTRIBUTION")
    print("=" * 80)

    true_counts = work["label"].value_counts()

    for label in [-1, 0, 1]:

        count = int(true_counts.get(label, 0))

        pct = (
            count / len(work) * 100
            if len(work)
            else 0
        )

        print(
            f"{LABEL_TO_NAME[label]:<6} "
            f"{count:>6} "
            f"({pct:>6.2f}%)"
        )

    # ========================================================
    # CONFIDENCE STATISTICS
    # ========================================================

    print()
    print("=" * 80)
    print("RAW CONFIDENCE")
    print("=" * 80)

    print(
        work["confidence"]
        .describe()
        .to_string()
    )

    # ========================================================
    # CONFIDENCE BUCKET ANALYSIS
    # ========================================================

    bins = [
        0.00,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        1.01,
    ]

    work["confidence_bucket"] = pd.cut(
        work["confidence"],
        bins=bins,
        right=False,
    )

    print()
    print("=" * 100)
    print("ACCURACY BY CONFIDENCE")
    print("=" * 100)

    print(
        f"{'Confidence':<18}"
        f"{'Samples':>10}"
        f"{'Accuracy':>12}"
        f"{'Balanced':>12}"
    )

    print("-" * 100)

    for interval, group in work.groupby(
        "confidence_bucket",
        observed=True,
    ):

        if len(group) == 0:
            continue

        acc = accuracy_score(
            group["label"],
            group["pred_label"],
        )

        try:
            bal = balanced_accuracy_score(
                group["label"],
                group["pred_label"],
            )
        except Exception:
            bal = float("nan")

        print(
            f"{str(interval):<18}"
            f"{len(group):>10}"
            f"{acc * 100:>11.2f}%"
            f"{bal * 100:>11.2f}%"
        )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print()
    print("=" * 100)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 100)

    print(
        f"{'Threshold':<12}"
        f"{'Samples':>10}"
        f"{'% Kept':>10}"
        f"{'Accuracy':>12}"
        f"{'Balanced':>12}"
        f"{'BUY':>8}"
        f"{'SELL':>8}"
        f"{'HOLD':>8}"
    )

    print("-" * 100)

    threshold_results = []

    for threshold in CONFIDENCE_THRESHOLDS:

        selected = work[
            work["confidence"] >= threshold
        ]

        if len(selected) == 0:

            print(
                f"{threshold:<12.2f}"
                f"{0:>10}"
                f"{0:>9.2f}%"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{0:>8}"
                f"{0:>8}"
                f"{0:>8}"
            )

            continue

        acc = accuracy_score(
            selected["label"],
            selected["pred_label"],
        )

        try:
            bal = balanced_accuracy_score(
                selected["label"],
                selected["pred_label"],
            )
        except Exception:
            bal = float("nan")

        kept_pct = (
            len(selected)
            / len(work)
            * 100
        )

        buy_count = int(
            (selected["pred_label"] == 1).sum()
        )

        sell_count = int(
            (selected["pred_label"] == -1).sum()
        )

        hold_count = int(
            (selected["pred_label"] == 0).sum()
        )

        print(
            f"{threshold:<12.2f}"
            f"{len(selected):>10}"
            f"{kept_pct:>9.2f}%"
            f"{acc * 100:>11.2f}%"
            f"{bal * 100:>11.2f}%"
            f"{buy_count:>8}"
            f"{sell_count:>8}"
            f"{hold_count:>8}"
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "samples": len(selected),
                "kept_pct": kept_pct,
                "accuracy": acc,
                "balanced_accuracy": bal,
            }
        )

    # ========================================================
    # BUY / SELL / HOLD PERFORMANCE
    # ========================================================

    print()
    print("=" * 100)
    print("PER-PREDICTION-SIDE ACCURACY")
    print("=" * 100)

    print(
        f"{'Prediction':<12}"
        f"{'Samples':>10}"
        f"{'Correct':>10}"
        f"{'Accuracy':>12}"
    )

    print("-" * 100)

    for label in [-1, 0, 1]:

        group = work[
            work["pred_label"] == label
        ]

        if len(group) == 0:
            continue

        correct = (
            group["pred_label"]
            == group["label"]
        ).sum()

        acc = correct / len(group)

        print(
            f"{LABEL_TO_NAME[label]:<12}"
            f"{len(group):>10}"
            f"{correct:>10}"
            f"{acc * 100:>11.2f}%"
        )

    # ========================================================
    # BEST THRESHOLD
    # ========================================================

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if threshold_results:

        # Require at least 100 samples so that a tiny
        # high-confidence subset doesn't automatically win.
        meaningful = [
            r
            for r in threshold_results
            if r["samples"] >= 100
        ]

        if meaningful:

            best = max(
                meaningful,
                key=lambda x: x["accuracy"],
            )

            print(
                f"Best accuracy threshold with >=100 samples: "
                f"{best['threshold']:.2f}"
            )

            print(
                f"Accuracy: {best['accuracy'] * 100:.2f}%"
            )

            print(
                f"Balanced accuracy: "
                f"{best['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Samples: {best['samples']} "
                f"({best['kept_pct']:.2f}% of final test)"
            )

        else:

            print(
                "No confidence threshold retained at least "
                "100 final-test samples."
            )

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print(
        "This script evaluates the model against the SAME "
        "triple-barrier labeling configuration used during training."
    )

    print(
        "It uses RAW ensemble confidence and does NOT use the "
        "isotonic calibrator."
    )

    print(
        "Do not select a threshold solely from this table. "
        "The next step is confidence-threshold backtesting "
        "with fees, slippage, SL/TP and position sizing."
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())