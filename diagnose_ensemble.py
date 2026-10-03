"""
diagnose_ensemble.py
---------------------
READ-ONLY diagnostic. Does not retrain anything and does not touch
ml/train.py, ml/models/ensemble.py, or ml/calibration/. It reloads the
already-trained artifacts for one symbol/timeframe, rebuilds the exact
same chronological train/calibration/final-test split that
ml.train.run_training_pipeline used (same fractions, same code path:
build_feature_matrix -> build_labels -> prepare_dataset), and reports,
per stage, the numbers needed to separate "base models are biased" from
"stacking destroys HOLD" from "calibration is unreliable":

  - actual class distribution (final test)
  - per-base-model: predicted class distribution, confusion matrix,
    mean P(class) per class, balanced accuracy, macro F1
  - simple average of the 3 base models: same metrics
  - stacked (meta-learner) output: same metrics
  - meta-learner coefficients / intercept (class 0/1/2 rows)
  - HOLD-specific: raw P(HOLD) distribution (min/25/50/75/max) for each
    of the above, before vs after calibration
  - calibration diagnostics: log loss and Brier score, raw vs
    calibrated, computed on the untouched final test set (reporting
    only — calibration itself was fit on the calibration holdout,
    never on final test)

Run from the XTrading/ root (same place you run run_training_from_db.py):

    python diagnose_ensemble.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h

Requires that model-dir already has ensemble.pkl, calibrator.pkl,
feature_columns.pkl (i.e. training has already run for this symbol/tf).
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, f1_score,
    log_loss, brier_score_loss,
)

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.utils.preprocessing import build_feature_matrix
from ml.train import build_labels, prepare_dataset, LABEL_MAP, INVERSE_LABEL_MAP

CLASS_NAMES = [INVERSE_LABEL_MAP[i] for i in range(len(INVERSE_LABEL_MAP))]  # index 0,1,2 -> raw label
CLASS_LABELS = {-1: "SELL", 0: "HOLD", 1: "BUY"}
DISPLAY_NAMES = [CLASS_LABELS[raw] for raw in CLASS_NAMES]  # e.g. ['SELL','HOLD','BUY'] in index 0,1,2 order


def rebuild_split(symbol, timeframe, exchange, calibration_holdout_frac=0.15, final_test_frac=0.25):
    """Exact same split logic/fractions as ml.train.run_training_pipeline,
    duplicated read-only here rather than importing private internals."""
    print(f"Loading {symbol} {timeframe} from {exchange}...")
    df = load_ohlcv(symbol, timeframe, exchange=exchange)

    feat_df = build_feature_matrix(df, has_volume=True)
    labels = build_labels(feat_df)
    X, y, t1, ts = prepare_dataset(feat_df, labels, return_timestamps=True)

    n = len(X)
    train_frac = 1.0 - calibration_holdout_frac - final_test_frac
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + calibration_holdout_frac))

    X_hold, y_hold = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
    X_test, y_test = X.iloc[cal_end:], y.iloc[cal_end:]
    ts_test = ts.iloc[cal_end:]

    print(f"  total usable rows: {n}  |  calibration: {len(X_hold)}  |  final test: {len(X_test)}")
    print(f"  final test window: {ts_test.iloc[0]} -> {ts_test.iloc[-1]}")
    return X_hold, y_hold, X_test, y_test


def describe_proba(name, proba, y_true):
    pred = np.argmax(proba, axis=1)
    dist = pd.Series(pred).value_counts().reindex(range(3), fill_value=0)
    mean_p = proba.mean(axis=0)
    bal_acc = balanced_accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])

    print(f"\n--- {name} ---")
    print(f"  predicted counts   SELL={dist[0]:5d}  HOLD={dist[1]:5d}  BUY={dist[2]:5d}")
    print(f"  mean P(class)      SELL={mean_p[0]:.4f}  HOLD={mean_p[1]:.4f}  BUY={mean_p[2]:.4f}")
    print(f"  balanced accuracy  {bal_acc:.4f}")
    print(f"  macro F1           {macro_f1:.4f}")
    print(f"  confusion matrix (rows=actual SELL/HOLD/BUY, cols=predicted SELL/HOLD/BUY):")
    print(f"    {cm[0].tolist()}")
    print(f"    {cm[1].tolist()}")
    print(f"    {cm[2].tolist()}")

    hold_p = proba[:, 1]
    q = np.percentile(hold_p, [0, 25, 50, 75, 100])
    print(f"  P(HOLD) distribution  min={q[0]:.4f} p25={q[1]:.4f} median={q[2]:.4f} p75={q[3]:.4f} max={q[4]:.4f}")
    return {"pred": pred, "proba": proba, "balanced_accuracy": bal_acc, "macro_f1": macro_f1}


def calibration_metrics(name, proba, y_true):
    """Log loss and per-class Brier score — NOT accuracy. These measure
    whether the probability VALUES are trustworthy, independent of
    whether argmax happens to be correct."""
    ll = log_loss(y_true, proba, labels=[0, 1, 2])
    briers = {}
    for cls in [0, 1, 2]:
        briers[cls] = brier_score_loss((y_true == cls).astype(int), proba[:, cls])
    print(f"\n--- {name}: probability quality (final test) ---")
    print(f"  log loss    {ll:.4f}")
    print(f"  Brier(SELL) {briers[0]:.4f}   Brier(HOLD) {briers[1]:.4f}   Brier(BUY) {briers[2]:.4f}")
    return {"log_loss": ll, "brier": briers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=settings.symbols.symbols[0])
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--exchange", default=settings.exchange.default_exchange)
    ap.add_argument("--model-dir", required=True, help="e.g. models_artifacts/BTCUSDT_1h")
    args = ap.parse_args()

    ensemble = joblib.load(f"{args.model_dir}/ensemble.pkl")
    calibrator = joblib.load(f"{args.model_dir}/calibrator.pkl")

    X_hold, y_hold, X_test, y_test = rebuild_split(args.symbol, args.timeframe, args.exchange)

    print(f"\n{'='*70}\nACTUAL FINAL-TEST CLASS DISTRIBUTION\n{'='*70}")
    actual = y_test.value_counts().reindex(range(3), fill_value=0)
    print(f"  SELL={actual[0]}  HOLD={actual[1]}  BUY={actual[2]}")

    print(f"\n{'='*70}\nPER-BASE-MODEL (final test, raw probabilities)\n{'='*70}")
    base_probas = {}
    for bname, model in ensemble.fitted_base_models_.items():
        proba = model.predict_proba(X_test)
        base_probas[bname] = proba
        describe_proba(bname, proba, y_test.values)

    print(f"\n{'='*70}\nSIMPLE AVERAGE of the 3 base models (final test)\n{'='*70}")
    avg_proba = np.mean(list(base_probas.values()), axis=0)
    describe_proba("simple_average", avg_proba, y_test.values)

    print(f"\n{'='*70}\nSTACKED (meta-learner) — ensemble.predict_proba (final test)\n{'='*70}")
    stacked_raw = ensemble.predict_proba(X_test)
    describe_proba("stacking_raw", stacked_raw, y_test.values)

    if hasattr(ensemble.meta_learner, "coef_"):
        print(f"\n--- meta-learner coefficients ---")
        print(f"  classes_: {ensemble.meta_learner.classes_}")
        for i, row in enumerate(ensemble.meta_learner.coef_):
            print(f"  class {i} coef: {np.round(row, 4).tolist()}")
        print(f"  intercept: {np.round(ensemble.meta_learner.intercept_, 4).tolist()}")

    print(f"\n{'='*70}\nCALIBRATED (stacked -> calibrator.transform) — final test\n{'='*70}")
    stacked_calibrated = calibrator.transform(stacked_raw)
    describe_proba("stacking_calibrated", stacked_calibrated, y_test.values)

    print(f"\n{'='*70}\nPROBABILITY QUALITY: raw vs calibrated (final test, reporting only)\n{'='*70}")
    calibration_metrics("stacking_raw", stacked_raw, y_test.values)
    calibration_metrics("stacking_calibrated", stacked_calibrated, y_test.values)

    # Sanity check flagged in the problem writeup: does calibration change
    # WHICH class wins (argmax), not just the confidence number?
    raw_pred = np.argmax(stacked_raw, axis=1)
    cal_pred = np.argmax(stacked_calibrated, axis=1)
    flipped = (raw_pred != cal_pred).sum()
    print(f"\n  rows where calibration changed the predicted class: {flipped} / {len(raw_pred)}")
    if flipped > 0:
        flip_dirs = pd.Series(
            [f"{DISPLAY_NAMES[a]}->{DISPLAY_NAMES[b]}" for a, b in zip(raw_pred, cal_pred) if a != b]
        ).value_counts()
        print(f"  flip directions:\n{flip_dirs.to_string()}")

    # Calibration-holdout size per class — the isotonic sample-size risk
    # flagged in ml/calibration/probability_calibration.py's own docstring.
    print(f"\n{'='*70}\nCALIBRATION HOLDOUT class counts (what the isotonic curves were fit on)\n{'='*70}")
    hold_counts = y_hold.value_counts().reindex(range(3), fill_value=0)
    print(f"  SELL={hold_counts[0]}  HOLD={hold_counts[1]}  BUY={hold_counts[2]}")
    print(f"  (module docstring: isotonic wants ~1000+ positive examples per class to avoid overfitting the curve)")


if __name__ == "__main__":
    main()