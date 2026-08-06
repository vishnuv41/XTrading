"""
evaluate_model.py
------------------
Proper evaluation of a trained model: confusion matrix, per-class
precision/recall, a binomial confidence interval, majority-class
baseline, AND drift/collapse diagnostics:
  - has the DB grown since training (so this holdout != the training holdout)?
  - is the collapse caused by calibration clipping (raw vs calibrated
    prediction distributions), or is the base ensemble itself collapsed?

Usage:
    python evaluate_model.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h
"""
import argparse
import json
import os

import joblib
import numpy as np
from scipy import stats

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.utils.preprocessing import build_feature_matrix
from ml.train import build_labels, prepare_dataset, LABEL_MAP, INVERSE_LABEL_MAP


def binomial_ci(correct, n, conf=0.95):
    p = correct / n
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return center - margin, center + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--exchange", default=settings.exchange.default_exchange)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
    feat_df = build_feature_matrix(df, has_volume=True)
    labels = build_labels(feat_df)
    X, y, t1 = prepare_dataset(feat_df, labels)
    n = len(X)

    # --- drift check: has the dataset grown since training? ---
    meta_path = os.path.join(args.model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        trained_n = meta.get("n_samples")
        if trained_n is not None and trained_n != n:
            print(f"⚠️  DATA DRIFT: model was trained with n_samples={trained_n}, "
                  f"but the DB now yields n_samples={n} (+{n - trained_n}). "
                  f"This means the 15% holdout below is NOT the same holdout "
                  f"the model was calibrated on — it now includes newer candles "
                  f"the model has never seen. Treat this as a forward/live test, "
                  f"not a reproduction of the training-time number.\n")

    holdout_start = int(n * (1 - args.holdout_frac))
    X_hold, y_hold = X.iloc[holdout_start:], y.iloc[holdout_start:]

    ensemble = joblib.load(f"{args.model_dir}/ensemble.pkl")
    calibrator = joblib.load(f"{args.model_dir}/calibrator.pkl")

    raw = ensemble.predict_proba(X_hold)
    calibrated = calibrator.transform(raw)
    pred_raw = np.argmax(raw, axis=1)
    pred = np.argmax(calibrated, axis=1)
    y_true = y_hold.values

    # --- collapse diagnostic: is it the base ensemble or the calibrator? ---
    raw_dist = {int(c): int((pred_raw == c).sum()) for c in np.unique(pred_raw)}
    cal_dist = {int(c): int((pred == c).sum()) for c in np.unique(pred)}
    print(f"Raw (uncalibrated) prediction distribution: {raw_dist}")
    print(f"Calibrated prediction distribution:          {cal_dist}")
    if len(raw_dist) > 1 and len(cal_dist) == 1:
        print("-> Base ensemble still predicts multiple classes; the CALIBRATOR "
              "is collapsing everything to one class. Likely isotonic clipping "
              "on out-of-distribution raw scores (market regime shifted since "
              "the calibrator was fit). Consider: refit calibrator on recent "
              "data, or switch method='sigmoid' (extrapolates instead of "
              "clipping), or retrain entirely on fresher data.")
    elif len(raw_dist) == 1:
        print("-> The base ensemble itself always predicts one class — this is "
              "a model/feature problem, not a calibration problem. Check "
              "feature drift (SHAP / distribution of X_hold vs X_train) or "
              "retrain on more recent data.")
    print()

    correct = int((pred == y_true).sum())
    n_hold = len(y_true)
    acc = correct / n_hold
    lo, hi = binomial_ci(correct, n_hold)

    values, counts = np.unique(y_true, return_counts=True)
    majority_acc = counts.max() / n_hold

    classes = sorted(LABEL_MAP.values())
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1

    print(f"{args.symbol} {args.timeframe} — holdout n={n_hold}")
    print(f"Accuracy: {acc:.4f}  (95% CI: {lo:.4f} - {hi:.4f})")
    print(f"Majority-class baseline: {majority_acc:.4f}")
    print(f"Edge over baseline: {(acc - majority_acc)*100:+.2f} pts\n")

    print("Confusion matrix (rows=true, cols=pred), classes =",
          {c: INVERSE_LABEL_MAP[c] for c in classes}, ":")
    print(cm)

    print("\nPer-class precision / recall:")
    for i, c in enumerate(classes):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else float("nan")
        recall = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else float("nan")
        print(f"  class {c} (barrier={INVERSE_LABEL_MAP[c]:+d}): "
              f"precision={precision:.3f} recall={recall:.3f} support={cm[i,:].sum()}")

    if n_hold < 100:
        print(f"\n⚠️  WARNING: holdout has only {n_hold} samples — this accuracy "
              f"number has a huge confidence interval and should not be trusted "
              f"on its own.")


if __name__ == "__main__":
    main()