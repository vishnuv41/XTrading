"""
tune_threshold.py
------------------
Instead of naive argmax (which just defaults to whichever class has the
highest probability — here, "long", because the model is bullish-biased),
sweep the decision threshold for calling "short" (class 0) and find the
one that maximizes balanced accuracy on the holdout. Prints the result
and a comparison against plain argmax.

Usage:
    python tune_threshold.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h
"""
import argparse

import joblib
import numpy as np

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.utils.preprocessing import build_feature_matrix
from ml.train import build_labels, prepare_dataset, LABEL_MAP, INVERSE_LABEL_MAP


def balanced_accuracy(y_true, y_pred, classes):
    """Mean of per-class recall — robust to the class imbalance you have."""
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == c).mean())
    return np.mean(recalls)


def predict_with_short_threshold(calibrated_proba, short_threshold, short_class=0, long_class=2):
    """
    Call `short_class` whenever its calibrated probability exceeds
    `short_threshold` (even if it's not the argmax) — otherwise fall
    back to argmax between the remaining classes. This directly
    counteracts a model that's biased toward always picking the
    higher-average-probability class.
    """
    pred = np.argmax(calibrated_proba, axis=1)
    short_prob = calibrated_proba[:, short_class]
    pred = np.where(short_prob >= short_threshold, short_class, pred)
    return pred


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
    holdout_start = int(n * (1 - args.holdout_frac))
    X_hold, y_hold = X.iloc[holdout_start:], y.iloc[holdout_start:]
    y_true = y_hold.values

    ensemble = joblib.load(f"{args.model_dir}/ensemble.pkl")
    calibrator = joblib.load(f"{args.model_dir}/calibrator.pkl")
    calibrated = calibrator.transform(ensemble.predict_proba(X_hold))

    classes = sorted(LABEL_MAP.values())

    # baseline: plain argmax
    pred_argmax = np.argmax(calibrated, axis=1)
    base_acc = (pred_argmax == y_true).mean()
    base_bal = balanced_accuracy(y_true, pred_argmax, classes)

    print(f"Plain argmax    — accuracy: {base_acc:.4f}  balanced accuracy: {base_bal:.4f}")

    # sweep short-class threshold
    best = {"threshold": None, "acc": -1, "bal": -1}
    for thresh in np.arange(0.05, 0.95, 0.01):
        pred = predict_with_short_threshold(calibrated, thresh)
        acc = (pred == y_true).mean()
        bal = balanced_accuracy(y_true, pred, classes)
        if bal > best["bal"]:
            best = {"threshold": thresh, "acc": acc, "bal": bal}

    print(f"Best threshold  — short_threshold={best['threshold']:.2f}  "
          f"accuracy: {best['acc']:.4f}  balanced accuracy: {best['bal']:.4f}")
    print(f"\nBalanced-accuracy improvement: {(best['bal'] - base_bal)*100:+.2f} pts")

    pred_best = predict_with_short_threshold(calibrated, best["threshold"])
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, pred_best):
        cm[t, p] += 1
    print("\nConfusion matrix at tuned threshold (rows=true, cols=pred), classes =",
          {c: INVERSE_LABEL_MAP[c] for c in classes}, ":")
    print(cm)
    print(f"\nSave this threshold ({best['threshold']:.2f}) and use "
          f"predict_with_short_threshold() at inference time instead of plain argmax.")


if __name__ == "__main__":
    main()