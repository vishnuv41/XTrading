"""
compare_calibration_methods.py
--------------------------------
READ-ONLY. Does not modify ml/calibration/, does not overwrite the
production calibrator.pkl, does not retrain base models or the
meta-learner. Reuses the already-fitted ensemble.pkl.

Question: is isotonic calibration (currently in production, fit on only
~148 HOLD positives in the calibration holdout) actively harmful
relative to sigmoid (Platt scaling — needs fewer samples, assumes a
simpler sigmoid-shaped miscalibration)?

Method-selection discipline: the 1288-row calibration holdout is split
chronologically 70/30 into calib_fit / calib_val. Both calibration
methods are fit on calib_fit ONLY and compared on calib_val. That
comparison — not the final test set — is what should drive picking a
method, per the "final test untouched for model selection" rule.
Final-test numbers are printed afterward, clearly labeled as reporting
only, not as the basis for the decision.

Run:
    python compare_calibration_methods.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts
"""

import argparse

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, brier_score_loss

from config.settings import settings
from ml.calibration import ProbabilityCalibrator
from diagnose_ensemble import rebuild_split, DISPLAY_NAMES  # reuse the exact same split logic


def evaluate(name, proba, y_true):
    pred = np.argmax(proba, axis=1)
    bal_acc = balanced_accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    ll = log_loss(y_true, proba, labels=[0, 1, 2])
    hold_recall = ((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
    hold_predicted = (pred == 1).sum()
    brier_hold = brier_score_loss((y_true == 1).astype(int), proba[:, 1])
    print(f"  {name:22s}  bal_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}  log_loss={ll:.4f}  "
          f"Brier(HOLD)={brier_hold:.4f}  HOLD_predicted={hold_predicted:4d}  HOLD_recall={hold_recall:.4f}")
    return dict(bal_acc=bal_acc, macro_f1=macro_f1, log_loss=ll, brier_hold=brier_hold,
                hold_predicted=hold_predicted, hold_recall=hold_recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=settings.symbols.symbols[0])
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--exchange", default=settings.exchange.default_exchange)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    ensemble = joblib.load(f"{args.model_dir}/ensemble.pkl")
    X_hold, y_hold, X_test, y_test = rebuild_split(args.symbol, args.timeframe, args.exchange)

    # chronological 70/30 split of the calibration holdout itself
    split_at = int(len(X_hold) * 0.7)
    X_cf, y_cf = X_hold.iloc[:split_at], y_hold.iloc[:split_at]
    X_cv, y_cv = X_hold.iloc[split_at:], y_hold.iloc[split_at:]
    print(f"\ncalib_fit: {len(X_cf)} rows (HOLD={int((y_cf == 1).sum())})   "
          f"calib_val: {len(X_cv)} rows (HOLD={int((y_cv == 1).sum())})")

    raw_cf = ensemble.predict_proba(X_cf)
    raw_cv = ensemble.predict_proba(X_cv)
    raw_test = ensemble.predict_proba(X_test)

    print(f"\n{'='*78}\nSELECTION METRICS on calib_val (this decides the method — NOT final test)\n{'='*78}")
    print("  raw (uncalibrated) baseline on calib_val:")
    evaluate("raw", raw_cv, y_cv.values)

    results = {}
    for method in ("isotonic", "sigmoid"):
        cal = ProbabilityCalibrator(method=method)
        cal.fit(raw_cf, y_cf.values)
        calibrated_cv = cal.transform(raw_cv)
        print(f"\n  {method} (fit on calib_fit={len(X_cf)} rows):")
        results[method] = evaluate(method, calibrated_cv, y_cv.values)
        results[method]["_calibrator"] = cal

    better = max(results, key=lambda m: results[m]["macro_f1"])
    print(f"\n  -> Higher macro_f1 on calib_val: {better} "
          f"({results[better]['macro_f1']:.4f} vs {results[[m for m in results if m != better][0]]['macro_f1']:.4f})")

    print(f"\n{'='*78}\nFINAL TEST — reporting only, NOT used to pick the method above\n{'='*78}")
    print("  raw (uncalibrated) baseline on final test:")
    evaluate("raw", raw_test, y_test.values)
    for method in ("isotonic", "sigmoid"):
        calibrated_test = results[method]["_calibrator"].transform(raw_test)
        evaluate(method, calibrated_test, y_test.values)


if __name__ == "__main__":
    main()