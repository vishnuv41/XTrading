"""
ml/train.py
-------------
Orchestrates the full training pipeline:

    OHLCV -> indicators -> ml features -> triple-barrier labels
          -> purged walk-forward CV -> ensemble fit -> calibration
          -> save models to disk

Run directly (`python -m ml.train`) for a smoke run on synthetic data,
or import `run_training_pipeline` and call it with a real OHLCV
DataFrame from Person 1's PostgreSQL output.
"""

import os
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score

from ml.utils.preprocessing import build_feature_matrix, NON_FEATURE_COLS
from ml.labeling import triple_barrier_labels
from ml.models import XGBoostModel, LightGBMModel, CatBoostModel, EnsembleModel
from ml.calibration import ProbabilityCalibrator
from ml.validation import PurgedKFold


LABEL_MAP = {-1: 0, 0: 1, 1: 2}          # triple-barrier label -> class index
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# build_feature_matrix / NON_FEATURE_COLS now live in ml/utils/preprocessing.py
# (the single shared implementation used by train, predict, and explainability)
# and are re-exported here so existing imports of `from ml.train import
# build_feature_matrix` keep working unchanged.


def build_labels(
    df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 20,
) -> pd.DataFrame:
    """Triple-barrier labels, using ATR-normalized volatility if available."""
    volatility = None
    if "ATR14" in df.columns:
        volatility = df["ATR14"] / df["close"]
    return triple_barrier_labels(
        df, volatility=volatility, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding
    )


def prepare_dataset(df: pd.DataFrame, labels: pd.DataFrame, return_timestamps: bool = False,
                     label_map: dict = None):
    """
    Align features and labels, drop warm-up/cool-down NaN rows, and
    return (X, y, t1) ready for training. y is remapped via label_map
    (defaults to the module-level LABEL_MAP, i.e. triple-barrier
    {-1,0,1} -> {0,1,2}) for classification; t1 is touch_idx, needed by
    PurgedKFold to purge overlapping-window leakage.

    label_map: optional override of the {raw_label: class_index}
        mapping. Added so alternate label schemes (e.g.
        enhanced_ml/labeling/tp_before_sl.py's binary {0,1} labels,
        which need an identity map rather than the triple-barrier
        3-class map) can reuse this function unchanged. Every existing
        caller passes nothing and gets the exact previous behavior.

    return_timestamps: if True, also returns `ts` (the surviving rows'
    'timestamp' column, same order/index as X/y/t1) as a 4th value —
    used by run_training_pipeline to record the final-test window's
    start/end in metadata.json. Default False keeps every existing
    caller (evaluate_model.py, tune_threshold.py) unpacking 3 values
    working unchanged.
    """
    if label_map is None:
        label_map = LABEL_MAP
    combined = df.join(labels[["label", "touch_idx"]])
    combined = combined.dropna(subset=["label", "touch_idx"])

    feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLS and c not in ("touch_idx",)]
    X = combined[feature_cols].copy()

    # drop any remaining NaN rows from indicator/feature warm-up windows
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = combined.loc[valid_mask, "label"].map(label_map).astype(int)

    # `touch_idx` (from triple_barrier_labels) holds POSITIONAL indices
    # into the original, un-filtered `df` (0..len(df)-1) — it was
    # computed before any warm-up-NaN rows were dropped above. X/y have
    # just been compacted to a *new* 0..(m-1) range via reset_index, so
    # a raw touch_idx value is stale: it can point far past the end of
    # the compacted array (e.g. warm-up indicators like SMA200 alone
    # drop enough rows that touch_idx routinely exceeds len(X)).
    # PurgedKFold treats t1.iloc[i] as a position *within X*, so an
    # unremapped stale value makes it purge almost everything (or
    # crashes with an empty training fold on small datasets). Remap
    # each touch_idx to the position, in the compacted array, of the
    # nearest surviving row at-or-after that original position — this
    # preserves "how far forward this label's info extends" in the new
    # index space. If the touched row itself was dropped, clip forward
    # to the next survivor (or the last row, if none survive after it).
    surviving_orig_pos = combined.index[valid_mask].to_numpy()
    raw_touch = combined.loc[valid_mask, "touch_idx"].to_numpy()
    new_pos = np.searchsorted(surviving_orig_pos, raw_touch, side="left")
    new_pos = np.clip(new_pos, 0, len(surviving_orig_pos) - 1)
    t1 = pd.Series(new_pos, dtype=float)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if return_timestamps:
        ts = combined.loc[valid_mask, "timestamp"].reset_index(drop=True) if "timestamp" in combined.columns else pd.Series([pd.NaT] * len(X))
        return X, y, t1, ts
    return X, y, t1


def run_training_pipeline(
    df: pd.DataFrame,
    has_volume: bool = True,
    n_splits: int = 5,
    pct_embargo: float = 0.02,
    calibration_holdout_frac: float = 0.15,
    final_test_frac: float = 0.25,
    output_dir: str = "models_artifacts",
    feature_fn=None,
    label_fn=None,
    label_map: dict = None,
):
    """
    Full pipeline: features -> labels -> purged-CV stacking ensemble ->
    calibration on a chronological holdout -> save to disk.

    Chronological 3-way split (train / calibration / final test), in
    that order, oldest-to-newest — train fraction is whatever's left
    after calibration_holdout_frac + final_test_frac (default 0.70):
      TRAIN:       fits the ensemble (via PurgedKFold CV) only.
      CALIBRATION: fits the isotonic calibrator only — never seen by
                   the ensemble during CV fitting.
      FINAL TEST:  touched by NOTHING above. Not used to fit the
                   ensemble, not used to select CV folds, not used to
                   fit or select the calibrator. Scored once, at the
                   end, purely for reporting — this is what should be
                   compared against ml/baselines.py's simple baselines,
                   NOT the calibration-holdout accuracy (that number is
                   influenced by which calibration method/hyperparams
                   got picked, however lightly).

    Returns
    -------
    dict with keys: 'ensemble', 'calibrator', 'X_columns', 'metrics'

    feature_fn, label_fn, label_map : optional hooks for alternate
        pipelines (e.g. enhanced_ml's TP-before-SL label / MTF features)
        to reuse this exact function — the purged CV, stacking ensemble,
        calibration, integrity checks, and artifact saving below are
        identical either way, only the feature/label GENERATION differs.
        All three default to None, which reproduces the original,
        unmodified V1 behavior byte-for-byte (build_feature_matrix +
        build_labels + the module-level LABEL_MAP) — no existing caller
        passes these, so V1 training runs are unaffected.
          feature_fn(df, has_volume=has_volume) -> feat_df, if given.
          label_fn(feat_df) -> DataFrame with 'label' + 'touch_idx'
              columns (same contract as build_labels' output), if given.
          label_map: {raw_label: class_index}, if given (e.g. {0: 0,
              1: 1} for tp_before_sl's already-binary 0/1 labels, vs.
              the default {-1: 0, 0: 1, 1: 2} for triple-barrier).
    """
    _feature_fn = feature_fn if feature_fn is not None else build_feature_matrix
    _label_fn = label_fn if label_fn is not None else build_labels
    _label_map = label_map if label_map is not None else LABEL_MAP
    _inverse_label_map = {v: k for k, v in _label_map.items()}
    n_classes = len(_label_map)

    print("[1/6] Computing indicators + features...")
    feat_df = _feature_fn(df, has_volume=has_volume)

    print("[2/6] Computing triple-barrier labels...")
    labels = _label_fn(feat_df)

    print("[3/6] Preparing dataset...")
    X, y, t1, ts = prepare_dataset(feat_df, labels, return_timestamps=True, label_map=_label_map)
    print(f"    -> {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    -> label distribution: {y.value_counts().to_dict()}")

    # Chronological 3-way split: train / calibration / final test.
    # Final test is carved out FIRST (from the end) and never touched
    # again below — no fitting, no CV, no calibration uses it.
    n = len(X)
    train_frac = 1.0 - calibration_holdout_frac - final_test_frac
    if train_frac <= 0:
        raise ValueError(
            f"calibration_holdout_frac ({calibration_holdout_frac}) + final_test_frac "
            f"({final_test_frac}) must be < 1.0"
        )
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + calibration_holdout_frac))

    X_train, y_train, t1_train = X.iloc[:train_end], y.iloc[:train_end], t1.iloc[:train_end]
    X_hold, y_hold = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
    X_test, y_test = X.iloc[cal_end:], y.iloc[cal_end:]
    ts_test = ts.iloc[cal_end:]

    print(f"    -> train: {len(X_train)} rows, calibration: {len(X_hold)} rows, "
          f"final test: {len(X_test)} rows (untouched until final scoring)")

    print("[4/6] Fitting stacking ensemble with purged K-fold CV...")
    # Fixed at 3 (len(LABEL_MAP)) rather than y.nunique(): a class can be
    # legitimately absent from a given sample/fold (e.g. very few 'flat'
    # vertical-barrier touches), and inferring num_class from what's
    # present breaks XGBoost's binary-vs-multiclass path when the
    # surviving label *values* aren't contiguous from 0.
    # n_classes computed above from _label_map (defaults to len(LABEL_MAP) == 3,
    # identical to the original hardcoded value for every existing V1 caller).
    base_models = {
        "xgboost": XGBoostModel(num_class=n_classes, class_weight="balanced"),
        "lightgbm": LightGBMModel(num_class=n_classes, class_weight="balanced"),
        "catboost": CatBoostModel(num_class=n_classes, class_weight="balanced"),
    }
    ensemble = EnsembleModel(base_models, blend_method="stacking")
    cv = PurgedKFold(n_splits=n_splits, t1=t1_train, pct_embargo=pct_embargo)
    ensemble.fit(X_train, y_train, cv=list(cv.split(X_train)))

    # --- artifact integrity check: fail loudly rather than silently save
    # a collapsed model. This is the exact failure mode that produced the
    # "HOLD disappeared" incident — meta_learner.classes_ / a base
    # model's present_classes_ narrower than the full label space, most
    # often because a stale/mismatched artifact directory got inspected
    # rather than because THIS run actually collapsed a class. Checking
    # here means a genuinely broken run can never be saved and mistaken
    # for a good one again. ---
    expected_classes = set(range(n_classes))
    if hasattr(ensemble.meta_learner, "classes_"):
        meta_classes = set(int(c) for c in ensemble.meta_learner.classes_)
        if meta_classes != expected_classes:
            raise RuntimeError(
                f"Refusing to save: meta-learner only saw classes {sorted(meta_classes)} "
                f"during OOF fitting, expected {sorted(expected_classes)}. This means at least "
                f"one class (check LABEL_MAP for which) never appeared in y.values[oof_filled] — "
                f"do not save/deploy this artifact. Re-run training; if this recurs, the OOF "
                f"masking/fold logic in ml/models/ensemble.py needs investigating, not the labels "
                f"(y_train's own class distribution is printed above and should be checked first)."
            )
    for name, model in ensemble.fitted_base_models_.items():
        present = getattr(getattr(model, "label_space_", None), "present_classes_", None)
        if present is not None and set(int(c) for c in present) != expected_classes:
            raise RuntimeError(
                f"Refusing to save: base model '{name}' only saw classes {sorted(set(int(c) for c in present))} "
                f"during its full-training-set refit, expected {sorted(expected_classes)}. "
                f"y_train passed into ensemble.fit() should be checked — this means the y reaching "
                f"model.fit() inside EnsembleModel.fit()'s refit loop was missing a class, not that "
                f"the label pipeline itself is wrong (that's verified separately, above)."
            )
    print(f"    -> artifact integrity check passed: all classes {sorted(expected_classes)} present in "
          f"meta-learner and all {len(ensemble.fitted_base_models_)} base models.")

    print("[5/6] Calibrating probabilities on chronological holdout...")
    raw_proba_hold = ensemble.predict_proba(X_hold)
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_proba_hold, y_hold.values)

    calibrated_hold = calibrator.transform(raw_proba_hold)
    # Decide from RAW probabilities, matching ml/predict.py's fix — using
    # argmax(calibrated) here would make training's self-reported
    # accuracy measure a different decision rule than what predict()
    # actually uses in production, silently drifting the two apart again.
    pred_hold = np.argmax(raw_proba_hold, axis=1)
    accuracy = (pred_hold == y_hold.values).mean()

    print(f"    -> Calibration-holdout accuracy: {accuracy:.4f}")

    print("[6/6] Scoring untouched final test set (reporting only)...")
    raw_proba_test = ensemble.predict_proba(X_test)
    calibrated_test = calibrator.transform(raw_proba_test)
    pred_test = np.argmax(raw_proba_test, axis=1)

    final_test_accuracy = (pred_test == y_test.values).mean()
    final_test_balanced_accuracy = balanced_accuracy_score(y_test.values, pred_test)
    final_test_report = classification_report(
        y_test.values, pred_test, labels=list(_inverse_label_map.keys()),
        target_names=[str(k) for k in _inverse_label_map.values()],
        output_dict=True, zero_division=0,
    )
    print(f"    -> FINAL TEST accuracy: {final_test_accuracy:.4f} "
          f"(balanced: {final_test_balanced_accuracy:.4f})")
    print("    -> Compare this — not calibration-holdout accuracy — against "
          "ml/baselines.py before trusting the model.")

    metrics = {
        "holdout_accuracy": accuracy,
        "final_test_accuracy": final_test_accuracy,
        "final_test_balanced_accuracy": final_test_balanced_accuracy,
        "final_test_classification_report": final_test_report,
        "n_samples": n,
        "n_features": X.shape[1],
    }
    feature_columns = list(X.columns)

    os.makedirs(output_dir, exist_ok=True)

    # Per-base-model artifacts (native xgboost/lightgbm/catboost formats,
    # via each wrapper's own .save()) — kept for version-safe reloading
    # and backward compatibility with ml.predict.load_ensemble's fallback.
    for name, model in ensemble.fitted_base_models_.items():
        model.save(os.path.join(output_dir, f"{name}_model"))

    # Whole-ensemble artifact (joblib pickle of the fitted EnsembleModel,
    # including its meta-learner and blend_method). This is what
    # inference/api.py loads directly as `ensemble.pkl`, and what
    # ml.predict.load_ensemble prefers when present — reloading it
    # reproduces the *exact* predict_proba behavior (stacking + the
    # trained meta-learner) that the calibrator below was fit against,
    # instead of falling back to plain averaging.
    joblib.dump(ensemble, os.path.join(output_dir, "ensemble.pkl"))

    # Calibrator — persisted as the object itself (not just its raw
    # state) so `joblib.load(...)` returns something with .transform()
    # ready to use, matching how inference/api.py loads it.
    calibrator.save(os.path.join(output_dir, "calibrator.pkl"))

    # Exact training-time feature column order. Predicting on new data
    # must reproduce this order exactly, since the boosting libraries
    # match columns positionally as well as by name.
    joblib.dump(feature_columns, os.path.join(output_dir, "feature_columns.pkl"))

    # Label mapping (triple-barrier {-1,0,1} <-> model class index
    # {0,1,2}), so any consumer can decode predictions without importing
    # ml.train and without risk of the mapping silently changing later.
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(
            {
                "LABEL_MAP": {str(k): v for k, v in _label_map.items()},
                "INVERSE_LABEL_MAP": {str(k): v for k, v in _inverse_label_map.items()},
            },
            f,
            indent=2,
        )

    # Model metadata — a quick-reference manifest for this training run,
    # useful for debugging, monitoring dashboards, and sanity-checking
    # that a deployed model matches what was actually trained.
    metadata = {
        "model_version": "1.0",
        "trained_on": datetime.now(timezone.utc).isoformat(),
        "n_samples": n,
        "n_features": X.shape[1],
        "feature_names": feature_columns,
        "holdout_accuracy": accuracy,
        "final_test_accuracy": final_test_accuracy,
        "final_test_balanced_accuracy": final_test_balanced_accuracy,
        "final_test_start": str(ts_test.iloc[0]) if len(ts_test) else None,
        "final_test_end": str(ts_test.iloc[-1]) if len(ts_test) else None,
        "final_test_n_samples": len(X_test),
        "base_models": list(ensemble.fitted_base_models_.keys()),
        "blend_method": ensemble.blend_method,
        "calibration_method": calibrator.method,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "ensemble": ensemble,
        "calibrator": calibrator,
        "X_columns": feature_columns,
        "metrics": metrics,
    }


def _make_synthetic_ohlcv(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.random(n) * 1.5
    low = close - rng.random(n) * 1.5
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100, 5000, n)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }
    )
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


if __name__ == "__main__":
    df = _make_synthetic_ohlcv()
    result = run_training_pipeline(df, output_dir="models_artifacts")
    print("\nDone. Metrics:", result["metrics"])