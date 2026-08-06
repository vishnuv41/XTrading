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


def prepare_dataset(df: pd.DataFrame, labels: pd.DataFrame):
    """
    Align features and labels, drop warm-up/cool-down NaN rows, and
    return (X, y, t1) ready for training. y is remapped to {0,1,2} via
    LABEL_MAP for multiclass classification; t1 is touch_idx, needed by
    PurgedKFold to purge overlapping-window leakage.
    """
    combined = df.join(labels[["label", "touch_idx"]])
    combined = combined.dropna(subset=["label", "touch_idx"])

    feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLS and c not in ("touch_idx",)]
    X = combined[feature_cols].copy()

    # drop any remaining NaN rows from indicator/feature warm-up windows
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = combined.loc[valid_mask, "label"].map(LABEL_MAP).astype(int)

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
    return X, y, t1


def run_training_pipeline(
    df: pd.DataFrame,
    has_volume: bool = True,
    n_splits: int = 5,
    pct_embargo: float = 0.02,
    calibration_holdout_frac: float = 0.15,
    output_dir: str = "models_artifacts",
):
    """
    Full pipeline: features -> labels -> purged-CV stacking ensemble ->
    calibration on a final chronological holdout -> save to disk.

    Returns
    -------
    dict with keys: 'ensemble', 'calibrator', 'X_columns', 'metrics'
    """
    print("[1/6] Computing indicators + features...")
    feat_df = build_feature_matrix(df, has_volume=has_volume)

    print("[2/6] Computing triple-barrier labels...")
    labels = build_labels(feat_df)

    print("[3/6] Preparing dataset...")
    X, y, t1 = prepare_dataset(feat_df, labels)
    print(f"    -> {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    -> label distribution: {y.value_counts().to_dict()}")

    # chronological holdout for calibration — never used in CV fitting
    n = len(X)
    holdout_start = int(n * (1 - calibration_holdout_frac))
    X_train, y_train, t1_train = X.iloc[:holdout_start], y.iloc[:holdout_start], t1.iloc[:holdout_start]
    X_hold, y_hold = X.iloc[holdout_start:], y.iloc[holdout_start:]

    print("[4/6] Fitting stacking ensemble with purged K-fold CV...")
    # Fixed at 3 (len(LABEL_MAP)) rather than y.nunique(): a class can be
    # legitimately absent from a given sample/fold (e.g. very few 'flat'
    # vertical-barrier touches), and inferring num_class from what's
    # present breaks XGBoost's binary-vs-multiclass path when the
    # surviving label *values* aren't contiguous from 0.
    n_classes = len(LABEL_MAP)
    base_models = {
        "xgboost": XGBoostModel(num_class=n_classes, class_weight="balanced"),
        "lightgbm": LightGBMModel(num_class=n_classes, class_weight="balanced"),
        "catboost": CatBoostModel(num_class=n_classes, class_weight="balanced"),
    }
    ensemble = EnsembleModel(base_models, blend_method="stacking")
    cv = PurgedKFold(n_splits=n_splits, t1=t1_train, pct_embargo=pct_embargo)
    ensemble.fit(X_train, y_train, cv=list(cv.split(X_train)))

    print("[5/6] Calibrating probabilities on chronological holdout...")
    raw_proba_hold = ensemble.predict_proba(X_hold)
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_proba_hold, y_hold.values)

    calibrated_hold = calibrator.transform(raw_proba_hold)
    pred_hold = np.argmax(calibrated_hold, axis=1)
    accuracy = (pred_hold == y_hold.values).mean()

    print(f"[6/6] Holdout accuracy (calibrated): {accuracy:.4f}")

    metrics = {"holdout_accuracy": accuracy, "n_samples": n, "n_features": X.shape[1]}
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
                "LABEL_MAP": {str(k): v for k, v in LABEL_MAP.items()},
                "INVERSE_LABEL_MAP": {str(k): v for k, v in INVERSE_LABEL_MAP.items()},
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