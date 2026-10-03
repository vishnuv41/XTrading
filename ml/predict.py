"""
ml/predict.py
----------------
Loads trained base models, rebuilds the feature stack for new OHLCV
data, and returns calibrated probabilities + a confidence-scored
directional call per bar. This is the module inference/realtime_pipeline.py
should import rather than duplicating feature/model logic.

2026-08-09 CHANGE (see diagnose_ensemble.py / compare_calibration_methods.py
results): predict() used to pick pred_class from the CALIBRATED
probabilities (np.argmax(calibrator.transform(raw_proba))). Measured on
two independent holdouts (calib_val, split off before final test, and
final test itself, never used for method selection), this consistently
destroyed the HOLD class: every calibration flip moved rows AWAY from
HOLD or INTO BUY, and BOTH isotonic and sigmoid did this — ruling out
"wrong calibration method" and pointing at the mechanism itself:
ProbabilityCalibrator fits each class independently as its own one-vs-
rest curve, then renormalizes. Nothing about that process guarantees
the class with the highest RAW score still has the highest score after
calibration, and empirically it didn't.

Fix: pred_class is now decided from raw_proba (what the ensemble/
meta-learner actually believes), never re-ranked by calibration.
calibrator is used only to convert the winning class's raw score into a
calibrated confidence number — a scalar rescaling of an already-made
decision, not a re-decision. prob_down/prob_flat/prob_up are still the
full calibrated distribution (useful for expected-value math
downstream), but callers should NOT assume the column matching
pred_label is that row's max value anymore — confidence is the number
to use for "how much to trust this call", not max(prob_down,
prob_flat, prob_up).

This is a single, isolated change (2 lines of actual logic, in the
block marked below). Nothing in ml/calibration/, ml/models/ensemble.py,
or ml/train.py was touched. Compare against the pre-change baseline
(same models_artifacts/BTCUSDT_1h/, still isotonic-fit, unchanged)
before/after this file changes to confirm HOLD calls actually reappear
in inference/backtest output, per the "one controlled change -> re-
evaluate" rule — do not fold in any other change until that's confirmed.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from ml.models import XGBoostModel, LightGBMModel, CatBoostModel, EnsembleModel
from ml.calibration import ProbabilityCalibrator
from ml.train import LABEL_MAP, INVERSE_LABEL_MAP
from ml.utils.preprocessing import build_feature_matrix, prepare_model_input


def load_ensemble(model_dir: str, num_class: int = 3) -> EnsembleModel:
    """
    Reload a trained ensemble from `model_dir`.

    Prefers `ensemble.pkl` (a single joblib pickle of the fitted
    EnsembleModel, written by ml.train.run_training_pipeline) — this
    preserves the exact blend_method and trained meta-learner, so
    predict_proba behaves identically to how it did during training and
    calibration. Falls back to reconstructing from the three individual
    base-model files (native xgboost/lightgbm/catboost formats) with
    blend_method='average' if ensemble.pkl isn't present, e.g. when
    loading artifacts saved by an older version of the pipeline.
    """
    ensemble_path = os.path.join(model_dir, "ensemble.pkl")
    if os.path.exists(ensemble_path):
        return joblib.load(ensemble_path)

    xgb_model = XGBoostModel(num_class=num_class).load(os.path.join(model_dir, "xgboost_model"))
    lgb_model = LightGBMModel(num_class=num_class).load(os.path.join(model_dir, "lightgbm_model"))
    cat_model = CatBoostModel(num_class=num_class).load(os.path.join(model_dir, "catboost_model"))

    ensemble = EnsembleModel(
        {"xgboost": xgb_model, "lightgbm": lgb_model, "catboost": cat_model},
        blend_method="average",  # no meta-learner available from individual files; average is the safe fallback
    )
    ensemble.fitted_base_models_ = ensemble.base_models
    ensemble.classes_ = np.arange(num_class)
    return ensemble


def load_training_artifacts(model_dir: str, num_class: int = 3) -> dict:
    """
    Convenience loader for everything ml.train.run_training_pipeline
    persists to `model_dir`: the ensemble, the calibrator, the exact
    training-time feature column order, and the label mapping/metadata.
    Any artifact that isn't found (e.g. calibrator.pkl, for a run that
    predates calibrator persistence) is returned as None instead of
    raising, so callers can decide how to handle a partial model dir.

    Returns
    -------
    dict with keys: 'ensemble', 'calibrator', 'feature_columns',
    'label_map', 'inverse_label_map', 'metadata'.
    """
    ensemble = load_ensemble(model_dir, num_class=num_class)

    calibrator_path = os.path.join(model_dir, "calibrator.pkl")
    calibrator = ProbabilityCalibrator.load(calibrator_path) if os.path.exists(calibrator_path) else None

    features_path = os.path.join(model_dir, "feature_columns.pkl")
    feature_columns = joblib.load(features_path) if os.path.exists(features_path) else None

    label_map_path = os.path.join(model_dir, "label_map.json")
    label_map, inverse_label_map = LABEL_MAP, INVERSE_LABEL_MAP
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            saved = json.load(f)
        label_map = {int(k): v for k, v in saved["LABEL_MAP"].items()}
        inverse_label_map = {int(k): v for k, v in saved["INVERSE_LABEL_MAP"].items()}

    metadata_path = os.path.join(model_dir, "metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {
        "ensemble": ensemble,
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "label_map": label_map,
        "inverse_label_map": inverse_label_map,
        "metadata": metadata,
    }


def predict(
    df: pd.DataFrame,
    ensemble: EnsembleModel,
    feature_columns: list,
    calibrator=None,
    has_volume: bool = True,
) -> pd.DataFrame:
    """
    Compute features for `df` and return per-bar predictions.

    Parameters
    ----------
    df : raw OHLCV DataFrame (same shape/columns as training input).
    ensemble : a fitted EnsembleModel (from load_ensemble or in-memory
        from ml.train.run_training_pipeline).
    feature_columns : the exact training-time feature column order
        (ensemble['X_columns'] from run_training_pipeline's return dict).
    calibrator : optional fitted ProbabilityCalibrator; if given,
        confidence (and the reported per-class probabilities) are
        calibrated. The predicted class itself is always decided from
        raw, uncalibrated probabilities — see module docstring for why.
    has_volume : must match what the model was trained with.

    Returns
    -------
    DataFrame indexed like `df`, columns:
        - 'pred_label'    : predicted class, remapped back to {-1, 0, 1}
                             — decided from RAW ensemble probabilities.
        - 'prob_down', 'prob_flat', 'prob_up' : per-class probability
          (calibrated, if a calibrator is given). Informational / for
          expected-value math — NOT guaranteed to have its max in the
          column matching pred_label once calibrated.
        - 'confidence'    : calibrated probability of the class in
          pred_label specifically (or raw, if no calibrator given).
          This is the number to use for "how much to trust this call",
          not max(prob_down, prob_flat, prob_up).
    """
    X, _feat_df = prepare_model_input(df, feature_columns=feature_columns, has_volume=has_volume)
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    raw_proba = ensemble.predict_proba(X_valid)
    proba = calibrator.transform(raw_proba) if calibrator is not None else raw_proba

    # --- the fix: decide the class from RAW probabilities, never from
    # calibration's independently-fit-per-class-then-renormalized output
    # (confirmed on calib_val + final test to silently drop HOLD) ---
    pred_class = np.argmax(raw_proba, axis=1)
    confidence = proba[np.arange(len(pred_class)), pred_class]
    # --- end fix ---

    pred_label = pd.Series(pred_class).map(INVERSE_LABEL_MAP).values

    result = pd.DataFrame(
        {
            "pred_label": np.nan,
            "prob_down": np.nan,
            "prob_flat": np.nan,
            "prob_up": np.nan,
            "confidence": np.nan,
        },
        index=df.index,
    )
    valid_idx = X_valid.index
    result.loc[valid_idx, "pred_label"] = pred_label
    result.loc[valid_idx, "prob_down"] = proba[:, LABEL_MAP[-1]]
    result.loc[valid_idx, "prob_flat"] = proba[:, LABEL_MAP[0]]
    result.loc[valid_idx, "prob_up"] = proba[:, LABEL_MAP[1]]
    result.loc[valid_idx, "confidence"] = confidence

    return result