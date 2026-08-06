"""
ml/predict.py
----------------
Loads trained base models, rebuilds the feature stack for new OHLCV
data, and returns calibrated probabilities + a confidence-scored
directional call per bar. This is the module inference/realtime_pipeline.py
should import rather than duplicating feature/model logic.
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
    calibrator : optional fitted ProbabilityCalibrator; if given, output
        probabilities are calibrated.
    has_volume : must match what the model was trained with.

    Returns
    -------
    DataFrame indexed like `df`, columns:
        - 'pred_label'    : predicted class, remapped back to {-1, 0, 1}
        - 'prob_down', 'prob_flat', 'prob_up' : per-class probability
        - 'confidence'    : max class probability (0-1)
    """
    X, _feat_df = prepare_model_input(df, feature_columns=feature_columns, has_volume=has_volume)
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    raw_proba = ensemble.predict_proba(X_valid)
    proba = calibrator.transform(raw_proba) if calibrator is not None else raw_proba

    pred_class = np.argmax(proba, axis=1)
    pred_label = pd.Series(pred_class).map(INVERSE_LABEL_MAP).values
    confidence = proba.max(axis=1)

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