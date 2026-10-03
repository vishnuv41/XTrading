"""
enhanced_ml/predict_enhanced.py
----------------------------------
ml.predict.predict() cannot be reused as-is for the enhanced (V2)
TP-before-SL model, for two concrete reasons:

  1. It decodes pred_class through the module-level, hardcoded 3-class
     `LABEL_MAP`/`INVERSE_LABEL_MAP` imported from ml.train, not the
     per-model label_map.json that load_training_artifacts() already
     loads correctly but predict() ignores. For V2's binary {0,1}
     labels this silently produces WRONG classes (e.g. class 1 would
     decode to V1's "0/flat" via the 3-class map).
  2. It unconditionally reads `proba[:, LABEL_MAP[1]]` for 'prob_up'
     — index 2 — which is out of bounds on a 2-column binary
     probability array and raises IndexError outright.

  3. Separately: V2 features include MTF columns (see
     enhanced_ml.train_enhanced.build_mtf_feature_fn) that
     ml.utils.preprocessing.prepare_model_input does not know to
     rebuild — it only reconstructs V1's base 115 features. Predicting
     with those alone would select a feature_columns set the loaded
     columns don't fully contain.

This module fixes both by: reading class semantics generically from
the ensemble's own `.classes_` (no hardcoded label map at all — V2's
label_map is already an identity {0:0, 1:1}, so no remapping is even
needed), and rebuilding features via the exact same feature_fn used at
V2 training time.

Does not modify ml/predict.py. load_training_artifacts() from that
module IS reused as-is (safe — it only reads model_dir contents and
its `num_class` default is irrelevant whenever ensemble.pkl exists,
which every enhanced_ml.train_enhanced run always writes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models import EnsembleModel
from ml.calibration import ProbabilityCalibrator
from enhanced_ml.train_enhanced import build_mtf_feature_fn


def predict_enhanced(
    df: pd.DataFrame,
    ensemble: EnsembleModel,
    feature_columns: list,
    symbol: str,
    higher_timeframes: list[str],
    exchange: str,
    calibrator: ProbabilityCalibrator = None,
    has_volume: bool = True,
) -> pd.DataFrame:
    """
    Compute V2 (MTF) features for `df` and return per-bar TP-before-SL
    predictions.

    Parameters
    ----------
    df : raw OHLCV DataFrame.
    ensemble : fitted EnsembleModel loaded from a V2 model_dir (e.g. via
        ml.predict.load_training_artifacts("models_artifacts/<SYM>_<tf>_v2")).
    feature_columns : exact training-time feature column order for this
        V2 model (artifacts['feature_columns']).
    symbol, higher_timeframes, exchange : MUST match what was passed to
        enhanced_ml.train_enhanced.run_enhanced_training when this model
        was trained, or the rebuilt MTF features won't match what the
        model was fit on. These aren't currently persisted in
        metadata.json (run_training_pipeline's generic metadata doesn't
        know about the enhanced adapter's extra params) — same
        known caveat confidence_accuracy.py already has for V1's
        pt_mult/sl_mult; keep your own record of what you trained with.
    calibrator : optional fitted ProbabilityCalibrator. As in
        ml.predict.predict(), the predicted class is always decided from
        RAW probabilities — calibrator only rescales the reported
        confidence number, never re-ranks the decision.
    has_volume : must match what the model was trained with.

    Returns
    -------
    DataFrame indexed like `df`, columns:
        - 'pred_label'     : 0 (SL reached first) or 1 (TP reached
                              first), decided from RAW ensemble
                              probabilities. NaN where features/warm-up
                              made a row unusable.
        - 'prob_sl_first'  : calibrated (if calibrator given) P(class=0)
        - 'prob_tp_first'  : calibrated (if calibrator given) P(class=1)
        - 'confidence'     : calibrated probability of pred_label
                              specifically — the number to threshold on,
                              not max(prob_sl_first, prob_tp_first).
    """
    feature_fn = build_mtf_feature_fn(symbol, higher_timeframes, exchange=exchange)
    feat_df = feature_fn(df, has_volume=has_volume)

    missing = [c for c in feature_columns if c not in feat_df.columns]
    if missing:
        raise ValueError(
            f"Rebuilt V2 features are missing {len(missing)} column(s) the model "
            f"was trained on (e.g. {missing[:5]}) — check symbol/higher_timeframes/"
            f"exchange match what was used to train this model_dir."
        )

    X = feat_df[feature_columns]
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    raw_proba = ensemble.predict_proba(X_valid)
    if raw_proba.shape[1] != 2:
        raise ValueError(
            f"predict_enhanced expects a binary (2-class) V2 model, got "
            f"{raw_proba.shape[1]} classes — is this actually a V1 model_dir?"
        )

    proba = calibrator.transform(raw_proba) if calibrator is not None else raw_proba

    # Decide from RAW probabilities, never from calibration's independently
    # -fit-per-class output — same fix/rationale as ml.predict.predict().
    pred_class = np.argmax(raw_proba, axis=1)
    confidence = proba[np.arange(len(pred_class)), pred_class]

    result = pd.DataFrame(
        {"pred_label": np.nan, "prob_sl_first": np.nan, "prob_tp_first": np.nan, "confidence": np.nan},
        index=df.index,
    )
    valid_idx = X_valid.index
    result.loc[valid_idx, "pred_label"] = pred_class
    result.loc[valid_idx, "prob_sl_first"] = proba[:, 0]
    result.loc[valid_idx, "prob_tp_first"] = proba[:, 1]
    result.loc[valid_idx, "confidence"] = confidence

    return result