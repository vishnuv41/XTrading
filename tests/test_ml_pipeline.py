"""
Smoke test for the full ml/ pipeline: features -> labels -> purged CV
stacking ensemble -> calibration -> predict -> backtest.
Run: python -m tests.test_ml_pipeline   (from person2/)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ml.train import run_training_pipeline, build_feature_matrix, _make_synthetic_ohlcv
from ml.features import calculate_cross_asset_features
from ml.labeling import triple_barrier_labels, generate_meta_labels
from ml.validation import walk_forward_splits, PurgedKFold
from ml.predict import predict
from ml.backtest import run_backtest
from ml.explainability import compute_shap_values, get_shap_feature_importance


def main():
    df = _make_synthetic_ohlcv(n=1500, seed=11)

    print("=== [1] Feature engineering ===")
    feat_df = build_feature_matrix(df)
    print(f"Shape: {feat_df.shape}")
    assert "RET_1" in feat_df.columns
    assert "VOLUME_ZSCORE_20" in feat_df.columns
    assert "REALIZED_VOL_20" in feat_df.columns
    assert "SESSION_ASIA" in feat_df.columns

    print("\n=== [2] Cross-asset features ===")
    ref_df = _make_synthetic_ohlcv(n=1500, seed=99)
    cross_df = calculate_cross_asset_features(feat_df, {"BTC": ref_df})
    assert "CORR_BTC_20" in cross_df.columns
    assert "BETA_BTC_20" in cross_df.columns
    print("Cross-asset columns OK:", [c for c in cross_df.columns if "BTC" in c])

    print("\n=== [3] Triple-barrier labeling ===")
    labels = triple_barrier_labels(feat_df, volatility=feat_df["ATR14"] / feat_df["close"])
    print(labels["touch_type"].value_counts())
    assert labels["label"].dropna().isin([-1, 0, 1]).all()

    print("\n=== [4] Meta-labeling ===")
    fake_side = pd.Series(np.sign(np.random.default_rng(0).normal(size=len(labels))), index=labels.index)
    meta = generate_meta_labels(fake_side, labels)
    assert meta["meta_label"].dropna().isin([0, 1]).all()
    print("Meta-label distribution:", meta["meta_label"].value_counts().to_dict())

    print("\n=== [5] Validation splitters ===")
    n = 500
    wf_folds = list(walk_forward_splits(n, n_splits=4, gap=20))
    assert len(wf_folds) > 0
    for tr, te in wf_folds:
        assert tr.max() < te.min(), "walk-forward leaked: train overlaps/after test"
    print(f"Walk-forward: {len(wf_folds)} folds OK, no train/test overlap")

    t1 = pd.Series(np.minimum(np.arange(n) + 15, n - 1))
    pkf = PurgedKFold(n_splits=4, t1=t1, pct_embargo=0.02)
    pkf_folds = list(pkf.split(np.zeros((n, 1))))
    for tr, te in pkf_folds:
        assert len(set(tr) & set(te)) == 0, "purged K-fold leaked: train/test overlap"
    print(f"Purged K-fold: {len(pkf_folds)} folds OK, no train/test index overlap")

    import tempfile
    smoke_out_dir = tempfile.mkdtemp(prefix="ml_smoke_test_")
    result = run_training_pipeline(df, n_splits=3, output_dir=smoke_out_dir)
    print("Metrics:", result["metrics"])
    assert 0 <= result["metrics"]["holdout_accuracy"] <= 1

    print("\n=== [7] Predict on new data ===")
    new_df = _make_synthetic_ohlcv(n=300, seed=123)
    preds = predict(
        new_df,
        result["ensemble"],
        result["X_columns"],
        calibrator=result["calibrator"],
    )
    print(preds.dropna().tail(3))
    assert preds["pred_label"].dropna().isin([-1, 0, 1]).all()
    valid_conf = preds["confidence"].dropna()
    assert valid_conf.between(0, 1).all()

    print("\n=== [8] Backtest ===")
    bt = run_backtest(new_df, preds, max_holding=15, confidence_threshold=0.35)
    print("Backtest metrics:", bt["metrics"])
    assert "sharpe_ratio" in bt["metrics"]
    assert len(bt["equity_curve"]) == len(new_df)

    print("\n=== [9] SHAP explainability (xgboost base model) ===")
    xgb_wrapper = result["ensemble"].fitted_base_models_["xgboost"]
    X_sample = new_df.copy()
    X_sample = build_feature_matrix(X_sample)[result["X_columns"]].dropna()
    shap_values, X_used = compute_shap_values(xgb_wrapper, X_sample, sample_size=100)
    importance = get_shap_feature_importance(shap_values, result["X_columns"])
    print(importance.head(10))
    assert len(importance) == len(result["X_columns"])

    print("\nALL ML PIPELINE CHECKS PASSED.")


def test_ml_pipeline_smoke():
    """pytest entry point — wraps main() so `pytest tests/` discovers this."""
    main()


if __name__ == "__main__":
    main()