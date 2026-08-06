"""
SHAP analysis
----------------
Feature-attribution helpers for the tree models in ml/models/. Works
with either a raw fitted tree model or one of our wrapper classes
(XGBoostModel/LightGBMModel/CatBoostModel), which all expose `.model`
for the underlying fitted estimator.

Kept deliberately non-visual (returns arrays/DataFrames, not matplotlib
figures) so it works the same in a notebook, a script, or a pipeline
step that logs feature importances to a monitoring system.
"""

import numpy as np
import pandas as pd
import shap


def _unwrap(model):
    """Accept either a raw fitted model or one of our wrapper classes."""
    return model.model if hasattr(model, "model") else model


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = None):
    """
    Compute SHAP values for a tree-based model.

    Parameters
    ----------
    model : a fitted XGBoostModel/LightGBMModel/CatBoostModel wrapper,
        or a raw fitted tree model compatible with shap.TreeExplainer.
    X : feature DataFrame to explain.
    sample_size : if given, subsample X to this many rows before
        explaining (SHAP on large trees can be slow; a few thousand
        rows is usually enough for stable importance rankings).

    Returns
    -------
    (shap_values, X_used) : shap_values is either an ndarray
        [n_samples, n_features] (binary/regression) or a list of such
        arrays, one per class (multiclass) — matches shap's own output
        convention for TreeExplainer. X_used is the (possibly
        subsampled) DataFrame the values correspond to, for indexing.
    """
    raw_model = _unwrap(model)

    X_used = X
    if sample_size is not None and len(X) > sample_size:
        X_used = X.sample(sample_size, random_state=42)

    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X_used)

    return shap_values, X_used


def get_shap_feature_importance(shap_values, feature_names: list, class_index: int = None) -> pd.DataFrame:
    """
    Reduce raw SHAP values to a ranked mean-|SHAP| feature importance table.

    Parameters
    ----------
    shap_values : output of compute_shap_values (ndarray or list of ndarrays).
    feature_names : column names corresponding to the SHAP values' columns.
    class_index : for multiclass output (list of arrays), which class's
        SHAP values to summarize. If None and shap_values is a list,
        averages absolute importance across all classes.

    Returns
    -------
    DataFrame with columns ['feature', 'mean_abs_shap'], sorted descending.
    """
    if isinstance(shap_values, list):
        if class_index is not None:
            values = shap_values[class_index]
        else:
            values = np.mean([np.abs(v) for v in shap_values], axis=0)
            values = values  # already mean abs across classes
            mean_abs = values.mean(axis=0)
            return (
                pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
    else:
        values = shap_values

    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )