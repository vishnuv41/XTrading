"""
ml/utils/preprocessing.py
-----------------------------
Single shared entry point for turning a raw OHLCV DataFrame into the
exact feature representation the models expect.

Before this module existed, the "run indicators + features, then encode
the MACD_trend string column" logic was duplicated across ml/train.py,
ml/predict.py and ml/explainability/shap_analysis.py usage sites. Any
change to that logic (e.g. adding a new categorical column that needs
encoding) had to be made in every copy, and it was easy for them to
drift out of sync. Now everyone imports from here instead.

Two entry points:
    build_feature_matrix(df)   -> full engineered DataFrame (indicators
                                   + ml features + encoded categoricals),
                                   still containing OHLCV/meta columns.
    prepare_model_input(df)    -> build_feature_matrix(df) plus column
                                   selection: either the exact training-
                                   time `feature_columns` (for inference)
                                   or every non-metadata column (for
                                   training), ready to hand to a model.
"""

import pandas as pd

from indicators import calculate_all_indicators
from ml.features import (
    calculate_price_features,
    calculate_volume_features,
    calculate_volatility_features,
    calculate_time_features,
)


# Columns that are raw OHLCV, labeling artifacts, or non-numeric
# passthroughs — never fed to the models as features.
NON_FEATURE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume",
    "label", "touch_type", "touch_idx", "ret", "barrier_upper",
    "barrier_lower", "holding_bars",
    "MACD_trend",  # string version; numeric MACD_trend_bull is used instead
}


def build_feature_matrix(df: pd.DataFrame, has_volume: bool = True) -> pd.DataFrame:
    """Run the full indicator + feature stack on a raw OHLCV DataFrame."""
    df = calculate_all_indicators(df, has_volume=has_volume)
    df = calculate_price_features(df)
    if has_volume:
        df = calculate_volume_features(df)
    df = calculate_volatility_features(df)
    df = calculate_time_features(df)

    # MACD_trend is a categorical string ('bullish'/'bearish') kept for the
    # strategy layer's readability (strategy/trend.py, entry.py, exit.py).
    # Boosting libraries reject string dtypes, so encode it numerically here
    # instead of losing the signal. The raw string column is excluded from
    # the feature matrix via NON_FEATURE_COLS above.
    if "MACD_trend" in df.columns:
        df["MACD_trend_bull"] = (df["MACD_trend"] == "bullish").astype(int)

    return df


def prepare_model_input(
    df: pd.DataFrame,
    feature_columns: list = None,
    has_volume: bool = True,
):
    """
    Build the feature matrix and select the model-ready column set.

    Parameters
    ----------
    df : raw OHLCV DataFrame.
    feature_columns : exact training-time feature column order to select
        (e.g. the persisted `feature_columns.pkl`). If given, raises a
        ValueError listing any columns that couldn't be computed rather
        than silently returning a mismatched matrix — this is what makes
        inference-time feature drift fail loudly instead of feeding the
        model garbage. If None, every column not in NON_FEATURE_COLS is
        returned (the training-time "give me everything" case).
    has_volume : must match what the model was/will be trained with.

    Returns
    -------
    (X, feat_df) : X is the model-ready DataFrame (selected/ordered
        columns, NaN rows still included); feat_df is the full engineered
        DataFrame (indicators, features, and any metadata columns) for
        callers that need it (e.g. label alignment during training).
    """
    feat_df = build_feature_matrix(df, has_volume=has_volume)

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in feat_df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from computed features: {missing}")
        X = feat_df[feature_columns]
    else:
        cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
        X = feat_df[cols]

    return X, feat_df
