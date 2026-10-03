"""
Single shared entry point for turning a raw OHLCV DataFrame into the
exact feature representation the models expect.

Before this module existed, the "run indicators + features, then encode
the MACD_trend string column" logic was duplicated across ml/train.py,
ml/predict.py and ml/explainability/shap_analysis.py usage sites. Any
change to that logic (e.g. adding a new categorical column that needs
encoding) had to be made in every copy, and it was easy for them to drift
out of sync. Now everyone imports from here instead.

Two entry points:

build_feature_matrix(df)
    -> full engineered DataFrame (indicators + ML features +
       encoded categoricals), still containing OHLCV/meta columns.

prepare_model_input(df)
    -> build_feature_matrix(df) plus column selection:
       either the exact training-time feature_columns (for inference)
       or every non-metadata/non-price-level column (for training),
       ready to hand to a model.
"""

import pandas as pd

from indicators import calculate_all_indicators
from ml.features import (
    calculate_price_features,
    calculate_volume_features,
    calculate_volatility_features,
    calculate_time_features,
)


# ---------------------------------------------------------------------------
# Columns that must NEVER be fed directly to the ML model.
# ---------------------------------------------------------------------------
#
# These are:
#   - raw OHLCV columns
#   - timestamps
#   - labeling artifacts
#   - non-numeric categorical strings
#
NON_FEATURE_COLS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "touch_type",
    "touch_idx",
    "ret",
    "barrier_upper",
    "barrier_lower",
    "holding_bars",
    "MACD_trend",  # string version; numeric MACD_trend_bull is used instead
}


# ---------------------------------------------------------------------------
# Absolute price-level features
# ---------------------------------------------------------------------------
#
# These features contain the absolute BTC price level.
#
# Example:
#
#     EMA20      = 95,000
#     EMA20      = 120,000
#
# The model can therefore learn relationships that are specific to the
# historical BTC price regime instead of learning more general relationships
# such as:
#
#     price relative to EMA
#     price relative to VWAP
#     volatility
#     momentum
#     trend strength
#
# This is particularly important for time-series validation because the
# BTC price regime in the training period can be very different from the
# final test period.
#
# We keep the normalized/relative versions such as:
#
#     PRICE_VS_EMA20
#     PRICE_VS_EMA50
#     PRICE_VS_EMA200
#     VWAP_DEVIATION
#     ATR_NORM
#
# because those describe relationships rather than absolute price level.
#
PRICE_LEVEL_FEATURES = {
    # Moving averages
    "EMA20",
    "EMA50",
    "EMA200",
    "SMA20",
    "SMA50",
    "SMA200",

    # Supertrend absolute price levels
    "supertrend_upper",
    "supertrend_lower",
    "supertrend",

    # Ichimoku absolute price levels
    "ichimoku_tenkan",
    "ichimoku_kijun",
    "ichimoku_senkou_a",
    "ichimoku_senkou_b",
    "ichimoku_chikou",

    # Bollinger absolute price levels
    "BB_middle",
    "BB_upper",
    "BB_lower",

    # Keltner absolute price levels
    "keltner_mid",
    "keltner_upper",
    "keltner_lower",

    # Donchian absolute price levels
    "donchian_upper",
    "donchian_lower",
    "donchian_mid",

    # VWAP absolute price level
    "VWAP",
}


def build_feature_matrix(
    df: pd.DataFrame,
    has_volume: bool = True,
) -> pd.DataFrame:
    """
    Run the full indicator + feature stack on a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.

    has_volume : bool
        Whether volume-based features should be calculated.

    Returns
    -------
    pd.DataFrame
        Full engineered DataFrame containing:

        - OHLCV
        - indicators
        - ML features
        - encoded categorical features
        - metadata columns
    """

    # ------------------------------------------------------------------
    # 1. Technical indicators
    # ------------------------------------------------------------------
    df = calculate_all_indicators(
        df,
        has_volume=has_volume,
    )

    # ------------------------------------------------------------------
    # 2. Price features
    # ------------------------------------------------------------------
    df = calculate_price_features(df)

    # ------------------------------------------------------------------
    # 3. Volume features
    # ------------------------------------------------------------------
    if has_volume:
        df = calculate_volume_features(df)

    # ------------------------------------------------------------------
    # 4. Volatility features
    # ------------------------------------------------------------------
    df = calculate_volatility_features(df)

    # ------------------------------------------------------------------
    # 5. Time/session features
    # ------------------------------------------------------------------
    df = calculate_time_features(df)

    # ------------------------------------------------------------------
    # 6. Encode MACD_trend
    # ------------------------------------------------------------------
    #
    # MACD_trend is a categorical string:
    #
    #     "bullish"
    #     "bearish"
    #
    # Boosting models cannot directly consume this string column, so
    # create a numeric representation.
    #
    # The original MACD_trend column remains available for the strategy
    # layer, but is excluded from the ML feature matrix by
    # NON_FEATURE_COLS.
    #
    if "MACD_trend" in df.columns:
        df["MACD_trend_bull"] = (
            df["MACD_trend"] == "bullish"
        ).astype(int)

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
    df : pd.DataFrame
        Raw OHLCV DataFrame.

    feature_columns : list, optional
        Exact training-time feature column order to select.

        This is normally loaded from:

            models_artifacts/feature_columns.pkl

        when doing inference.

        If supplied, the function checks that every requested feature
        exists. Missing features cause a ValueError instead of silently
        producing a mismatched model input.

        This protects against training/inference feature drift.

    has_volume : bool
        Must match the setting used when the model was trained.

    Returns
    -------
    X : pd.DataFrame
        Model-ready feature matrix.

    feat_df : pd.DataFrame
        Full engineered DataFrame, including OHLCV/meta columns.

    Notes
    -----
    When feature_columns is supplied:

        X = exact saved training feature set

    This path is used for inference.

    When feature_columns is None:

        X = every engineered feature except:

            NON_FEATURE_COLS
            PRICE_LEVEL_FEATURES

    This path is used when creating a NEW training dataset.
    """

    # ------------------------------------------------------------------
    # Build the complete feature matrix first.
    # ------------------------------------------------------------------
    feat_df = build_feature_matrix(
        df,
        has_volume=has_volume,
    )

    # ------------------------------------------------------------------
    # INFERENCE PATH
    # ------------------------------------------------------------------
    #
    # If an already-trained model provides its exact feature list,
    # ALWAYS respect that list and its order.
    #
    # This is important because changing the global feature-selection
    # rules must not silently break an existing trained model.
    #
    if feature_columns is not None:

        missing = [
            c
            for c in feature_columns
            if c not in feat_df.columns
        ]

        if missing:
            raise ValueError(
                "Feature columns missing from computed features: "
                f"{missing}"
            )

        X = feat_df[feature_columns]

    # ------------------------------------------------------------------
    # NEW TRAINING PATH
    # ------------------------------------------------------------------
    #
    # No saved feature list means we are constructing the feature set
    # for a new model.
    #
    # Remove:
    #
    #   1. raw/meta/label columns
    #   2. absolute BTC price-level features
    #
    # Keep normalized/relative features such as:
    #
    #   PRICE_VS_EMA20
    #   VWAP_DEVIATION
    #   ATR_NORM
    #   BB_WIDTH_ZSCORE_10
    #
    else:

        cols = [
            c
            for c in feat_df.columns
            if c not in NON_FEATURE_COLS
            and c not in PRICE_LEVEL_FEATURES
        ]

        X = feat_df[cols]

    return X, feat_df