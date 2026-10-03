"""
enhanced_ml/features/feature_engine.py
------------------------------------------
FreqAI concept borrowed: a registry that expands a base feature set by
running additional feature-generating functions over it, rather than
one hardcoded pipeline you have to edit every time a new feature group
(multi-timeframe, lagged candles, ...) is added.

This does NOT reimplement indicators. The base feature set is exactly
V1's ml.utils.preprocessing.build_feature_matrix() output — same 115
features, same code path, same behavior. This module only adds an
extension point on top of it.

enhanced_ml/features/mtf_features.py and lagged_features.py (later
phases) will register themselves here rather than being wired in by
hand each time.
"""

from typing import Callable, List

import pandas as pd

from ml.utils.preprocessing import build_feature_matrix


class FeatureEngine:
    """
    Usage:
        engine = FeatureEngine()
        engine.register(some_extra_feature_fn)   # optional, none needed yet
        features_df = engine.build(raw_ohlcv_df, has_volume=True)

    Each registered function must accept the DataFrame returned by the
    previous step and return a DataFrame with the same index plus
    additional columns (or the same df with columns added in place then
    returned) — never fewer rows, never a different index, so features
    from different generators stay aligned.
    """

    def __init__(self):
        self._extra_feature_fns: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

    def register(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "FeatureEngine":
        """Add an extra feature-generating function to the pipeline. Returns
        self so calls can be chained: engine.register(a).register(b)."""
        self._extra_feature_fns.append(fn)
        return self

    def build(self, df: pd.DataFrame, has_volume: bool = True) -> pd.DataFrame:
        """
        Run V1's base feature pipeline, then any registered V2 extras
        in the order they were registered.
        """
        n_before = len(df)
        features = build_feature_matrix(df, has_volume=has_volume)

        for fn in self._extra_feature_fns:
            features = fn(features)
            if len(features) != n_before:
                raise ValueError(
                    f"{fn.__name__} changed row count from {n_before} to {len(features)} — "
                    f"feature generators must preserve the index (append columns, don't drop rows)."
                )

        return features