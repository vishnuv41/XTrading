"""
Cross-asset features
---------------------
Rolling correlation / beta / relative-strength features of the primary
symbol against one or more reference assets (e.g. BTC dominance for
altcoins, or any correlated pair). These help the model pick up
regime-level co-movement that single-asset indicators can't see.
"""

import numpy as np
import pandas as pd


def calculate_cross_asset_features(
    df: pd.DataFrame,
    reference_assets: dict,
    windows: list = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Append cross-asset correlation/beta/relative-strength features.

    Parameters
    ----------
    df : pd.DataFrame
        Primary asset OHLCV/feature dataframe with a `price_col` column,
        row-aligned by position (index doesn't need to match `reference_assets`,
        only length/order — if you have real timestamps, align them
        upstream before calling this).
    reference_assets : dict[str, pd.DataFrame]
        e.g. {"BTC": btc_df}. Each value must contain `price_col` and
        be the same length as `df` (reindex/merge_asof beforehand for
        real timestamp-aligned data).
    windows : list[int]
        Rolling windows for correlation/beta. Default [20, 50].

    Adds per reference asset `name`, per window `w`:
        CORR_{name}_{w}   : rolling correlation of returns
        BETA_{name}_{w}   : rolling beta (cov / var) of primary vs reference
        RS_{name}         : relative strength = primary cumulative return / reference cumulative return
    """
    df = df.copy()
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain column: {price_col}")

    if windows is None:
        windows = [20, 50]

    primary_ret = df[price_col].pct_change()

    for name, ref_df in reference_assets.items():
        if price_col not in ref_df.columns:
            raise ValueError(f"Reference asset '{name}' must contain column: {price_col}")

        ref_price = ref_df[price_col].reset_index(drop=True)
        ref_price.index = df.index[: len(ref_price)]
        ref_price = ref_price.reindex(df.index)
        ref_ret = ref_price.pct_change()

        for w in windows:
            df[f"CORR_{name}_{w}"] = primary_ret.rolling(w).corr(ref_ret)
            cov = primary_ret.rolling(w).cov(ref_ret)
            var = ref_ret.rolling(w).var()
            df[f"BETA_{name}_{w}"] = cov / var

        primary_cum = (1 + primary_ret.fillna(0)).cumprod()
        ref_cum = (1 + ref_ret.fillna(0)).cumprod()
        df[f"RS_{name}"] = primary_cum / ref_cum

    return df
