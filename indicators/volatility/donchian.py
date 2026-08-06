"""
Donchian Channels
-------------------
Pure price-extremes channel (highest high / lowest low over N bars).
Classic breakout-system input; also useful as a simple range/regime
feature since channel width tracks realized volatility.

Expected input: DataFrame with columns ['high', 'low'].
"""

import pandas as pd


def calculate_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute Donchian Channels.

    Parameters
    ----------
    period : lookback window (default 20).

    Returns
    -------
    DataFrame (copy of input) with added columns:
        - 'donchian_upper'  : highest high over `period` bars
        - 'donchian_lower'  : lowest low over `period` bars
        - 'donchian_mid'    : midpoint of upper/lower
        - 'donchian_width'  : upper - lower (volatility/range proxy)
    """
    df = df.copy()
    required = {"high", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()

    df["donchian_upper"] = upper
    df["donchian_lower"] = lower
    df["donchian_mid"] = (upper + lower) / 2
    df["donchian_width"] = upper - lower

    return df