"""
Williams %R
-------------
Momentum oscillator, essentially an inverted/rescaled Stochastic %K.
Bounded [-100, 0]; readings near 0 = overbought, near -100 = oversold.

Expected input: DataFrame with columns ['high', 'low', 'close'].
"""

import pandas as pd


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Williams %R.

    Parameters
    ----------
    period : lookback window (default 14).

    Returns
    -------
    DataFrame (copy of input) with added column:
        - 'williams_r', bounded [-100, 0]
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()

    df["williams_r"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    return df