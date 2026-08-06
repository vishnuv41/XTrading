"""
Stochastic Oscillator
-----------------------
Momentum oscillator comparing close to its recent high/low range.
Bounded 0-100, useful for overbought/oversold conditions and %K/%D
crossover signals in the strategy engine.

Expected input: DataFrame with columns ['high', 'low', 'close'].
"""

import pandas as pd


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Compute the (slow) Stochastic Oscillator.

    Parameters
    ----------
    k_period : lookback window for the raw %K (default 14).
    smooth_k : smoothing applied to raw %K to get the final %K line
        (default 3; set to 1 for the "fast" stochastic).
    d_period : SMA window applied to %K to get %D (the signal line).

    Returns
    -------
    DataFrame (copy of input) with added columns:
        - 'stoch_k' : smoothed %K, bounded [0, 100]
        - 'stoch_d' : %D signal line (SMA of %K)
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()

    raw_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean() if smooth_k > 1 else raw_k
    d = k.rolling(d_period).mean()

    df["stoch_k"] = k
    df["stoch_d"] = d

    return df