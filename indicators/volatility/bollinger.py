"""
Bollinger Bands

Volatility bands around an SMA (typically 20-period, 2 std dev). Price
touching/exceeding the upper or lower band signals relative extension;
band width (squeeze) signals impending volatility expansion — useful as
a regime feature (tight bands = low-vol regime, likely breakout ahead).
"""

import pandas as pd


def calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                         column: str = "close") -> pd.DataFrame:
    """
    Add Bollinger Band columns.

    Args:
        df: DataFrame with at least a `column` (default 'close') column.
        period: SMA lookback period (default 20).
        std_dev: Number of standard deviations for the bands (default 2).
        column: Source column.

    Returns:
        df with new columns: 'BB_middle', 'BB_upper', 'BB_lower',
        'BB_width' (upper-lower normalized by middle, a squeeze/volatility
        feature), and 'BB_pct' (%B: where price sits within the bands, 0-1+).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    middle = df[column].rolling(window=period, min_periods=period).mean()
    rolling_std = df[column].rolling(window=period, min_periods=period).std()

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    df["BB_middle"] = middle
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    df["BB_width"] = (upper - lower) / middle
    df["BB_pct"] = (df[column] - lower) / (upper - lower)
    return df


if __name__ == "__main__":
    import numpy as np
    dummy = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})
    dummy = calculate_bollinger(dummy)
    print(dummy.tail())