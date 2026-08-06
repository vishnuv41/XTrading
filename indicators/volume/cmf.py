"""
Chaikin Money Flow (CMF)

Volume-weighted average of the Accumulation/Distribution multiplier over
a rolling window. Measures buying vs selling pressure: positive CMF =
accumulation (buying pressure dominant), negative = distribution. More
volume-sensitive than OBV since it weights each candle by *where* the
close sits within its own high-low range, not just its sign.
"""

import pandas as pd
import numpy as np


def calculate_cmf(df: pd.DataFrame, period: int = 20,
                   high_col: str = "high", low_col: str = "low",
                   close_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    """
    Add a CMF column.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        period: Rolling window (default 20).

    Returns:
        df with a new column named e.g. 'CMF20'.
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    high, low, close, volume = df[high_col], df[low_col], df[close_col], df[volume_col]

    hl_range = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume

    col_name = f"CMF{period}"
    df[col_name] = (
        mf_volume.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum()
    )
    return df


if __name__ == "__main__":
    n = 100
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "high": close + rng.random(n),
        "low": close - rng.random(n),
        "volume": rng.integers(100, 1000, n),
    })
    dummy = calculate_cmf(dummy)
    print(dummy.tail())