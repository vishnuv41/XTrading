"""
Money Flow Index (MFI)

"Volume-weighted RSI" — same overbought/oversold interpretation as RSI
(>80 / <20) but incorporates volume via typical price * volume as the
raw money flow, split into positive/negative flow based on whether
typical price rose or fell candle-to-candle. Catches momentum shifts
that are volume-confirmed, which tend to be more reliable than price-
only momentum.
"""

import pandas as pd
import numpy as np


def calculate_mfi(df: pd.DataFrame, period: int = 14,
                   high_col: str = "high", low_col: str = "low",
                   close_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    """
    Add an MFI column.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        period: Lookback period (default 14).

    Returns:
        df with a new column named e.g. 'MFI14'.
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    raw_money_flow = typical_price * df[volume_col]

    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    positive_sum = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=period).sum()

    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi = mfi.where(negative_sum != 0, 100.0)

    col_name = f"MFI{period}"
    df[col_name] = mfi
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
    dummy = calculate_mfi(dummy)
    print(dummy.tail())