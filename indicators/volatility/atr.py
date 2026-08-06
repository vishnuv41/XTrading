"""
Average True Range (ATR)

Volatility indicator (not directional). Used throughout the risk engine
to size stop-loss and take-profit distances relative to current volatility
instead of fixed pips/percent (e.g. SL = entry - 2*ATR).
"""

import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14,
                   high_col: str = "high", low_col: str = "low",
                   close_col: str = "close") -> pd.DataFrame:
    """
    Add an ATR column using Wilder's smoothing of True Range.

    True Range = max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close)
    )

    Args:
        df: DataFrame with 'high', 'low', 'close' columns (OHLCV standard).
        period: Lookback period (default 14).

    Returns:
        df with a new column named e.g. 'ATR14'.
    """
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    prev_close = df[close_col].shift(1)

    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - prev_close).abs()
    tr3 = (df[low_col] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    col_name = f"ATR{period}"
    df[col_name] = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return df


if __name__ == "__main__":
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
    })
    dummy = calculate_atr(dummy)
    print(dummy.tail())