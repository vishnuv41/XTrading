"""
Relative Strength Index (RSI)

Momentum oscillator (0-100) measuring the speed/magnitude of recent price
changes. >70 conventionally "overbought", <30 "oversold". Used here mainly
as a momentum confirmation filter (e.g. RSI > 55 supports a BUY signal)
rather than pure reversal trading, which tends to fail in strong trends.
"""

import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """
    Add an RSI column using Wilder's smoothing method (the standard, not the
    naive simple-average version which drifts from what most charting
    platforms show).

    Args:
        df: DataFrame with at least a `column` (default 'close') column.
        period: Lookback period (default 14).
        column: Source column.

    Returns:
        df with a new column named e.g. 'RSI14'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    delta = df[column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (all gains), RSI should be 100
    rsi = rsi.where(avg_loss != 0, 100.0)

    col_name = f"RSI{period}"
    df[col_name] = rsi
    return df


if __name__ == "__main__":
    dummy = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})
    dummy = calculate_rsi(dummy)
    print(dummy.tail())