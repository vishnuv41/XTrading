"""
Moving Average Convergence Divergence (MACD)

Trend + momentum indicator: difference between a fast and slow EMA (MACD
line), an EMA of that line (signal line), and their difference (histogram).
Bullish when MACD crosses above signal; bearish on the reverse.
"""

import pandas as pd
import numpy as np


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                    signal: int = 9, column: str = "close") -> pd.DataFrame:
    """
    Add MACD, MACD_signal, MACD_hist, and a categorical MACD_trend column.

    Args:
        df: DataFrame with at least a `column` (default 'close') column.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).
        column: Source column.

    Returns:
        df with new columns: 'MACD', 'MACD_signal', 'MACD_hist', 'MACD_trend'.
        MACD_trend is 'bullish' when MACD > signal, else 'bearish'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = histogram
    df["MACD_trend"] = np.where(macd_line > signal_line, "bullish", "bearish")
    return df


if __name__ == "__main__":
    dummy = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})
    dummy = calculate_macd(dummy)
    print(dummy.tail())