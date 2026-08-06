"""
Simple Moving Average (SMA)

Unweighted rolling average. Useful as a slower, smoother reference line
against EMA (e.g. detecting when EMA crosses above/below SMA), and for
classic support/resistance levels like SMA200.
"""

import pandas as pd


def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """
    Add an SMA column to the dataframe.

    Args:
        df: DataFrame with at least a `column` (default 'close') column.
        period: Lookback period.
        column: Source column to calculate SMA on.

    Returns:
        df with a new column named e.g. 'SMA20'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    col_name = f"SMA{period}"
    df[col_name] = df[column].rolling(window=period, min_periods=period).mean()
    return df


def calculate_multiple_smas(df: pd.DataFrame, periods: list[int] = [20, 50, 200],
                             column: str = "close") -> pd.DataFrame:
    for period in periods:
        df = calculate_sma(df, period=period, column=column)
    return df


if __name__ == "__main__":
    import numpy as np
    dummy = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})
    dummy = calculate_multiple_smas(dummy)
    print(dummy.tail())