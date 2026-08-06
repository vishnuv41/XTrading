"""
Exponential Moving Average (EMA)

EMA weights recent candles more heavily than older ones, making it more
responsive to new price action than a simple moving average. Used as the
backbone of trend-following logic (e.g. EMA20 > EMA50 = uptrend).
"""

import pandas as pd


def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """
    Add an EMA column to the dataframe.

    Args:
        df: DataFrame with at least a `column` (default 'close') column.
        period: Lookback period (e.g. 20, 50, 200).
        column: Source column to calculate EMA on.

    Returns:
        df with a new column named e.g. 'EMA20'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    col_name = f"EMA{period}"
    df[col_name] = df[column].ewm(span=period, adjust=False).mean()
    return df


def calculate_multiple_emas(df: pd.DataFrame, periods: list[int] = [20, 50, 200],
                             column: str = "close") -> pd.DataFrame:
    """Convenience wrapper to add several EMAs at once (EMA20, EMA50, EMA200 etc)."""
    for period in periods:
        df = calculate_ema(df, period=period, column=column)
    return df


if __name__ == "__main__":
    # Quick self-test with synthetic data
    import numpy as np
    dummy = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})
    dummy = calculate_multiple_emas(dummy)
    print(dummy.tail())