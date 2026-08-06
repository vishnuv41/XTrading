"""
Average Directional Index (ADX)

Measures trend STRENGTH (not direction). Critical for regime detection:
ADX > 25 typically = trending market (trend-following strategies work),
ADX < 20 = ranging/choppy market (trend strategies tend to whipsaw and lose).
Also produces +DI/-DI which give directional bias.
"""

import pandas as pd
import numpy as np


def calculate_adx(df: pd.DataFrame, period: int = 14,
                   high_col: str = "high", low_col: str = "low",
                   close_col: str = "close") -> pd.DataFrame:
    """
    Add ADX, +DI, -DI columns (Wilder's original method).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: Lookback period (default 14).

    Returns:
        df with new columns: 'ADX14', 'PLUS_DI14', 'MINUS_DI14'.
    """
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    df[f"ADX{period}"] = adx
    df[f"PLUS_DI{period}"] = plus_di
    df[f"MINUS_DI{period}"] = minus_di
    return df


if __name__ == "__main__":
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
    })
    dummy = calculate_adx(dummy)
    print(dummy.tail())