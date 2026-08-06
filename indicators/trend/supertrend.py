"""
Supertrend Indicator
---------------------
Trend-following overlay built on ATR bands. Flips direction when price
closes through the current band, making it a clean trend/regime signal
that plugs directly into the regime detector and strategy engine.

Expected input: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
(volume unused here but kept for a consistent OHLCV signature across the engine).
"""

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Compute the Supertrend indicator.

    Parameters
    ----------
    df : DataFrame with 'high', 'low', 'close' columns.
    period : ATR lookback window (default 10).
    multiplier : ATR multiplier controlling band width (default 3.0).

    Returns
    -------
    DataFrame (copy of input) with added columns:
        - 'supertrend'        : the active band level
        - 'supertrend_direction' : +1 (uptrend) / -1 (downtrend)
        - 'supertrend_upper'  : raw upper band (pre-flip)
        - 'supertrend_lower'  : raw lower band (pre-flip)
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    hl2 = (df["high"] + df["low"]) / 2
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    close = df["close"].values
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n)
    supertrend = np.full(n, np.nan)

    ub = upper_band.values
    lb = lower_band.values

    for i in range(n):
        if i == 0 or np.isnan(atr.iloc[i - 1]):
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            direction[i] = 1
            supertrend[i] = final_lower[i] if not np.isnan(final_lower[i]) else np.nan
            continue

        # band "sticking" logic: bands only move in the trend-confirming direction
        final_upper[i] = ub[i] if (ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = lb[i] if (lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]

        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    df["supertrend_upper"] = final_upper
    df["supertrend_lower"] = final_lower
    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction.astype(int)

    return df