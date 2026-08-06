"""
Price features
----------------
Pure price-derived ML features: returns, momentum, candle shape, and
distance-to-extremes. Assumes 'open','high','low','close' are already
in df; safe to call after indicators.calculate_all_indicators (will
also pick up EMA/SMA columns if present to build price-vs-MA features).
"""

import numpy as np
import pandas as pd


def calculate_price_features(
    df: pd.DataFrame,
    return_periods: list = None,
    momentum_periods: list = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Append price-derived features.

    Parameters
    ----------
    return_periods : lookback periods for simple/log returns (default [1,3,5,10,20]).
    momentum_periods : lookback periods for momentum/ROC (default [5,10,20,50]).

    Adds columns:
        RET_{n}, LOGRET_{n}           : n-bar simple / log returns
        MOM_{n}                        : n-bar rate of change (%)
        CANDLE_BODY, CANDLE_RANGE      : |close-open|, high-low
        CANDLE_BODY_RATIO              : body / range (0 = doji, 1 = full range body)
        UPPER_WICK_RATIO, LOWER_WICK_RATIO
        GAP                             : (open - prev close) / prev close
        DIST_FROM_HIGH_{n}, DIST_FROM_LOW_{n} : distance to rolling n-bar high/low (%)
        PRICE_VS_EMA{p}, PRICE_VS_SMA{p} : % distance from each EMA/SMA column
            already present in df (auto-detected, skipped if none found)
    """
    df = df.copy()
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if return_periods is None:
        return_periods = [1, 3, 5, 10, 20]
    if momentum_periods is None:
        momentum_periods = [5, 10, 20, 50]

    close = df[price_col]

    for n in return_periods:
        df[f"RET_{n}"] = close.pct_change(n)
        df[f"LOGRET_{n}"] = np.log(close / close.shift(n))

    for n in momentum_periods:
        df[f"MOM_{n}"] = (close / close.shift(n) - 1) * 100

    # candle shape
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["CANDLE_BODY"] = body
    df["CANDLE_RANGE"] = df["high"] - df["low"]
    df["CANDLE_BODY_RATIO"] = body / rng
    df["UPPER_WICK_RATIO"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["LOWER_WICK_RATIO"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    # gap from prior close
    prev_close = close.shift(1)
    df["GAP"] = (df["open"] - prev_close) / prev_close

    # distance from rolling extremes
    for n in [10, 20, 50]:
        roll_high = df["high"].rolling(n).max()
        roll_low = df["low"].rolling(n).min()
        df[f"DIST_FROM_HIGH_{n}"] = (close - roll_high) / roll_high * 100
        df[f"DIST_FROM_LOW_{n}"] = (close - roll_low) / roll_low * 100

    # price vs any moving-average columns already computed upstream
    ma_cols = [c for c in df.columns if c.startswith("EMA") or c.startswith("SMA")]
    for col in ma_cols:
        df[f"PRICE_VS_{col}"] = (close - df[col]) / df[col] * 100

    return df