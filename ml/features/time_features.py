"""
Time features
---------------
Calendar + trading-session features. Crypto/FX trade 24/7, so session
overlap (Asia/EU/US) is a genuinely informative liquidity/volatility
proxy, not just a stocks-market artifact.

Expected input: DataFrame with a datetime column or DatetimeIndex.
Session boundaries are UTC hour ranges (adjust if your data isn't UTC):
    Asia    00:00-09:00 UTC
    Europe  07:00-16:00 UTC
    US      12:00-21:00 UTC
"""

import numpy as np
import pandas as pd


def calculate_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Append calendar/session features.

    Adds columns:
        HOUR, DAY_OF_WEEK, DAY_OF_MONTH, MONTH
        IS_WEEKEND
        HOUR_SIN, HOUR_COS               : cyclical encoding of hour-of-day
        DOW_SIN, DOW_COS                  : cyclical encoding of day-of-week
        SESSION_ASIA, SESSION_EUROPE, SESSION_US : 0/1 flags
        SESSION_OVERLAP_EU_US             : 0/1, the highest-liquidity window
        N_ACTIVE_SESSIONS                 : count of simultaneously active sessions
    """
    df = df.copy()

    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        raise ValueError(
            f"No '{timestamp_col}' column and index is not a DatetimeIndex; "
            "pass timestamp_col explicitly."
        )

    df["HOUR"] = ts.dt.hour.values
    df["DAY_OF_WEEK"] = ts.dt.dayofweek.values
    df["DAY_OF_MONTH"] = ts.dt.day.values
    df["MONTH"] = ts.dt.month.values
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)

    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7)
    df["DOW_COS"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7)

    h = df["HOUR"]
    df["SESSION_ASIA"] = ((h >= 0) & (h < 9)).astype(int)
    df["SESSION_EUROPE"] = ((h >= 7) & (h < 16)).astype(int)
    df["SESSION_US"] = ((h >= 12) & (h < 21)).astype(int)
    df["SESSION_OVERLAP_EU_US"] = ((h >= 12) & (h < 16)).astype(int)
    df["N_ACTIVE_SESSIONS"] = df["SESSION_ASIA"] + df["SESSION_EUROPE"] + df["SESSION_US"]

    return df