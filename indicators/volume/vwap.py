"""
Volume Weighted Average Price (VWAP)

Average price weighted by volume, typically reset per session/day. Widely
used as a fair-value reference — price above VWAP suggests buyers in
control for the session, below suggests sellers. Requires a session/date
grouping key since VWAP conventionally resets (it is NOT a rolling
indicator over the whole history, unlike EMA/SMA).
"""

import pandas as pd


def calculate_vwap(df: pd.DataFrame, high_col: str = "high", low_col: str = "low",
                    close_col: str = "close", volume_col: str = "volume",
                    session_col: str | None = None) -> pd.DataFrame:
    """
    Add a VWAP column.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        session_col: Optional column (e.g. a date string) to reset VWAP per
            session/day. If None, VWAP is cumulative over the entire
            dataframe (fine for a single day's data, not for multi-day).

    Returns:
        df with a new column 'VWAP'.
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    tp_vol = typical_price * df[volume_col]

    if session_col is not None:
        if session_col not in df.columns:
            raise ValueError(f"Session column '{session_col}' not found in dataframe")
        cum_tp_vol = tp_vol.groupby(df[session_col]).cumsum()
        cum_vol = df[volume_col].groupby(df[session_col]).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = df[volume_col].cumsum()

    df["VWAP"] = cum_tp_vol / cum_vol
    return df


if __name__ == "__main__":
    import numpy as np
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
        "volume": np.random.randint(100, 1000, n),
    })
    dummy = calculate_vwap(dummy)
    print(dummy.tail())