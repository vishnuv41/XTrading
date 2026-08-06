"""
Keltner Channels
------------------
ATR-based volatility bands around an EMA midline. Similar role to
Bollinger Bands but ATR-driven rather than std-dev-driven; useful for
squeeze detection (Keltner vs Bollinger width) in the regime detector.

Expected input: DataFrame with columns ['high', 'low', 'close'].
"""

import pandas as pd


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_keltner(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Keltner Channels.

    Parameters
    ----------
    ema_period : EMA window for the midline (default 20).
    atr_period : ATR window used for band width (default 10).
    multiplier : ATR multiplier controlling channel width (default 2.0).

    Returns
    -------
    DataFrame (copy of input) with added columns:
        - 'keltner_mid'
        - 'keltner_upper'
        - 'keltner_lower'
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    mid = df["close"].ewm(span=ema_period, adjust=False).mean()
    atr = _atr(df, atr_period)

    df["keltner_mid"] = mid
    df["keltner_upper"] = mid + multiplier * atr
    df["keltner_lower"] = mid - multiplier * atr

    return df