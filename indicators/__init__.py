"""
indicators package

Single entry point: calculate_all_indicators(df) runs the full indicator
suite and returns an enriched dataframe. Downstream modules (strategy/,
ml/features/) should import from here rather than calling individual
indicator files directly, so adding/removing an indicator only requires
a change in this one place.

Indicators are organized by category:
    trend/          EMA, SMA, MACD, ADX, Supertrend, Ichimoku
    momentum/       RSI, Stochastic, CCI, Williams %R
    volatility/     ATR, Bollinger, Keltner, Donchian
    volume/         VWAP, OBV, CMF, MFI
    microstructure/ (reserved for future order-flow/tape features)
"""

import pandas as pd

from .trend import (
    calculate_ema, calculate_multiple_emas,
    calculate_sma, calculate_multiple_smas,
    calculate_adx,
    calculate_supertrend,
    calculate_ichimoku,
)
from .momentum import (
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
    calculate_cci,
    calculate_williams_r,
)
from .volatility import (
    calculate_atr,
    calculate_bollinger,
    calculate_keltner,
    calculate_donchian,
)
from .volume import (
    calculate_vwap,
    calculate_obv,
    calculate_cmf,
    calculate_mfi,
)

__all__ = [
    "calculate_ema", "calculate_multiple_emas",
    "calculate_sma", "calculate_multiple_smas",
    "calculate_macd",
    "calculate_adx",
    "calculate_supertrend",
    "calculate_ichimoku",
    "calculate_rsi",
    "calculate_stochastic",
    "calculate_cci",
    "calculate_williams_r",
    "calculate_atr",
    "calculate_bollinger",
    "calculate_keltner",
    "calculate_donchian",
    "calculate_vwap",
    "calculate_obv",
    "calculate_cmf",
    "calculate_mfi",
    "calculate_all_indicators",
]


def calculate_all_indicators(
    df: pd.DataFrame,
    has_volume: bool = True,
    session_col: str = None,
) -> pd.DataFrame:
    """
    Run the full indicator suite on an OHLCV dataframe.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns and
            (if has_volume=True) a 'volume' column. Expected to come
            straight from Person 1's PostgreSQL query, sorted ascending
            by timestamp.
        has_volume: Set False if volume data isn't available yet
            (skips VWAP/OBV/CMF/MFI rather than failing).
        session_col: Optional date/session key column so VWAP resets
            daily instead of accumulating across the whole history.
            Ignored if has_volume=False.

    Returns:
        df with all indicator columns appended. NaNs will appear in the
        first `period` rows of each indicator (warm-up window), and
        Ichimoku's senkou_a/senkou_b are forward-projected — this is
        expected and should be dropped before ML training, not filled.
    """
    # --- trend ---
    df = calculate_multiple_emas(df, periods=[20, 50, 200])
    df = calculate_multiple_smas(df, periods=[20, 50, 200])
    df = calculate_macd(df)
    df = calculate_adx(df, period=14)
    df = calculate_supertrend(df, period=10, multiplier=3.0)
    df = calculate_ichimoku(df)

    # --- momentum ---
    df = calculate_rsi(df, period=14)
    df = calculate_stochastic(df)
    df = calculate_cci(df, period=20)
    df = calculate_williams_r(df, period=14)

    # --- volatility ---
    df = calculate_atr(df, period=14)
    df = calculate_bollinger(df, period=20)
    df = calculate_keltner(df)
    df = calculate_donchian(df, period=20)

    # --- volume ---
    if has_volume:
        df = calculate_vwap(df, session_col=session_col)
        df = calculate_obv(df)
        df = calculate_cmf(df, period=20)
        df = calculate_mfi(df, period=14)

    return df