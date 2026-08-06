"""
preprocessing/resampler.py
------------------------------
Derive higher timeframes (5m/15m/1h/4h/1d) from the base timeframe
(default 1m, see config.settings.symbols.base_timeframe) rather than
fetching each independently from the exchange. This guarantees every
timeframe agrees exactly on OHLC values (impossible if fetched
separately, since exchanges compute their own bucket boundaries) and
cuts REST/websocket calls by (n_timeframes - 1).
"""

import pandas as pd

_PANDAS_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Resample a base-timeframe OHLCV DataFrame up to `target_timeframe`.

    Args:
        df: DataFrame with ts_col + open/high/low/close/volume, sorted
            ascending, tz-aware UTC (see preprocessing/timezone.py).
        target_timeframe: one of the keys in _PANDAS_FREQ.

    Returns:
        Resampled DataFrame, one row per target-timeframe bucket. The
        last bucket is dropped if it's incomplete (fewer base-timeframe
        rows than the bucket should contain) — an in-progress candle
        shouldn't be written to the ohlcv table as if it were closed.
    """
    if target_timeframe not in _PANDAS_FREQ:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")

    freq = _PANDAS_FREQ[target_timeframe]
    indexed = df.set_index(ts_col)

    resampled = indexed.resample(freq, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])

    # Drop the final bucket if it isn't full yet (still accumulating).
    bucket_seconds = pd.Timedelta(freq).total_seconds()
    base_seconds = (df[ts_col].iloc[1] - df[ts_col].iloc[0]).total_seconds() if len(df) > 1 else 60
    expected_rows_per_bucket = max(int(bucket_seconds / base_seconds), 1)

    if len(resampled) > 0:
        last_bucket_start = resampled.index[-1]
        last_bucket_end = last_bucket_start + pd.Timedelta(freq)
        rows_in_last_bucket = indexed.loc[last_bucket_start:last_bucket_end].index.nunique()
        if rows_in_last_bucket < expected_rows_per_bucket:
            resampled = resampled.iloc[:-1]

    return resampled.reset_index().rename(columns={ts_col: ts_col})
