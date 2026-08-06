"""tests/test_resampler.py — 1m -> higher timeframe aggregation."""

import pandas as pd

from preprocessing.resampler import resample_ohlcv


def _make_1m_df(n_minutes=10, start="2024-01-01T00:00:00Z"):
    ts = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": range(100, 100 + n_minutes),
        "high": [x + 1 for x in range(100, 100 + n_minutes)],
        "low": [x - 1 for x in range(100, 100 + n_minutes)],
        "close": [x + 0.5 for x in range(100, 100 + n_minutes)],
        "volume": [10] * n_minutes,
    })


def test_resample_5m_ohlc_correct():
    df = _make_1m_df(n_minutes=10)  # exactly two full 5m buckets
    result = resample_ohlcv(df, "5m")
    assert len(result) == 2
    first = result.iloc[0]
    assert first["open"] == 100          # first open of the bucket
    assert first["close"] == df.iloc[4]["close"]  # last close of the bucket
    assert first["high"] == df.iloc[0:5]["high"].max()
    assert first["low"] == df.iloc[0:5]["low"].min()
    assert first["volume"] == 50          # sum of 5 rows of volume=10


def test_incomplete_final_bucket_dropped():
    df = _make_1m_df(n_minutes=7)  # one full 5m bucket + 2 leftover minutes
    result = resample_ohlcv(df, "5m")
    assert len(result) == 1  # incomplete second bucket dropped


def test_unknown_timeframe_raises():
    df = _make_1m_df(n_minutes=5)
    try:
        resample_ohlcv(df, "3m")
        assert False, "should have raised"
    except ValueError:
        pass
