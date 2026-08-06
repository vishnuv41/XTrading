"""tests/test_cleaner.py — dedup and zero-volume/zero-range dropping."""

from datetime import datetime, timezone

from preprocessing.cleaner import clean_ohlcv


def _row(ts, o=100, h=101, l=99, c=100.5, v=10):
    return {"exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1m", "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_dedup_keeps_last():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [_row(ts, c=100), _row(ts, c=200)]  # same key, different close
    result = clean_ohlcv(rows)
    assert len(result) == 1
    assert result[0]["close"] == 200


def test_drops_zero_volume_zero_range():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    flat = _row(ts, o=100, h=100, l=100, c=100, v=0)
    result = clean_ohlcv([flat])
    assert result == []


def test_keeps_zero_volume_if_range_nonzero():
    # zero volume alone (real quiet period with a price move) should NOT be dropped
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    row = _row(ts, o=100, h=102, l=99, c=101, v=0)
    result = clean_ohlcv([row])
    assert len(result) == 1


def test_sorted_ascending():
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = clean_ohlcv([_row(ts2), _row(ts1)])
    assert [r["ts"] for r in result] == [ts1, ts2]
