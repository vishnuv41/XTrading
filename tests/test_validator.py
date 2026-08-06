"""tests/test_validator.py — OHLC consistency checks."""

from preprocessing.validator import validate_ohlcv


def _row(o=100, h=101, l=99, c=100.5, v=10):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def test_valid_row_passes():
    assert validate_ohlcv([_row()]) == [_row()]


def test_high_below_close_dropped():
    bad = _row(h=99, c=100.5)  # high < close, impossible
    assert validate_ohlcv([bad]) == []


def test_low_above_open_dropped():
    bad = _row(l=101, o=100)  # low > open, impossible
    assert validate_ohlcv([bad]) == []


def test_negative_price_dropped():
    bad = _row(o=-1)
    assert validate_ohlcv([bad]) == []


def test_negative_volume_dropped():
    bad = _row(v=-5)
    assert validate_ohlcv([bad]) == []


def test_nan_dropped():
    bad = _row(c=float("nan"))
    assert validate_ohlcv([bad]) == []


def test_mixed_batch_keeps_only_valid():
    rows = [_row(), _row(h=-1), _row(c=200, h=201)]
    result = validate_ohlcv(rows)
    assert len(result) == 2
