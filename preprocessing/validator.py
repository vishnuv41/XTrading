"""
preprocessing/validator.py
------------------------------
OHLC sanity checks. Runs after cleaner.py, right before a write to
PostgreSQL — this is the last line of defense against garbage data
reaching Person 2's feature engineering, which assumes every row it
sees is internally consistent.
"""

import logging

logger = logging.getLogger(__name__)


import math

def _is_valid_row(row: dict) -> bool:
    o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]

    if any(x is None for x in (o, h, l, c, v)):
        return False
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in (o, h, l, c, v)):
        return False
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return False
    if v < 0:
        return False
    if h < max(o, c, l):
        return False
    if l > min(o, c, h):
        return False
    if h < l:
        return False
    return True


def validate_ohlcv(rows: list[dict]) -> list[dict]:
    """
    Filter out rows that fail basic OHLC consistency checks (high is
    actually the highest, low is actually the lowest, no negative/zero
    prices, no NaNs). Invalid rows are dropped and logged, not raised —
    one bad candle from a flaky exchange response shouldn't halt an
    entire backfill/stream.
    """
    valid, dropped = [], 0
    for row in rows:
        if _is_valid_row(row):
            valid.append(row)
        else:
            dropped += 1
            logger.warning("Dropped invalid OHLCV row: %s", row)

    if dropped:
        logger.warning("validate_ohlcv: dropped %d/%d rows", dropped, len(rows))
    return valid
