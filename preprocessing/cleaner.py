"""
preprocessing/cleaner.py
----------------------------
Dedup + drop obviously-bad rows before validation. Runs on the raw
row-dicts historical_loader.py/realtime_stream.py build from exchange
responses, before they're written to PostgreSQL.
"""

import logging

logger = logging.getLogger(__name__)


def clean_ohlcv(rows: list[dict]) -> list[dict]:
    """
    - Deduplicate by (exchange, symbol, timeframe, ts), keeping the last
      occurrence (exchanges occasionally return the same candle twice
      across paginated requests).
    - Drop zero-volume AND zero-range (open==high==low==close) rows —
      these are typically an exchange placeholder for "no trades this
      period" on illiquid pairs, not real market data, and would distort
      volatility/volume features downstream.
    """
    deduped = {}
    for row in rows:
        key = (row["exchange"], row["symbol"], row["timeframe"], row["ts"])
        deduped[key] = row  # later occurrence overwrites earlier — "keep last"

    cleaned = []
    dropped = 0
    for row in deduped.values():
        zero_volume = row["volume"] == 0
        zero_range = row["open"] == row["high"] == row["low"] == row["close"]
        if zero_volume and zero_range:
            dropped += 1
            continue
        cleaned.append(row)

    if dropped:
        logger.info("clean_ohlcv: dropped %d zero-volume/zero-range rows", dropped)

    return sorted(cleaned, key=lambda r: r["ts"])
