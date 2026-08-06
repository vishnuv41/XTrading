"""
ingestion/gap_filler.py
--------------------------
Detects missing candles in a time range (e.g. after downtime, a missed
websocket reconnect window, or before realtime_stream.py was running)
and re-fetches just those gaps via REST, rather than re-backfilling the
whole range.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from database.connection import get_engine
from database.timescaledb import upsert_candles
from exchange.base_exchange import BaseExchange
from ingestion.historical_loader import _row_from_candle
from preprocessing.cleaner import clean_ohlcv
from preprocessing.validator import validate_ohlcv

logger = logging.getLogger(__name__)

_TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def find_gaps(exchange_name: str, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[tuple]:
    """
    Return a list of (gap_start, gap_end) datetime pairs where expected
    candles are missing from the ohlcv table, by comparing actual row
    timestamps against the expected regular grid for this timeframe.
    """
    step = timedelta(milliseconds=_TIMEFRAME_MS.get(timeframe, 60_000))
    engine = get_engine()
    query = text(
        """
        SELECT ts FROM ohlcv
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
          AND ts >= :start AND ts <= :end
        ORDER BY ts
        """
    )
    with engine.connect() as conn:
        existing = [row[0] for row in conn.execute(query, {
            "exchange": exchange_name, "symbol": symbol, "timeframe": timeframe,
            "start": start, "end": end,
        })]

    gaps = []
    cursor = start
    for ts in existing:
        if ts > cursor:
            gaps.append((cursor, ts))
        cursor = max(cursor, ts + step)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


async def fill_gaps(exchange: BaseExchange, symbol: str, timeframe: str, start: datetime, end: datetime = None):
    """Find and backfill every gap for symbol/timeframe in [start, end] (default end: now)."""
    end = end or datetime.now(timezone.utc)
    gaps = find_gaps(exchange.name, symbol, timeframe, start, end)

    if not gaps:
        logger.info("%s %s: no gaps found in [%s, %s]", symbol, timeframe, start, end)
        return 0

    total_written = 0
    for gap_start, gap_end in gaps:
        logger.info("%s %s: filling gap [%s, %s]", symbol, timeframe, gap_start, gap_end)
        since_ms = int(gap_start.timestamp() * 1000)
        candles = await exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        candles = [c for c in candles if gap_start.timestamp() * 1000 <= c[0] <= gap_end.timestamp() * 1000]
        if not candles:
            continue
        rows = [_row_from_candle(exchange.name, symbol, timeframe, c) for c in candles]
        for row in rows:
            row["is_synthetic"] = True  # mark reconstructed-from-gap candles, per schema.sql's is_synthetic flag
        rows = validate_ohlcv(clean_ohlcv(rows))
        total_written += upsert_candles(rows)

    logger.info("%s %s: gap fill complete, %d rows written", symbol, timeframe, total_written)
    return total_written
