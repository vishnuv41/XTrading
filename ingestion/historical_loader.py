"""
ingestion/historical_loader.py
---------------------------------
Bulk backfill: paginate an exchange's fetch_ohlcv from a start time up
to now, writing to PostgreSQL in batches. This is what Person 2 runs
once to get real historical data for training (replacing
_make_synthetic_ohlcv), and what a fresh symbol/timeframe uses to catch
up before realtime_stream.py takes over.
"""

import asyncio
import logging
from datetime import datetime, timezone

from database.timescaledb import upsert_candles, update_ingestion_state
from preprocessing.cleaner import clean_ohlcv
from preprocessing.validator import validate_ohlcv
from exchange.base_exchange import BaseExchange

logger = logging.getLogger(__name__)

_TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _row_from_candle(exchange_name: str, symbol: str, timeframe: str, candle: list) -> dict:
    ts_ms, o, h, l, c, v = candle
    return {
        "exchange": exchange_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }


async def backfill_symbol(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime = None,
    batch_size: int = 1000,
):
    """
    Fetch and store all candles for `symbol`/`timeframe` from `since` to
    `until` (default: now). Paginates using each batch's last timestamp
    as the next `since`, so it works regardless of how far back `since`
    is or how many candles that spans.
    """
    until = until or datetime.now(timezone.utc)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    step_ms = _TIMEFRAME_MS.get(timeframe, 60_000)

    total_written = 0
    cursor = since_ms

    while cursor < until_ms:
        candles = await exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_size)
        if not candles:
            break

        rows = [_row_from_candle(exchange.name, symbol, timeframe, c) for c in candles]
        rows = clean_ohlcv(rows)
        rows = validate_ohlcv(rows)

        written = upsert_candles(rows)
        total_written += written

        last_ts_ms = candles[-1][0]
        if last_ts_ms <= cursor:
            # Exchange returned no forward progress (e.g. end of history) — stop instead of looping forever.
            break
        cursor = last_ts_ms + step_ms

        logger.info("%s %s: backfilled up to %s (%d rows this batch)",
                     symbol, timeframe, datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc), written)

        if len(candles) < batch_size:
            break  # exchange returned a partial page -> caught up to its latest data

    update_ingestion_state(exchange.name, symbol, timeframe, last_synced_ts=until, status="ok")
    logger.info("Backfill complete for %s %s: %d rows written", symbol, timeframe, total_written)
    return total_written


async def backfill_all(exchange: BaseExchange, symbols: list, timeframes: list, since: datetime):
    """Backfill every symbol x timeframe combination sequentially (kept simple/sequential to stay within exchange rate limits)."""
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                await backfill_symbol(exchange, symbol, timeframe, since)
            except Exception as exc:
                logger.error("Backfill failed for %s %s: %s", symbol, timeframe, exc)
                update_ingestion_state(exchange.name, symbol, timeframe, last_synced_ts=None, status="error", error=str(exc))
