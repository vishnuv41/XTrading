"""
ingestion/realtime_stream.py
-------------------------------
Live path: websocket -> clean/validate -> DB write + Redis publish, one
write per closed candle (not per tick — see _is_candle_closed below).
Run as a long-lived task per symbol/timeframe (see scheduler.py for how
these get started).
"""

import asyncio
import logging
from datetime import datetime, timezone

from database.timescaledb import upsert_candles
from database.redis_cache import publish_candle, set_latest_price
from exchange.base_exchange import BaseExchange
from exchange.websocket_client import stream_ohlcv
from preprocessing.cleaner import clean_ohlcv
from preprocessing.validator import validate_ohlcv

logger = logging.getLogger(__name__)

_TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _is_candle_closed(ts_ms: int, timeframe: str, now_ms: int) -> bool:
    """A candle is closed once we're past its bucket's end time — ccxt.pro streams the still-forming candle too, which we don't want written yet."""
    step_ms = _TIMEFRAME_MS.get(timeframe, 60_000)
    return now_ms >= ts_ms + step_ms


async def run_stream(exchange: BaseExchange, symbol: str, timeframe: str):
    """
    Consume the live candle stream for one symbol/timeframe forever.
    Every message updates the Redis latest-price cache (for the
    dashboard); only *closed* candles get written to PostgreSQL and
    published to subscribers (Person 2's live inference loop reads
    closed candles only, to match how it was trained).
    """
    last_written_ts_ms = None

    async for candle in stream_ohlcv(exchange, symbol, timeframe):
        ts_ms, o, h, l, c, v = candle
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        await set_latest_price(symbol, price=c, ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat())

        if not _is_candle_closed(ts_ms, timeframe, now_ms):
            continue
        if last_written_ts_ms is not None and ts_ms <= last_written_ts_ms:
            continue  # already wrote this candle; ccxt.pro can repeat the last closed one

        row = {
            "exchange": exchange.name, "symbol": symbol, "timeframe": timeframe,
            "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        }
        rows = validate_ohlcv(clean_ohlcv([row]))
        if not rows:
            logger.warning("%s %s candle at %s failed validation, skipped", symbol, timeframe, row["ts"])
            continue

        upsert_candles(rows)
        await publish_candle(symbol, timeframe, rows[0] | {"ts": rows[0]["ts"].isoformat()})
        last_written_ts_ms = ts_ms
        logger.debug("%s %s candle closed @ %s written", symbol, timeframe, row["ts"])


async def run_all(exchange: BaseExchange, symbols: list, timeframes: list):
    """Start one run_stream task per symbol x timeframe and run them concurrently."""
    tasks = [
        asyncio.create_task(run_stream(exchange, symbol, timeframe))
        for symbol in symbols for timeframe in timeframes
    ]
    await asyncio.gather(*tasks)
