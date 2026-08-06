"""
exchange/websocket_client.py
------------------------------
Venue-agnostic wrapper around BaseExchange.watch_ohlcv that adds the
auto-reconnect + backoff behavior ingestion/realtime_stream.py needs,
on top of whatever reconnection ccxt.pro already does internally. This
is the outer safety net for when a venue implementation's watch_ohlcv
generator itself dies (not just a single message drop).
"""

import asyncio
import logging

from exchange.base_exchange import BaseExchange

logger = logging.getLogger(__name__)

_RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]


async def stream_ohlcv(exchange: BaseExchange, symbol: str, timeframe: str):
    """
    Async generator yielding candles from exchange.watch_ohlcv(),
    transparently reconnecting (with backoff) if the underlying
    generator raises or exits unexpectedly. Runs forever — the caller
    (realtime_stream.py) is expected to `async for candle in
    stream_ohlcv(...)` in a long-lived task.
    """
    attempt = 0
    while True:
        try:
            async for candle in exchange.watch_ohlcv(symbol, timeframe):
                attempt = 0  # reset backoff after any successful message
                yield candle
            # Generator exited without raising — treat as a drop, not success.
            logger.warning("watch_ohlcv(%s, %s) stream ended unexpectedly, reconnecting", symbol, timeframe)
        except asyncio.CancelledError:
            raise  # let real shutdown propagate
        except Exception as exc:
            logger.error("watch_ohlcv(%s, %s) failed: %s", symbol, timeframe, exc)

        delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
        logger.info("Reconnecting %s %s in %ss", symbol, timeframe, delay)
        await asyncio.sleep(delay)
        attempt += 1
