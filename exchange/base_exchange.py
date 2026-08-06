"""
exchange/base_exchange.py
---------------------------
Abstract interface every exchange client implements. ingestion/ and
api/ code against this interface, not against binance.py/bybit.py
directly, so adding a new venue never requires touching ingestion
logic — only a new exchange/<name>.py implementing this class.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class BaseExchange(ABC):
    """
    All timestamps in and out are UTC datetimes (or ms-since-epoch ints
    for `since`, matching ccxt convention) — never naive/local time.
    See preprocessing/timezone.py for the project-wide UTC rule.
    """

    name: str  # e.g. "binance" — used as the `exchange` column value in ohlcv

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int = None,
        limit: int = 1000,
    ) -> list[list]:
        """
        Fetch OHLCV candles.

        Returns a list of [timestamp_ms, open, high, low, close, volume],
        oldest first — the raw ccxt-style shape. Callers (historical_loader,
        gap_filler) are responsible for converting to the ohlcv table's
        row-dict shape; this layer stays a thin, uniform data source.
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch_symbols(self) -> list[str]:
        """Return all tradeable symbols on this venue, e.g. ['BTC/USDT', 'ETH/USDT', ...]."""
        raise NotImplementedError

    @abstractmethod
    async def watch_ohlcv(self, symbol: str, timeframe: str):
        """
        Async generator yielding each new/updated candle as
        [timestamp_ms, open, high, low, close, volume] over a websocket
        connection. Used by ingestion/realtime_stream.py.
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        """Release any open connections (REST session, websocket)."""
        raise NotImplementedError
