"""
exchange/bybit.py
--------------------
Bybit implementation of BaseExchange. Optional second venue — same
interface as binance.py, so ingestion code doesn't change to support it,
only config/symbols.yaml needs an entry with exchange: bybit.
"""

import logging

from exchange.base_exchange import BaseExchange
from exchange.ccxt_client import build_ccxt_client, with_retry
from config.settings import settings

logger = logging.getLogger(__name__)


class BybitExchange(BaseExchange):
    name = "bybit"

    def __init__(self, api_key: str = "", api_secret: str = "", use_testnet: bool = None):
        self._client = build_ccxt_client(
            "bybit",
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet if use_testnet is not None else settings.exchange.use_testnet,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000) -> list[list]:
        return await with_retry(self._client.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit)

    async def fetch_symbols(self) -> list[str]:
        markets = await with_retry(self._client.load_markets)
        return [m for m, info in markets.items() if info.get("spot", True)]

    async def watch_ohlcv(self, symbol: str, timeframe: str):
        while True:
            try:
                candles = await self._client.watch_ohlcv(symbol, timeframe)
                if candles:
                    yield candles[-1]
            except Exception as exc:
                logger.warning("watch_ohlcv(%s, %s) dropped: %s — reconnecting", symbol, timeframe, exc)
                continue

    async def close(self):
        await self._client.close()
