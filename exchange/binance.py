"""
exchange/binance.py
---------------------
Binance implementation of BaseExchange, built on the shared ccxt.pro
client from exchange/ccxt_client.py.
"""

import logging

from exchange.base_exchange import BaseExchange
from exchange.ccxt_client import build_ccxt_client, with_retry
from config.settings import settings

logger = logging.getLogger(__name__)


class BinanceExchange(BaseExchange):
    name = "binance"

    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = None):
        self._client = build_ccxt_client(
            "binance",
            api_key=api_key if api_key is not None else settings.exchange.binance_api_key,
            api_secret=api_secret if api_secret is not None else settings.exchange.binance_api_secret,
            use_testnet=use_testnet if use_testnet is not None else settings.exchange.use_testnet,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000) -> list[list]:
        return await with_retry(self._client.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit)

    async def fetch_symbols(self) -> list[str]:
        markets = await with_retry(self._client.load_markets)
        return [m for m, info in markets.items() if info.get("spot", True)]

    async def watch_ohlcv(self, symbol: str, timeframe: str):
        """
        Yields the latest candle every time it updates (ccxt.pro's
        watch_ohlcv returns the full recent window on each tick — we
        only yield the last, still-forming/just-closed candle).
        Reconnection on drop is handled by ccxt.pro internally; a wrapper
        retry loop here covers the case where it still raises.
        """
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
