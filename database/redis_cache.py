"""
database/redis_cache.py
-------------------------
Thin Redis wrapper for two jobs:
  1. Latest-price cache — O(1) "what's BTC/USDT trading at right now"
     lookups without hitting PostgreSQL, for the dashboard's live
     ticker and any other hot-path read.
  2. Pub/sub fanout — realtime_stream.py publishes each closed candle
     once; any number of subscribers (Person 2's live inference loop,
     Person 3's websocket_gateway) can listen without realtime_stream.py
     needing to know who's downstream.
"""

import json
import logging

import redis.asyncio as aioredis

from config.settings import settings

logger = logging.getLogger(__name__)

_client: aioredis.Redis = None

CANDLE_CHANNEL_PREFIX = "candles"  # channel name: candles:{symbol}:{timeframe}
LATEST_PRICE_KEY_PREFIX = "latest"  # key name: latest:{symbol}


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis.url, decode_responses=True)
        logger.info("Created Redis client for %s:%s/%s", settings.redis.host, settings.redis.port, settings.redis.db)
    return _client


async def set_latest_price(symbol: str, price: float, ts: str):
    """Cache the latest trade/close price for a symbol. ts should be ISO-8601 UTC."""
    client = get_client()
    key = f"{LATEST_PRICE_KEY_PREFIX}:{symbol}"
    await client.set(key, json.dumps({"price": price, "ts": ts}))


async def get_latest_price(symbol: str) -> dict | None:
    client = get_client()
    key = f"{LATEST_PRICE_KEY_PREFIX}:{symbol}"
    raw = await client.get(key)
    return json.loads(raw) if raw else None


async def publish_candle(symbol: str, timeframe: str, candle: dict):
    """
    Publish a newly-closed candle to subscribers. `candle` should be
    JSON-serializable (the dict shape written to the ohlcv table:
    ts/open/high/low/close/volume).
    """
    client = get_client()
    channel = f"{CANDLE_CHANNEL_PREFIX}:{symbol}:{timeframe}"
    await client.publish(channel, json.dumps(candle))


async def subscribe_candles(symbol: str, timeframe: str):
    """
    Async generator yielding each new candle dict as it's published.
    Usage:
        async for candle in subscribe_candles("BTC/USDT", "1m"):
            ...
    """
    client = get_client()
    channel = f"{CANDLE_CHANNEL_PREFIX}:{symbol}:{timeframe}"
    pubsub = client.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            yield json.loads(message["data"])
    finally:
        await pubsub.unsubscribe(channel)
