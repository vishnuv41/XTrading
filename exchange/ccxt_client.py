"""
exchange/ccxt_client.py
-------------------------
Shared ccxt.pro setup used by every exchange/<name>.py — rate limiting,
retry policy, and sandbox/testnet toggling live here once instead of
being duplicated in binance.py, bybit.py, etc.

ccxt.pro (the async + websocket variant of ccxt) is used rather than
plain ccxt so REST and websocket calls share one client per venue.
"""

import asyncio
import logging

import ccxt.pro as ccxtpro

from config.settings import settings

logger = logging.getLogger(__name__)

# Exponential backoff schedule for transient errors (network blips,
# rate-limit responses). Not retried: auth errors, bad symbol errors —
# those are caller bugs, not transient failures, so they should raise.
_RETRY_DELAYS = [1, 2, 5, 10, 30]


def build_ccxt_client(exchange_id: str, api_key: str = "", api_secret: str = "", use_testnet: bool = True):
    """
    Construct a rate-limited ccxt.pro client for `exchange_id`
    (e.g. "binance", "bybit"). Credentials are optional — public
    endpoints (OHLCV, symbols) work without them; private endpoints
    (account/order data) will raise if called without real keys.
    """
    exchange_class = getattr(ccxtpro, exchange_id)
    client = exchange_class({
        "apiKey": api_key or None,
        "secret": api_secret or None,
        "enableRateLimit": True,  # let ccxt pace requests to the venue's documented limit
        "options": {"defaultType": "spot"},
    })
    if use_testnet and hasattr(client, "set_sandbox_mode"):
        client.set_sandbox_mode(True)
        logger.info("%s client running in sandbox/testnet mode", exchange_id)
    return client


async def with_retry(coro_fn, *args, retries: int = len(_RETRY_DELAYS), **kwargs):
    """
    Call an async ccxt method with exponential backoff on transient
    errors. Usage: `await with_retry(client.fetch_ohlcv, symbol, timeframe)`.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeNotAvailable, ccxtpro.RequestTimeout, ccxtpro.DDoSProtection, ccxtpro.RateLimitExceeded) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            logger.warning("Transient error (%s), retrying in %ss [%d/%d]", exc, delay, attempt + 1, retries)
            await asyncio.sleep(delay)
        # Auth/bad-request/exchange errors intentionally NOT caught here —
        # they indicate a real problem the caller needs to see immediately.
    raise last_exc
