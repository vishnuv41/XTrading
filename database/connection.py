"""
database/connection.py
-----------------------
Pooled connection factories. Two paths are exposed on purpose:

    get_engine()      -> SQLAlchemy sync engine (psycopg2), for scripts
                          and one-off tools (historical_loader batch
                          jobs, migrations, ad-hoc queries).
    get_async_pool()   -> asyncpg pool, for the websocket-driven live
                          path (realtime_stream.py, api/websocket_api.py)
                          where blocking I/O would stall the event loop.

Both are lazily created singletons — call get_engine()/get_async_pool()
wherever you need a connection; don't construct SQLAlchemy engines or
asyncpg pools directly elsewhere, so there's exactly one pool per
process instead of one per caller.
"""

import asyncio
import logging

import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)

_engine: Engine = None
_session_factory = None
_async_pool: asyncpg.Pool = None
_async_pool_lock = asyncio.Lock()


def get_engine() -> Engine:
    """Lazily create and return the shared sync SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.db.sync_url,
            pool_size=settings.db.pool_min_size,
            max_overflow=settings.db.pool_max_size - settings.db.pool_min_size,
            pool_pre_ping=True,  # drop dead connections instead of erroring on them
            future=True,
        )
        logger.info("Created sync DB engine for %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)
    return _engine


def get_session_factory():
    """Return a sessionmaker bound to the shared engine (for ORM usage via database/models.py)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


async def get_async_pool() -> asyncpg.Pool:
    """Lazily create and return the shared asyncpg pool. Must be awaited from within an event loop."""
    global _async_pool
    if _async_pool is not None:
        return _async_pool
    async with _async_pool_lock:
        if _async_pool is None:  # re-check after acquiring the lock
            _async_pool = await asyncpg.create_pool(
                dsn=settings.db.async_url,
                min_size=settings.db.pool_min_size,
                max_size=settings.db.pool_max_size,
            )
            logger.info("Created async DB pool for %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)
    return _async_pool


async def close_async_pool():
    """Call on graceful shutdown (e.g. FastAPI lifespan/shutdown event)."""
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None
