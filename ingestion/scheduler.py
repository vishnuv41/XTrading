"""
ingestion/scheduler.py
-------------------------
Ties everything in ingestion/ together into scheduled jobs:
  - startup: backfill any symbol/timeframe with no ingestion_state row
    (first run) or with a stale last_synced_ts (gap since last run),
    then hand off to the live websocket stream.
  - recurring: a daily gap-check sweep (catches anything the live
    stream missed — reconnect windows, brief outages) and a health
    ping.

Run this as the entrypoint: `python -m ingestion.scheduler`.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import settings
from database.timescaledb import apply_schema
from exchange.binance import BinanceExchange
from ingestion.historical_loader import backfill_all
from ingestion.gap_filler import fill_gaps
from ingestion.realtime_stream import run_all

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

DEFAULT_BACKFILL_START = datetime.now(timezone.utc) - timedelta(days=365)


async def _daily_gap_sweep(exchange):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=2)  # only re-check a short recent window; older history rarely develops new gaps
    for symbol in settings.symbols.symbols:
        for timeframe in settings.symbols.timeframes:
            try:
                await fill_gaps(exchange, symbol, timeframe, start, end)
            except Exception as exc:
                logger.error("Gap sweep failed for %s %s: %s", symbol, timeframe, exc)


async def main():
    apply_schema()

    exchange = BinanceExchange()

    logger.info("Running initial backfill for %s across %s", settings.symbols.symbols, settings.symbols.timeframes)
    await backfill_all(exchange, settings.symbols.symbols, settings.symbols.timeframes, DEFAULT_BACKFILL_START)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(lambda: asyncio.create_task(_daily_gap_sweep(exchange)), "cron", hour=0, minute=5)
    scheduler.start()
    logger.info("Scheduler started: daily gap sweep at 00:05 UTC")

    logger.info("Starting live stream for %s across %s", settings.symbols.symbols, settings.symbols.timeframes)
    await run_all(exchange, settings.symbols.symbols, settings.symbols.timeframes)


if __name__ == "__main__":
    asyncio.run(main())
