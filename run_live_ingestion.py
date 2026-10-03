"""
run_live_ingestion.py
---------------------
CLI entrypoint for streaming live closed 1H candles from Binance WebSocket to PostgreSQL & Redis.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exchange.binance import BinanceExchange
from ingestion.realtime_stream import run_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

async def main():
    exchange = BinanceExchange()
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["1h"]
    logging.info("Starting direct live WebSocket stream for %s %s...", symbols, timeframes)
    await run_all(exchange, symbols, timeframes)

if __name__ == "__main__":
    asyncio.run(main())
