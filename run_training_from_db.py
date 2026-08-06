"""
run_training_from_db.py
--------------------------
The real end-to-end workflow: Person 1's stored market data ->
Person 2's ML pipeline -> saved, reloadable model artifacts.

This replaces ml.train._make_synthetic_ohlcv() as the entry point once
Person 1's ingestion has real history in PostgreSQL. Usage:

    # one-time: backfill history for the symbol/timeframe you want to train on
    python -m ingestion.historical_loader   # or scheduler.py for the full loop

    # then:
    python run_training_from_db.py --symbol BTC/USDT --timeframe 1h

Falls back to a clear, actionable error (not a stack trace) if no data
is found — see pipeline.data_loader.load_ohlcv.
"""

import argparse
import json
import logging

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.train import run_training_pipeline

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Person 2's ML pipeline on Person 1's stored market data.")
    parser.add_argument("--symbol", default=settings.symbols.symbols[0], help="e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="e.g. 1h, 15m, 1d")
    parser.add_argument("--exchange", default=settings.exchange.default_exchange)
    parser.add_argument("--output-dir", default="models_artifacts")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    logger.info("Loading %s %s from %s...", args.symbol, args.timeframe, args.exchange)
    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)

    logger.info("Training on %d candles...", len(df))
    result = run_training_pipeline(df, n_splits=args.n_splits, output_dir=args.output_dir)

    print("\nTraining complete.")
    print(json.dumps({k: v for k, v in result["metrics"].items()}, indent=2, default=str))
    print(f"Artifacts saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
