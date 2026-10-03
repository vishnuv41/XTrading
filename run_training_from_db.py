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


def _default_output_dir(symbol: str, timeframe: str) -> str:
    # Matches run_paper_trading.py's _model_dir_for() and every other
    # script's --model-dir usage (models_artifacts/BTCUSDT_1h etc.) —
    # this MUST stay in sync with that convention. Previously this
    # defaulted to the flat "models_artifacts" with no per-symbol
    # subfolder, so every retrain wrote fresh 3-class artifacts to
    # models_artifacts/ensemble.pkl while every consumer (backtest,
    # paper trading, diagnose_ensemble.py, etc.) loaded from
    # models_artifacts/<SYMBOL>_<timeframe>/ instead — a directory
    # training was never actually writing to. That stale directory
    # (whatever originally created it) is what kept getting reloaded,
    # producing the repeated "HOLD disappeared again after retraining"
    # symptom that looked like a labeling/ensemble bug but wasn't.
    return f"models_artifacts/{symbol.replace('/', '')}_{timeframe}"


def main():
    parser = argparse.ArgumentParser(description="Train Person 2's ML pipeline on Person 1's stored market data.")
    parser.add_argument("--symbol", default=settings.symbols.symbols[0], help="e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="e.g. 1h, 15m, 1d")
    parser.add_argument("--exchange", default=settings.exchange.default_exchange)
    parser.add_argument("--output-dir", default=None,
                         help="Defaults to models_artifacts/<SYMBOL>_<timeframe>, matching every "
                              "other script's --model-dir convention. Only override this if you "
                              "intentionally want an artifact directory other scripts won't find.")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir(args.symbol, args.timeframe)

    logger.info("Loading %s %s from %s...", args.symbol, args.timeframe, args.exchange)
    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)

    logger.info("Training on %d candles...", len(df))
    result = run_training_pipeline(df, n_splits=args.n_splits, output_dir=output_dir)

    print("\nTraining complete.")
    print(json.dumps({k: v for k, v in result["metrics"].items()}, indent=2, default=str))
    print(f"Artifacts saved to: {output_dir}/")


if __name__ == "__main__":
    main()