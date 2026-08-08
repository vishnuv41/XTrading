"""
paper_trading/run_paper_trading.py
----------------------------------

CLI entrypoint for the paper trading engine.

Supports two modes:

1. Historical Replay
   Replays historical candles through the realtime pipeline
   exactly as if they arrived live.

2. Live Paper Trading
   Subscribes to Redis candle updates and continuously
   simulates trading without placing any real orders.

Example:

python -m paper_trading.run_paper_trading \
    --symbol BTC/USDT \
    --timeframe 1h \
    --model-dir models_artifacts/BTCUSDT_1h \
    --mode replay

python -m paper_trading.run_paper_trading \
    --symbol BTC/USDT \
    --timeframe 1h \
    --model-dir models_artifacts/BTCUSDT_1h \
    --mode live
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from indicators import calculate_all_indicators

from ml.predict import load_training_artifacts

from ml.utils.preprocessing import build_feature_matrix

from pipeline.data_loader import load_ohlcv

from paper_trading.engine import PaperTradingEngine

from paper_trading.metrics import (
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def prepare_dataframe(symbol: str, timeframe: str):
    """
    Load historical OHLCV and compute the same feature set used
    during model training.
    """

    logger.info(
        "Loading %s %s candles...",
        symbol,
        timeframe,
    )

    df = load_ohlcv(symbol, timeframe)

    logger.info(
        "Loaded %d candles",
        len(df),
    )

    df = calculate_all_indicators(df)

    df = build_feature_matrix(df)

    return df


def load_model(model_dir: str):
    """
    Load ensemble + calibrator + feature list.
    """

    logger.info("Loading model artifacts...")

    artifacts = load_training_artifacts(model_dir)

    return (
        artifacts["ensemble"],
        artifacts["feature_columns"],
        artifacts.get("calibrator"),
    )

# ----------------------------------------------------------
# Replay Mode
# ----------------------------------------------------------

def run_replay(
    symbol: str,
    timeframe: str,
    model_dir: str,
):
    df = prepare_dataframe(symbol, timeframe)

    model, feature_columns, calibrator = load_model(model_dir)

    engine = PaperTradingEngine(
        symbol=symbol,
        timeframe=timeframe,
        model=model,
        feature_columns=feature_columns,
        calibrator=calibrator,
    )

    logger.info("Starting historical replay...")

    results = engine.run_replay(df)

    logger.info(
        "Replay finished (%d processed bars).",
        len(results),
    )

    print_summary(engine.portfolio)


# ----------------------------------------------------------
# Live Mode
# ----------------------------------------------------------

async def run_live(
    symbol: str,
    timeframe: str,
    model_dir: str,
):
    model, feature_columns, calibrator = load_model(model_dir)

    engine = PaperTradingEngine(
        symbol=symbol,
        timeframe=timeframe,
        model=model,
        feature_columns=feature_columns,
        calibrator=calibrator,
    )

    logger.info(
        "Starting LIVE paper trading for %s %s...",
        symbol,
        timeframe,
    )

    await engine.run_live(
        history_loader=lambda: prepare_dataframe(symbol, timeframe)
    )


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper Trading Engine"
    )

    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
    )

    parser.add_argument(
        "--timeframe",
        default="1h",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
    )

    return parser.parse_args()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    args = parse_args()

    if args.mode == "replay":
        run_replay(
            symbol=args.symbol,
            timeframe=args.timeframe,
            model_dir=args.model_dir,
        )

    else:
        asyncio.run(
            run_live(
                symbol=args.symbol,
                timeframe=args.timeframe,
                model_dir=args.model_dir,
            )
        )


if __name__ == "__main__":
    main()