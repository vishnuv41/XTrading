"""
run_paper_trading.py
----------------------
CLI entrypoint.

Replay mode (offline, uses DB history — good for validating a model
before ever running it live):
    python run_paper_trading.py --symbol BTC/USDT --timeframe 1h --mode replay

Live mode (subscribes to Person 1's Redis candle fanout; run
ingestion/realtime_stream.py separately to actually feed it):
    python run_paper_trading.py --symbol BTC/USDT --timeframe 1h --mode live
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from config.settings import settings
from ml.predict import load_training_artifacts
from paper_trading.engine import PaperTradingEngine
from paper_trading.metrics import print_summary
from pipeline.data_loader import load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _model_dir_for(symbol: str, timeframe: str) -> str:
    # Matches this repo's models_artifacts/<SYMBOL>_<timeframe> convention
    # (see models_artifacts/BTCUSDT_1h etc.).
    return f"models_artifacts/{symbol.replace('/', '')}_{timeframe}"


def main():
    parser = argparse.ArgumentParser(description="Run the paper trading engine.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default=settings.exchange.default_exchange)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--warmup-bars", type=int, default=250)
    parser.add_argument("--starting-cash", type=float, default=None)
    parser.add_argument("--model-dir", default=None, help="Defaults to models_artifacts/<SYMBOL>_<timeframe>")
    parser.add_argument("--holdout-frac", type=float, default=0.15,
                         help="Replay only the trailing fraction of history (default matches "
                              "run_backtest_from_db.py's default, so the two are comparable). "
                              "Pass 1.0 to replay the full dataset (in-sample on most of it — "
                              "only for debugging, not for judging trading performance).")
    args = parser.parse_args()

    model_dir = args.model_dir or _model_dir_for(args.symbol, args.timeframe)
    logger.info("Loading model artifacts from %s", model_dir)
    artifacts = load_training_artifacts(model_dir)

    engine = PaperTradingEngine(
        symbol=args.symbol, timeframe=args.timeframe, exchange=args.exchange,
        model=artifacts["ensemble"], feature_columns=artifacts["feature_columns"],
        calibrator=artifacts["calibrator"], starting_cash=args.starting_cash,
    )

    if args.mode == "replay":
        df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
        n = len(df)
        holdout_start = int(n * (1 - args.holdout_frac))
        # keep warmup_bars of PRE-holdout history so the first traded bar
        # already has warmed-up indicators, but only bars from
        # holdout_start onward are ever passed to on_bar as a "new" bar —
        # run_replay's own loop starts at index warmup_bars within
        # whatever frame it's given, so slicing here is enough; nothing
        # in engine.py needs to change.
        slice_start = max(0, holdout_start - args.warmup_bars)
        df_replay = df.iloc[slice_start:].reset_index(drop=True)
        logger.info("Replaying %d bars for %s %s (holdout_frac=%.2f, %d of %d total candles, "
                    "%d reserved for warmup)...",
                    len(df_replay) - args.warmup_bars, args.symbol, args.timeframe,
                    args.holdout_frac, len(df_replay), n, args.warmup_bars)
        engine.run_replay(df_replay, warmup_bars=args.warmup_bars)
        print_summary(engine.portfolio, periods_per_year=_periods_per_year(args.timeframe))
    else:
        history_loader = lambda: load_ohlcv(  # noqa: E731
            args.symbol, args.timeframe, exchange=args.exchange, limit=args.warmup_bars,
        )
        logger.info("Starting live paper trading for %s %s (Ctrl+C to stop)...", args.symbol, args.timeframe)
        try:
            asyncio.run(engine.run_live(history_loader, warmup_bars=args.warmup_bars))
        except KeyboardInterrupt:
            pass
        finally:
            print_summary(engine.portfolio, periods_per_year=_periods_per_year(args.timeframe))


def _periods_per_year(timeframe: str) -> int:
    return {
        "1m": 60 * 24 * 365, "5m": 12 * 24 * 365, "15m": 4 * 24 * 365,
        "1h": 24 * 365, "4h": 6 * 365, "1d": 365,
    }.get(timeframe, 8760)


if __name__ == "__main__":
    main()