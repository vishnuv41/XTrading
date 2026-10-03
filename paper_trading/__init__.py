"""
paper_trading
--------------
Simulated (no real orders) trading engine that consumes
inference.realtime_pipeline.run_realtime_pipeline() output, executes
BUY/SELL/HOLD against a virtual portfolio with fees/slippage/SL/TP/
timeout, and logs every prediction and executed trade to PostgreSQL
(prediction_log, trade_log — see database/migrations/002_paper_trading.sql).

Quick start
-----------
    from ml.predict import load_training_artifacts
    from pipeline.data_loader import load_ohlcv
    from paper_trading.engine import PaperTradingEngine

    artifacts = load_training_artifacts("models_artifacts/BTCUSDT_1h")
    df = load_ohlcv("BTC/USDT", "1h", limit=1000)

    engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h",
        model=artifacts["ensemble"], feature_columns=artifacts["feature_columns"],
        calibrator=artifacts["calibrator"],
    )
    results = engine.run_replay(df, warmup_bars=250)

    from paper_trading.metrics import print_summary
    print_summary(engine.portfolio)
"""

from paper_trading.engine import PaperTradingEngine
from paper_trading.portfolio import VirtualPortfolio
from paper_trading.metrics import compute_metrics, print_summary

__all__ = ["PaperTradingEngine", "VirtualPortfolio", "compute_metrics", "print_summary"]
