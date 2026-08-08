"""
paper_trading/engine.py
---------------------------
Ties everything together. One PaperTradingEngine instance owns one
VirtualPortfolio and drives it bar-by-bar, either from a live Redis
candle subscription (run_live) or by replaying historical OHLCV
(run_replay, useful for testing and for "what if I'd been paper-trading
this model since date X").

Per-bar order of operations (on_bar), and why:
  1. Check SL/TP/timeout exits for any open position using the NEW
     bar's OHLC — a position opened on a previous bar's close should be
     checked against subsequent bars before anything else happens.
  2. Run the existing, unmodified inference.realtime_pipeline against
     the data up to and including this bar, passing this engine's
     current open positions / equity / today's realized P&L into it so
     the risk engine's portfolio-level gates (max open trades, daily
     loss circuit breaker, portfolio heat) see live paper-trading state,
     not defaults.
  3. If a position is still open and the new signal is the opposite
     direction, optionally close it (signal_flip exit — see
     paper_trading.config.settings.allow_signal_flip_exit).
  4. Attempt to open a new position from the (possibly risk-downgraded)
     prediction. The duplicate-position guard lives in
     VirtualPortfolio.open_position — a symbol with an already-open
     position simply won't get a second one.
  5. Log the prediction (always — HOLD, blocked, or executed) and any
     trade events, then mark the portfolio to market.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd

from database.redis_cache import subscribe_candles
from inference.realtime_pipeline import run_realtime_pipeline
from paper_trading import db_logger
from paper_trading.config import settings as pt_settings
from paper_trading.execution import ExecutionSimulator
from paper_trading.models import PredictionRecord
from paper_trading.portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model,
        feature_columns: list,
        calibrator=None,
        exchange: str = "binance",
        starting_cash: Optional[float] = None,
        correlation_matrix=None,
        db_engine=None,
        persist_to_db: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange
        self.model = model
        self.feature_columns = feature_columns
        self.calibrator = calibrator
        self.correlation_matrix = correlation_matrix
        self.db_engine = db_engine  # None -> db_logger uses database.connection.get_engine()
        self.persist_to_db = persist_to_db

        self.portfolio = VirtualPortfolio(starting_cash or pt_settings.starting_cash)
        self.execution = ExecutionSimulator(exchange=exchange)

    # ------------------------------------------------------------------
    # Core per-bar step
    # ------------------------------------------------------------------

    def on_bar(self, df_window: pd.DataFrame) -> dict:
        """
        df_window: OHLCV history ending at (and including) the new bar
        to process — same shape inference.realtime_pipeline.run_realtime_pipeline
        expects (timestamp/open/high/low/close/volume, ascending).
        Needs enough history for indicator warm-up (200+ bars recommended).

        Returns a small summary dict for logging/CLI output; the
        authoritative state lives in self.portfolio.
        """
        latest = df_window.iloc[-1]
        ts = pd.Timestamp(latest["timestamp"]) if "timestamp" in df_window.columns else df_window.index[-1]
        bar_high, bar_low, bar_close = float(latest["high"]), float(latest["low"]), float(latest["close"])
        prices = {self.symbol: bar_close}

        # --- Step 1: exit checks on the new bar, before anything else ---
        exit_trade = self.execution.check_and_close(
            self.portfolio, self.symbol, ts, bar_high, bar_low, bar_close,
        )
        if exit_trade and self.persist_to_db:
            db_logger.log_trade_close(
                exit_trade, self.portfolio.cash, self.portfolio.equity(prices), engine=self.db_engine,
            )

        # --- Step 2: fresh prediction, with live portfolio state fed into the risk engine ---
        open_positions_for_risk = [p.to_risk_engine_dict() for p in self.portfolio.open_positions.values()]
        result = run_realtime_pipeline(
            df=df_window,
            model=self.model,
            feature_columns=self.feature_columns,
            calibrator=self.calibrator,
            symbol=self.symbol,
            account_equity=self.portfolio.equity(prices),
            open_positions=open_positions_for_risk,
            daily_pnl_pct=self.portfolio.daily_pnl_pct(as_of=ts),
            correlation_matrix=self.correlation_matrix,
        )
        prediction = result["prediction"]

        # --- Step 3: opposite-signal exit (optional) ---
        flip_trade = self.execution.close_on_signal_flip(
            self.portfolio, self.symbol, ts, bar_close, prediction,
        )
        if flip_trade and self.persist_to_db:
            db_logger.log_trade_close(
                flip_trade, self.portfolio.cash, self.portfolio.equity(prices), engine=self.db_engine,
            )

        # --- Step 4: attempt entry ---
        risk = {
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
            "position_size": result.get("position_size"),
            "risk_pct": result.get("risk_pct"),
            "entry_price": result.get("entry"),
        }
        opened = self.execution.open_from_prediction(
            self.portfolio, self.symbol, self.timeframe, ts, prediction, risk,
        )
        executed = opened is not None
        if opened and self.persist_to_db:
            db_logger.log_trade_open(
                opened, self.portfolio.cash, self.portfolio.equity(prices), engine=self.db_engine,
            )

        # --- Step 5: always log the prediction, then mark to market ---
        pred_record = PredictionRecord.from_pipeline_result(
            self.exchange, self.symbol, self.timeframe, ts, result,
        )
        pred_record.executed = executed
        if self.persist_to_db:
            db_logger.log_prediction(pred_record, engine=self.db_engine)

        snapshot = self.portfolio.mark_to_market(ts, prices)

        return {
            "ts": ts, "prediction": prediction, "executed": executed,
            "equity": snapshot.equity, "cash": snapshot.cash,
            "closed_trade": exit_trade or flip_trade, "opened_position": opened,
        }

    # ------------------------------------------------------------------
    # Historical replay (offline testing / "what-if since date X")
    # ------------------------------------------------------------------

    def run_replay(self, df: pd.DataFrame, warmup_bars: int = 250, step: int = 1) -> list[dict]:
        """
        Replay historical OHLCV bar-by-bar as if it were arriving live.
        `warmup_bars` is how much history the FIRST pipeline call gets
        (must be >= the longest indicator lookback — 250 covers this
        repo's SMA200/EMA200). Each subsequent call gets one more bar,
        exactly as a live feed would deliver it.
        """
        if len(df) <= warmup_bars:
            raise ValueError(f"Need more than {warmup_bars} rows to replay; got {len(df)}.")

        results = []
        for i in range(warmup_bars, len(df), step):
            window = df.iloc[: i + 1]
            results.append(self.on_bar(window))
        return results

    # ------------------------------------------------------------------
    # Live mode (subscribes to Person 1's Redis candle fanout)
    # ------------------------------------------------------------------

    async def run_live(self, history_loader, warmup_bars: int = 250) -> None:
        """
        Subscribes to database.redis_cache.subscribe_candles for this
        engine's symbol/timeframe and calls on_bar() for every new
        closed candle forever.

        `history_loader`: zero-arg callable returning a fresh warm-up
        DataFrame (e.g. `lambda: pipeline.data_loader.load_ohlcv(symbol,
        timeframe, limit=warmup_bars)`), re-called on every new candle
        to rebuild the rolling window. Injected rather than imported
        directly so tests can supply synthetic history without a live DB.
        """
        window = history_loader()
        if len(window) < warmup_bars:
            logger.warning("Warm-up window has only %d rows (<%d requested); indicators may be NaN early on.",
                           len(window), warmup_bars)

        async for candle in subscribe_candles(self.symbol, self.timeframe):
            new_row = pd.DataFrame([{
                "timestamp": pd.to_datetime(candle["ts"], utc=True),
                "open": candle["open"], "high": candle["high"],
                "low": candle["low"], "close": candle["close"], "volume": candle["volume"],
            }])
            window = pd.concat([window, new_row], ignore_index=True)
            window = window.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
            # Keep the rolling window bounded so indicator calc doesn't
            # grow unboundedly over a long-running live process.
            if len(window) > warmup_bars * 4:
                window = window.iloc[-warmup_bars * 4:]

            try:
                summary = self.on_bar(window)
                logger.info("Bar %s: %s (executed=%s) equity=%.2f",
                            summary["ts"], summary["prediction"], summary["executed"], summary["equity"])
            except Exception:
                logger.exception("on_bar failed for candle at %s; continuing live loop.", candle.get("ts"))