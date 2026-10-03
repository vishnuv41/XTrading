"""
paper_trading/engine.py
---------------------------
Ties everything together. One PaperTradingEngine instance owns one
VirtualPortfolio and drives it bar-by-bar, either from a live Redis
candle subscription (run_live) or by replaying historical OHLCV
(run_replay, useful for testing and for "what if I'd been paper-trading
this model since date X").

Per-bar order of operations (on_bar), and why:
  0. If enabled (paper_trading.config.settings.enable_trailing_stop),
     trail any open position's stop-loss using this bar's Supertrend
     value BEFORE checking exits — so a newly-tightened stop is what
     this bar's high/low actually gets checked against, not last bar's
     stale level. Off by default; see exit_manager.update_trailing_stop.
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
from indicators.trend.supertrend import calculate_supertrend
from inference.realtime_pipeline import run_realtime_pipeline
from paper_trading import db_logger
from paper_trading.config import settings as pt_settings
from paper_trading.execution import ExecutionSimulator
from paper_trading.exit_manager import update_trailing_stop
from paper_trading.models import OpenPosition, PredictionRecord
from paper_trading.portfolio import VirtualPortfolio
from risk_engine.protections import TradeOutcome

logger = logging.getLogger(__name__)

# Bar length per timeframe string, for the consecutive-loss cooldown window.
TIMEFRAME_DURATIONS = {
    "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1),
}


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

        if self.persist_to_db:
            self.restore_state_from_db()

    def restore_state_from_db(self) -> None:
        """
        Loads canonical global portfolio cash balance and unclosed (active) open positions
        from PostgreSQL trade_log, computes bars_held from prediction_log, and hydrates self.portfolio.
        """
        try:
            global_state = db_logger.load_global_portfolio_state(
                starting_cash=self.portfolio.starting_cash,
                leverage=pt_settings.leverage,
                engine=self.db_engine,
            )
            self.portfolio.cash = global_state["available_cash"]

            active_trades = db_logger.load_active_open_positions(
                self.symbol, self.timeframe, engine=self.db_engine
            )
            for row in active_trades:
                bars_held = db_logger.count_bars_held_since(
                    self.symbol, self.timeframe, row["entry_ts"], engine=self.db_engine
                )
                pos = OpenPosition(
                    trade_id=row["trade_id"],
                    exchange=row["exchange"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    side=row["side"],
                    entry_ts=row["entry_ts"],
                    entry_price=float(row["entry_price"]),
                    size=float(row["size"]),
                    stop_loss=float(row["stop_loss"]),
                    take_profit=float(row["take_profit"]),
                    risk_pct=0.0,
                    entry_fee=float(row["entry_fee"]),
                    leverage=pt_settings.leverage,
                    bars_held=bars_held,
                )
                self.portfolio.open_positions[self.symbol] = pos
                logger.info(
                    "Restored active %s position for %s @ %.6f (size=%.6f, SL=%.6f, TP=%.6f, bars_held=%d/48, cash=%.2f)",
                    pos.side.upper(), pos.symbol, pos.entry_price, pos.size, pos.stop_loss, pos.take_profit, pos.bars_held, self.portfolio.cash
                )
            logger.info("Restored global portfolio cash = %.2f (total_realized_pnl = %.2f)", self.portfolio.cash, global_state["total_realized_pnl"])
        except Exception as exc:
            logger.warning("Could not restore state from DB for %s: %s", self.symbol, exc)


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

        # --- Step 0: trail the stop-loss using this bar's Supertrend, before checking exits ---
        # Must run before Step 1: the whole point of a trailing stop is
        # that THIS bar's tightened level is what gets checked against
        # THIS bar's high/low, not last bar's stale level.
        if pt_settings.enable_trailing_stop:
            position = self.portfolio.open_positions.get(self.symbol)
            if position is not None:
                st_df = calculate_supertrend(df_window)
                st_row = st_df.iloc[-1]
                new_stop = update_trailing_stop(
                    position,
                    supertrend_value=float(st_row["supertrend"]),
                    supertrend_direction=int(st_row["supertrend_direction"]),
                )
                if new_stop is not None:
                    logger.info(
                        "Trailing stop for %s %s: %.6f -> %.6f",
                        position.side.upper(), self.symbol, position.stop_loss, new_stop,
                    )
                    position.stop_loss = new_stop

        # --- Step 1: exit checks on the new bar, before anything else ---
        exit_trade = self.execution.check_and_close(
            self.portfolio, self.symbol, ts, bar_high, bar_low, bar_close,
        )
        if exit_trade and self.persist_to_db:
            db_logger.log_trade_close(
                exit_trade, self.portfolio.cash, self.portfolio.equity(prices), engine=self.db_engine,
            )

        # Sync global portfolio cash from DB to capture trades closed by other symbol daemons
        if self.persist_to_db:
            try:
                gstate = db_logger.load_global_portfolio_state(
                    starting_cash=self.portfolio.starting_cash,
                    leverage=pt_settings.leverage,
                    engine=self.db_engine,
                )
                self.portfolio.cash = gstate["available_cash"]
            except Exception as exc:
                logger.warning("Could not sync global cash in on_bar for %s: %s", self.symbol, exc)

        # --- Step 2: fresh prediction, with live portfolio state fed into the risk engine ---
        open_positions_for_risk = [p.to_risk_engine_dict() for p in self.portfolio.open_positions.values()]
        recent_trades = [
            TradeOutcome(closed_at=t.exit_ts, pnl=t.realized_pnl)
            for t in self.portfolio.closed_trades
            if t.symbol == self.symbol
        ]
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
            recent_trades=recent_trades,
            bar_duration=TIMEFRAME_DURATIONS.get(self.timeframe),
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

    def run_replay(self, df: pd.DataFrame, warmup_bars: int = 250, step: int = 1,
                    max_window_bars: Optional[int] = None) -> list[dict]:
        """
        Replay historical OHLCV bar-by-bar as if it were arriving live.
        `warmup_bars` is how much history the FIRST pipeline call gets
        (must be >= the longest indicator lookback — 250 covers this
        repo's SMA200/EMA200). Each subsequent call gets one more bar,
        exactly as a live feed would deliver it.

        `max_window_bars` bounds how much history each on_bar() call
        sees (default: 4x warmup_bars, matching run_live's rolling
        window cap). This matters more than it looks: run_realtime_pipeline
        recomputes the FULL indicator/regime stack (including a rolling
        Hurst-exponent regression in regime/trend_regime.py) from scratch
        on whatever window it's given, on every call. Feeding it an
        ever-growing window — bar 1 gets 250 rows, bar 8760 gets 8760
        rows — turns an O(n) replay into an O(n^2) one: a full year of
        1h bars (8760 rows) would eventually be recomputing every
        indicator over 8760 rows on every single one of those 8760
        calls. Capping the window keeps each call's cost roughly
        constant, at the cost of the Hurst/trend-regime calc only ever
        seeing recent history — which is what it needs anyway (its own
        lookback is a few hundred bars at most).
        """
        if len(df) <= warmup_bars:
            raise ValueError(f"Need more than {warmup_bars} rows to replay; got {len(df)}.")

        max_window = max_window_bars or (warmup_bars * 4)
        results = []
        for i in range(warmup_bars, len(df), step):
            window = df.iloc[max(0, i + 1 - max_window): i + 1]
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
