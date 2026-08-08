"""
paper_trading/db_logger.py
-----------------------------
Writes prediction_log and trade_log rows (see
database/migrations/002_paper_trading.sql). Uses the same shared sync
engine as the rest of the codebase (database.connection.get_engine())
by default, but every function accepts an optional `engine` override so
tests can point it at an in-memory SQLite engine instead of a live
Postgres instance — the SQL here is plain ANSI (parameterized INSERT +
ON CONFLICT DO NOTHING), which both understand identically.

All writes are idempotent (ON CONFLICT DO NOTHING on the same unique
keys the migration defines), so re-processing the same bar/trade event
after a crash-and-restart never double-logs.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from database.connection import get_engine
from paper_trading.models import ClosedTrade, OpenPosition, PredictionRecord

logger = logging.getLogger(__name__)


_INSERT_PREDICTION = text("""
    INSERT INTO prediction_log (
        exchange, symbol, timeframe, ts, prediction, confidence,
        prob_down, prob_flat, prob_up, market_state, trend, volatility,
        entry_price, stop_loss, take_profit, risk_reward_ratio,
        position_size, risk_pct, notional_value, risk_block_reason, executed
    ) VALUES (
        :exchange, :symbol, :timeframe, :ts, :prediction, :confidence,
        :prob_down, :prob_flat, :prob_up, :market_state, :trend, :volatility,
        :entry_price, :stop_loss, :take_profit, :risk_reward_ratio,
        :position_size, :risk_pct, :notional_value, :risk_block_reason, :executed
    )
    ON CONFLICT (exchange, symbol, timeframe, ts) DO NOTHING
""")

_INSERT_TRADE_OPEN = text("""
    INSERT INTO trade_log (
        trade_id, exchange, symbol, timeframe, side, action, ts, price, size, fee,
        stop_loss, take_profit, exit_reason, realized_pnl, realized_pnl_pct,
        bars_held, cash_after, equity_after
    ) VALUES (
        :trade_id, :exchange, :symbol, :timeframe, :side, 'OPEN', :ts, :price, :size, :fee,
        :stop_loss, :take_profit, NULL, NULL, NULL,
        NULL, :cash_after, :equity_after
    )
    ON CONFLICT (trade_id, action) DO NOTHING
""")

_INSERT_TRADE_CLOSE = text("""
    INSERT INTO trade_log (
        trade_id, exchange, symbol, timeframe, side, action, ts, price, size, fee,
        stop_loss, take_profit, exit_reason, realized_pnl, realized_pnl_pct,
        bars_held, cash_after, equity_after
    ) VALUES (
        :trade_id, :exchange, :symbol, :timeframe, :side, 'CLOSE', :ts, :price, :size, :fee,
        :stop_loss, :take_profit, :exit_reason, :realized_pnl, :realized_pnl_pct,
        :bars_held, :cash_after, :equity_after
    )
    ON CONFLICT (trade_id, action) DO NOTHING
""")


def _engine_or_default(engine: Optional[Engine]) -> Engine:
    return engine if engine is not None else get_engine()


def log_prediction(record: PredictionRecord, engine: Optional[Engine] = None) -> None:
    eng = _engine_or_default(engine)
    params = {
        "exchange": record.exchange, "symbol": record.symbol, "timeframe": record.timeframe,
        "ts": record.ts, "prediction": record.prediction, "confidence": record.confidence,
        "prob_down": record.prob_down, "prob_flat": record.prob_flat, "prob_up": record.prob_up,
        "market_state": record.market_state, "trend": record.trend, "volatility": record.volatility,
        "entry_price": record.entry_price, "stop_loss": record.stop_loss,
        "take_profit": record.take_profit, "risk_reward_ratio": record.risk_reward_ratio,
        "position_size": record.position_size, "risk_pct": record.risk_pct,
        "notional_value": record.notional_value, "risk_block_reason": record.risk_block_reason,
        "executed": record.executed,
    }
    with eng.begin() as conn:
        conn.execute(_INSERT_PREDICTION, params)


def log_trade_open(position: OpenPosition, cash_after: float, equity_after: float,
                    engine: Optional[Engine] = None) -> None:
    eng = _engine_or_default(engine)
    params = {
        "trade_id": position.trade_id, "exchange": position.exchange, "symbol": position.symbol,
        "timeframe": position.timeframe, "side": position.side, "ts": position.entry_ts,
        "price": position.entry_price, "size": position.size, "fee": position.entry_fee,
        "stop_loss": position.stop_loss, "take_profit": position.take_profit,
        "cash_after": cash_after, "equity_after": equity_after,
    }
    with eng.begin() as conn:
        conn.execute(_INSERT_TRADE_OPEN, params)


def log_trade_close(trade: ClosedTrade, cash_after: float, equity_after: float,
                     engine: Optional[Engine] = None) -> None:
    eng = _engine_or_default(engine)
    params = {
        "trade_id": trade.trade_id, "exchange": trade.exchange, "symbol": trade.symbol,
        "timeframe": trade.timeframe, "side": trade.side, "ts": trade.exit_ts,
        "price": trade.exit_price, "size": trade.size, "fee": trade.exit_fee,
        "stop_loss": trade.stop_loss, "take_profit": trade.take_profit,
        "exit_reason": trade.exit_reason, "realized_pnl": trade.realized_pnl,
        "realized_pnl_pct": trade.realized_pnl_pct, "bars_held": trade.bars_held,
        "cash_after": cash_after, "equity_after": equity_after,
    }
    with eng.begin() as conn:
        conn.execute(_INSERT_TRADE_CLOSE, params)