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


_SELECT_ACTIVE_OPEN_POSITIONS = text("""
    SELECT t1.trade_id, t1.exchange, t1.symbol, t1.timeframe, t1.side, t1.ts as entry_ts,
           t1.price as entry_price, t1.size, t1.fee as entry_fee, t1.stop_loss, t1.take_profit,
           t1.cash_after
    FROM trade_log t1
    LEFT JOIN trade_log t2 ON t1.trade_id = t2.trade_id AND t2.action = 'CLOSE'
    WHERE t1.symbol = :symbol AND t1.timeframe = :timeframe AND t1.action = 'OPEN' AND t2.id IS NULL
      AND t1.ts >= '2026-09-06 14:00:00+00'
    ORDER BY t1.ts DESC
    LIMIT 1
""")

_COUNT_BARS_HELD = text("""
    SELECT COUNT(*) FROM prediction_log
    WHERE symbol = :symbol AND timeframe = :timeframe AND ts > :entry_ts
""")


def load_active_open_positions(symbol: str, timeframe: str, engine: Optional[Engine] = None) -> list[dict]:
    eng = _engine_or_default(engine)
    with eng.connect() as conn:
        rows = conn.execute(
            _SELECT_ACTIVE_OPEN_POSITIONS,
            {"symbol": symbol, "timeframe": timeframe}
        ).mappings().all()
        return [dict(r) for r in rows]


def count_bars_held_since(symbol: str, timeframe: str, entry_ts, engine: Optional[Engine] = None) -> int:
    eng = _engine_or_default(engine)
    with eng.connect() as conn:
        cnt = conn.execute(
            _COUNT_BARS_HELD,
            {"symbol": symbol, "timeframe": timeframe, "entry_ts": entry_ts}
        ).scalar()
        return cnt or 0


_SELECT_GLOBAL_REALIZED_PNL = text("""
    SELECT COALESCE(SUM(realized_pnl), 0.0)
    FROM trade_log
    WHERE action = 'CLOSE'
""")

_SELECT_ALL_ACTIVE_OPEN_POSITIONS = text("""
    SELECT t1.trade_id, t1.exchange, t1.symbol, t1.timeframe, t1.side, t1.ts as entry_ts,
           t1.price as entry_price, t1.size, t1.fee as entry_fee, t1.stop_loss, t1.take_profit
    FROM trade_log t1
    LEFT JOIN trade_log t2 ON t1.trade_id = t2.trade_id AND t2.action = 'CLOSE'
    WHERE t1.action = 'OPEN' AND t2.id IS NULL
      AND t1.ts >= '2026-09-06 14:00:00+00'
""")


def load_global_portfolio_state(starting_cash: float = 10000.0, leverage: float = 3.0, engine: Optional[Engine] = None) -> dict:
    eng = _engine_or_default(engine)
    with eng.connect() as conn:
        res_pnl = conn.execute(_SELECT_GLOBAL_REALIZED_PNL).scalar() or 0.0
        active_rows = conn.execute(_SELECT_ALL_ACTIVE_OPEN_POSITIONS).mappings().all()

        tied_capital = 0.0
        for r in active_rows:
            notional = float(r["size"]) * float(r["entry_price"])
            margin = notional / leverage if leverage else notional
            entry_fee = float(r["entry_fee"])
            tied_capital += (margin + entry_fee)

        total_realized_pnl = float(res_pnl)
        available_cash = starting_cash + total_realized_pnl - tied_capital
        global_equity = starting_cash + total_realized_pnl

        return {
            "starting_cash": starting_cash,
            "total_realized_pnl": total_realized_pnl,
            "available_cash": available_cash,
            "global_equity": global_equity,
            "active_positions": [dict(r) for r in active_rows],
        }


