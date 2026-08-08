"""
paper_trading/models.py
--------------------------
Plain dataclasses shared across the paper trading package. Kept
DB-agnostic and pandas-agnostic on purpose — portfolio.py, execution.py,
and metrics.py all operate on these directly, and db_logger.py is the
only module that knows how to turn them into SQL rows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OpenPosition:
    """A currently-open simulated position."""

    trade_id: str
    exchange: str
    symbol: str
    timeframe: str
    side: str  # 'long' | 'short'
    entry_ts: datetime
    entry_price: float  # fill price, after slippage
    size: float  # units of the asset
    stop_loss: float
    take_profit: float
    risk_pct: float  # fraction of equity risked, for portfolio_risk.py's heat calc
    entry_fee: float
    bars_held: int = 0

    @staticmethod
    def new(exchange, symbol, timeframe, side, entry_ts, entry_price,
            size, stop_loss, take_profit, risk_pct, entry_fee) -> "OpenPosition":
        return OpenPosition(
            trade_id=str(uuid.uuid4()),
            exchange=exchange, symbol=symbol, timeframe=timeframe, side=side,
            entry_ts=entry_ts, entry_price=entry_price, size=size,
            stop_loss=stop_loss, take_profit=take_profit,
            risk_pct=risk_pct, entry_fee=entry_fee,
        )

    @property
    def direction(self) -> int:
        """+1 for long, -1 for short — matches ml/backtest.py's sign convention."""
        return 1 if self.side == "long" else -1

    @property
    def notional_value(self) -> float:
        return self.size * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * self.direction

    def to_risk_engine_dict(self) -> dict:
        """Shape expected by risk_engine.portfolio_risk.check_new_trade_allowed's open_positions arg."""
        return {"symbol": self.symbol, "risk_pct": self.risk_pct}


@dataclass
class ClosedTrade:
    """Record of a completed OPEN->CLOSE round trip."""

    trade_id: str
    exchange: str
    symbol: str
    timeframe: str
    side: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    size: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'stop_loss' | 'take_profit' | 'timeout' | 'signal_flip'
    entry_fee: float
    exit_fee: float
    bars_held: int
    realized_pnl: float
    realized_pnl_pct: float

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


@dataclass
class PredictionRecord:
    """One row of what the pipeline output for a given bar, whether or not it was acted on."""

    exchange: str
    symbol: str
    timeframe: str
    ts: datetime
    prediction: str  # 'BUY' | 'SELL' | 'HOLD'
    confidence: Optional[float] = None
    prob_down: Optional[float] = None
    prob_flat: Optional[float] = None
    prob_up: Optional[float] = None
    market_state: Optional[str] = None
    trend: Optional[str] = None
    volatility: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    position_size: Optional[float] = None
    risk_pct: Optional[float] = None
    notional_value: Optional[float] = None
    risk_block_reason: Optional[str] = None
    executed: bool = False

    @staticmethod
    def from_pipeline_result(exchange: str, symbol: str, timeframe: str,
                              ts: datetime, result: dict) -> "PredictionRecord":
        """Build directly from inference.realtime_pipeline.run_realtime_pipeline's return dict."""
        return PredictionRecord(
            exchange=exchange, symbol=symbol, timeframe=timeframe, ts=ts,
            prediction=result["prediction"], confidence=result.get("confidence"),
            market_state=result.get("market_state"), trend=result.get("trend"),
            volatility=result.get("volatility"), entry_price=result.get("entry"),
            stop_loss=result.get("stop_loss"), take_profit=result.get("take_profit"),
            risk_reward_ratio=result.get("risk_reward_ratio"),
            position_size=result.get("position_size"), risk_pct=result.get("risk_pct"),
            notional_value=result.get("notional_value"),
            risk_block_reason=result.get("risk_block_reason"),
        )


@dataclass
class EquitySnapshot:
    ts: datetime
    cash: float
    equity: float  # cash + unrealized PnL of all open positions