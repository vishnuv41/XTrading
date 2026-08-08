"""
paper_trading/portfolio.py
-----------------------------
The virtual account: cash balance, open positions (keyed by symbol —
see the duplicate-position guard below), and the equity curve used by
metrics.py. execution.py is the only module that should mutate a
VirtualPortfolio; everything else reads from it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from paper_trading.models import ClosedTrade, EquitySnapshot, OpenPosition


class DuplicatePositionError(Exception):
    """Raised when the engine tries to open a second position in a symbol that already has one open."""


class VirtualPortfolio:
    def __init__(self, starting_cash: float):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        # One open position per symbol, max — this dict IS the
        # duplicate-position guard: open_position() raises if the key
        # already exists rather than silently overwriting/stacking.
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[EquitySnapshot] = []

    # ------------------------------------------------------------------
    # Position lifecycle (called by execution.py — not meant to be
    # called directly by the engine, so fee/slippage math lives in one
    # place).
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.open_positions

    def get_open_position(self, symbol: str) -> Optional[OpenPosition]:
        return self.open_positions.get(symbol)

    def open_position(self, position: OpenPosition) -> None:
        if position.symbol in self.open_positions:
            raise DuplicatePositionError(
                f"{position.symbol} already has an open position "
                f"({self.open_positions[position.symbol].trade_id}); "
                f"close it before opening another."
            )
        cost = position.notional_value + position.entry_fee
        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash for {position.symbol}: need {cost:.2f}, have {self.cash:.2f}"
            )
        self.cash -= cost
        self.open_positions[position.symbol] = position

    def close_position(self, symbol: str, exit_ts: datetime, exit_price: float,
                        exit_fee: float, exit_reason: str) -> ClosedTrade:
        position = self.open_positions.pop(symbol, None)
        if position is None:
            raise KeyError(f"No open position for {symbol}")

        gross_pnl = position.unrealized_pnl(exit_price)
        net_pnl = gross_pnl - position.entry_fee - exit_fee
        proceeds = position.notional_value + gross_pnl - exit_fee
        self.cash += proceeds

        trade = ClosedTrade(
            trade_id=position.trade_id, exchange=position.exchange, symbol=symbol,
            timeframe=position.timeframe, side=position.side,
            entry_ts=position.entry_ts, entry_price=position.entry_price,
            exit_ts=exit_ts, exit_price=exit_price, size=position.size,
            stop_loss=position.stop_loss, take_profit=position.take_profit,
            exit_reason=exit_reason, entry_fee=position.entry_fee, exit_fee=exit_fee,
            bars_held=position.bars_held,
            realized_pnl=net_pnl,
            realized_pnl_pct=net_pnl / (position.notional_value) if position.notional_value else 0.0,
        )
        self.closed_trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        """prices: {symbol: current_price}. Missing symbols contribute 0 (stale-price bar, not marked)."""
        total = 0.0
        for symbol, position in self.open_positions.items():
            if symbol in prices:
                total += position.unrealized_pnl(prices[symbol])
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.unrealized_pnl(prices)

    def mark_to_market(self, ts: datetime, prices: dict[str, float]) -> EquitySnapshot:
        """Record one equity-curve point. Call once per bar, after exits/entries for that bar are resolved."""
        snapshot = EquitySnapshot(ts=ts, cash=self.cash, equity=self.equity(prices))
        self.equity_curve.append(snapshot)
        return snapshot

    def realized_pnl_total(self) -> float:
        return sum(t.realized_pnl for t in self.closed_trades)

    def daily_pnl_pct(self, as_of: datetime) -> float:
        """
        Today's realized P&L as a fraction of starting-of-day equity —
        feeds run_realtime_pipeline's daily_pnl_pct circuit-breaker arg.
        Uses closed trades only (realized), matching the circuit
        breaker's intent of reacting to locked-in losses, not paper
        drawdown on still-open positions.
        """
        day = as_of.date()
        todays_pnl = sum(
            t.realized_pnl for t in self.closed_trades if t.exit_ts.date() == day
        )
        # Equity at start of day: walk equity_curve back to the last
        # snapshot from a prior day; fall back to starting_cash if none.
        start_of_day_equity = self.starting_cash
        for snap in reversed(self.equity_curve):
            if snap.ts.date() < day:
                start_of_day_equity = snap.equity
                break
        if start_of_day_equity <= 0:
            return 0.0
        return todays_pnl / start_of_day_equity