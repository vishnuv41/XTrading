"""
paper_trading/execution.py
------------------------------
Turns a trading decision (BUY/SELL/HOLD, or an exit_manager.ExitDecision)
into an actual fill against a VirtualPortfolio: applies slippage to the
reference price, computes the fee, and calls portfolio.open_position /
close_position. No real orders are ever sent anywhere — this is the
entire "exchange" as far as the rest of the system is concerned.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from paper_trading.config import settings as pt_settings
from paper_trading.exit_manager import ExitDecision, check_exit
from paper_trading.models import ClosedTrade, OpenPosition
from paper_trading.portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)


def _apply_slippage(price: float, side: str, direction: str) -> float:
    """
    direction: 'entry' or 'exit'. Slippage always moves the fill AGAINST
    the trader:
      - long entry (buy)  -> fills higher
      - long exit  (sell) -> fills lower
      - short entry (sell)-> fills lower
      - short exit (buy)  -> fills higher
    """
    bps = pt_settings.slippage_bps / 10_000
    if side == "long":
        return price * (1 + bps) if direction == "entry" else price * (1 - bps)
    else:  # short
        return price * (1 - bps) if direction == "entry" else price * (1 + bps)


def _fee(notional: float) -> float:
    return abs(notional) * (pt_settings.fee_bps / 10_000)


class ExecutionSimulator:
    """
    Stateless w.r.t. any single call — all state lives in the
    VirtualPortfolio passed in, so one ExecutionSimulator instance can
    safely drive multiple portfolios/symbols if ever needed.
    """

    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def open_from_prediction(self, portfolio: VirtualPortfolio, symbol: str,
                              timeframe: str, ts: datetime, prediction: str,
                              risk: dict) -> Optional[OpenPosition]:
        """
        Attempt to open a position from a pipeline prediction + risk
        dict (inference.realtime_pipeline.run_realtime_pipeline's
        output). Returns None (and logs why) if the trade can't be
        opened — duplicate position, HOLD, missing risk info, or
        insufficient cash — rather than raising, since "can't trade
        this bar" is a normal, expected outcome the engine should just
        move past.
        """
        if prediction not in ("BUY", "SELL"):
            return None

        if portfolio.has_open_position(symbol):
            logger.debug("Skipping %s %s: position already open (duplicate-position guard).",
                         prediction, symbol)
            return None

        stop_loss = risk.get("stop_loss")
        take_profit = risk.get("take_profit")
        position_size = risk.get("position_size")
        risk_pct = risk.get("risk_pct")
        entry_price_ref = risk.get("entry_price")  # caller passes the pipeline's `entry`

        if stop_loss is None or take_profit is None or not position_size:
            logger.debug("Skipping %s %s: risk engine did not produce sizing (likely blocked).",
                         prediction, symbol)
            return None

        side = "long" if prediction == "BUY" else "short"
        fill_price = _apply_slippage(entry_price_ref, side, "entry")
        notional = position_size * fill_price
        fee = _fee(notional)

        position = OpenPosition.new(
            exchange=self.exchange, symbol=symbol, timeframe=timeframe, side=side,
            entry_ts=ts, entry_price=fill_price, size=position_size,
            stop_loss=stop_loss, take_profit=take_profit,
            risk_pct=risk_pct or 0.0, entry_fee=fee,
        )

        try:
            portfolio.open_position(position)
        except (ValueError,) as exc:
            # Insufficient cash — log and skip rather than crash the loop.
            logger.warning("Could not open %s %s: %s", prediction, symbol, exc)
            return None

        logger.info("OPEN %s %s @ %.6f size=%.6f SL=%.6f TP=%.6f fee=%.4f",
                    side.upper(), symbol, fill_price, position_size, stop_loss, take_profit, fee)
        return position

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------

    def check_and_close(self, portfolio: VirtualPortfolio, symbol: str, ts: datetime,
                         bar_high: float, bar_low: float, bar_close: float) -> Optional[ClosedTrade]:
        """
        Check the open position (if any) for symbol against this bar's
        OHLC and close it if SL/TP/timeout triggers. Increments
        bars_held either way (call once per bar for every open position,
        even ones that don't exit). Returns the ClosedTrade if one was
        closed, else None.
        """
        position = portfolio.get_open_position(symbol)
        if position is None:
            return None

        position.bars_held += 1

        decision: ExitDecision = check_exit(
            position, bar_high=bar_high, bar_low=bar_low,
            timeout_bars=pt_settings.timeout_bars,
        )
        if not decision.should_exit:
            return None

        raw_exit_price = decision.fill_price if decision.fill_price is not None else bar_close
        fill_price = _apply_slippage(raw_exit_price, position.side, "exit")
        fee = _fee(position.size * fill_price)

        trade = portfolio.close_position(
            symbol=symbol, exit_ts=ts, exit_price=fill_price,
            exit_fee=fee, exit_reason=decision.reason,
        )
        logger.info("CLOSE %s %s @ %.6f reason=%s pnl=%.4f (%.2f%%) fee=%.4f",
                    position.side.upper(), symbol, fill_price, decision.reason,
                    trade.realized_pnl, trade.realized_pnl_pct * 100, fee)
        return trade

    def close_on_signal_flip(self, portfolio: VirtualPortfolio, symbol: str, ts: datetime,
                              bar_close: float, new_prediction: str) -> Optional[ClosedTrade]:
        """
        If configured (paper_trading.config.settings.allow_signal_flip_exit)
        and an opposite-direction signal arrives while a position is
        open, close it at the current bar's close before evaluating the
        new entry. Returns the ClosedTrade if one was closed, else None.
        """
        if not pt_settings.allow_signal_flip_exit:
            return None

        position = portfolio.get_open_position(symbol)
        if position is None:
            return None

        wants_long = new_prediction == "BUY"
        wants_short = new_prediction == "SELL"
        is_opposite = (position.side == "long" and wants_short) or (position.side == "short" and wants_long)
        if not is_opposite:
            return None

        fill_price = _apply_slippage(bar_close, position.side, "exit")
        fee = _fee(position.size * fill_price)
        trade = portfolio.close_position(
            symbol=symbol, exit_ts=ts, exit_price=fill_price,
            exit_fee=fee, exit_reason="signal_flip",
        )
        logger.info("CLOSE %s %s @ %.6f reason=signal_flip pnl=%.4f fee=%.4f",
                    position.side.upper(), symbol, fill_price, trade.realized_pnl, fee)
        return trade