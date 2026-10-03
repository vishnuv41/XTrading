"""
paper_trading/exit_manager.py
--------------------------------
Decides whether an OPEN position should exit on the current bar, and
why. Pure decision logic — no portfolio mutation, no I/O — so it's
trivially unit-testable and reusable if a future backtest wants the
same intrabar SL/TP-touch semantics as the live paper engine.

Intrabar fill assumption: we only have OHLC for the bar, not tick data,
so if a bar's range touches BOTH stop and target we can't know which
happened first. We resolve that ambiguity conservatively — SL takes
priority over TP — since assuming the best-case fill on an ambiguous
bar would flatter the simulation's results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from paper_trading.models import OpenPosition


@dataclass
class ExitDecision:
    should_exit: bool
    reason: Optional[str] = None  # 'stop_loss' | 'take_profit' | 'timeout'
    fill_price: Optional[float] = None  # price the exit would print at, pre-slippage


def update_trailing_stop(
    position: OpenPosition,
    supertrend_value: float,
    supertrend_direction: int,
) -> Optional[float]:
    """
    Compute a trailed stop-loss level for an open position using the
    current bar's Supertrend line (risk_engine.stoploss.calculate_supertrend_stop
    implements the same "use Supertrend as the stop" idea; this is the
    per-bar update wrapper that decides whether to actually move the
    stop). Pure function — does not mutate `position`; the caller
    applies the returned value if not None.

    Only ever tightens the stop (moves it toward the current price, in
    the trade's favor) — a trailing stop that could also loosen isn't a
    trailing stop, it's just noise. If the trend has flipped against the
    position (supertrend_direction no longer agrees with position.side),
    this returns None rather than raising: exit.py's signal-flip/exit
    logic is what should close the trade in that case, not this
    function silently leaving a stale or wrong-direction stop in place.

    Args:
        position: the open position.
        supertrend_value: current bar's Supertrend line value.
        supertrend_direction: current bar's Supertrend direction, +1 or -1.

    Returns:
        The new stop-loss level if it should move, else None (no change —
        either the trend no longer agrees with the position side, or the
        new level would be less favorable than the current stop).
    """
    expected_dir = 1 if position.side == "long" else -1
    if supertrend_direction != expected_dir:
        return None

    if position.side == "long":
        if supertrend_value > position.stop_loss:
            return supertrend_value
    else:  # short
        if supertrend_value < position.stop_loss:
            return supertrend_value

    return None


def check_exit(position: OpenPosition, bar_high: float, bar_low: float,
                timeout_bars: int) -> ExitDecision:
    """
    Args:
        position: the open position to evaluate.
        bar_high, bar_low: current bar's High/Low (the bar AFTER entry —
            callers should not check exits on the same bar a position
            was just opened on, to avoid same-bar entry+exit ambiguity).
        timeout_bars: force-close after this many bars regardless of
            SL/TP (paper_trading.config.settings.timeout_bars).

    Returns:
        ExitDecision. fill_price is the raw stop/target/close price —
        execution.py applies slippage on top of this, exit_manager
        doesn't know about fees/slippage.
    """
    if position.side == "long":
        # SL priority on an ambiguous bar (see module docstring).
        if bar_low <= position.stop_loss:
            return ExitDecision(True, "stop_loss", position.stop_loss)
        if bar_high >= position.take_profit:
            return ExitDecision(True, "take_profit", position.take_profit)
    elif position.side == "short":
        if bar_high >= position.stop_loss:
            return ExitDecision(True, "stop_loss", position.stop_loss)
        if bar_low <= position.take_profit:
            return ExitDecision(True, "take_profit", position.take_profit)
    else:
        raise ValueError(f"Unknown side '{position.side}' on position {position.trade_id}")

    if position.bars_held >= timeout_bars:
        # Timeout exits at... we don't have a "current price" here by
        # design (that's the caller's bar close) — signal timeout and
        # let the caller supply the fill price (bar close), since this
        # function only sees high/low, not close.
        return ExitDecision(True, "timeout", None)

    return ExitDecision(False)
