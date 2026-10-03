"""
risk_engine/protections.py
------------------------------
Loss-streak and overtrading circuit breakers, complementing the
existing daily-loss-% breaker in inference/realtime_pipeline._apply_risk_engine.

Why these are separate from the daily-% breaker: a losing streak that
stays under the daily-loss threshold (e.g. several -0.5% days in a row,
each individually "fine") can still indicate the model/regime edge has
broken down and the strategy is repeatedly paying costs for no edge.
Consecutive-loss cooldown catches that pattern directly rather than
waiting for cumulative % damage to trip the daily breaker. max_trades_
per_day is a blunter guard against a model firing unusually often
(e.g. a confidence-threshold regression, a data glitch causing noisy
predictions) racking up fee/slippage drag before anyone notices.

Pure functions — no portfolio/DB access — so paper_trading.engine and
any future live-execution path can call these with whatever trade
history representation they already have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence


@dataclass
class TradeOutcome:
    """Minimal shape needed for these checks — map your trade/log record to this."""
    closed_at: datetime
    pnl: float  # realized P&L, any consistent unit (currency or %); sign is what matters


def consecutive_loss_cooldown_active(
    recent_trades: Sequence[TradeOutcome],
    current_time: datetime,
    max_consecutive_losses: int,
    cooldown_bars: int,
    bar_duration: timedelta,
) -> tuple[bool, Optional[str]]:
    """
    Check whether a consecutive-loss cooldown should block new entries.

    Parameters
    ----------
    recent_trades : trade history for ONE symbol, ordered oldest -> newest
        (only the trailing losses matter, but pass enough history to
        find the start of the current streak).
    current_time : the bar being evaluated for a potential new entry.
    max_consecutive_losses : streak length that triggers the cooldown.
    cooldown_bars : how many bars the cooldown lasts once triggered.
    bar_duration : the timeframe's bar length (e.g. timedelta(hours=1)
        for 1h), used to convert cooldown_bars to a wall-clock window
        against the last losing trade's close time.

    Returns
    -------
    (is_blocked, reason) — reason is None when not blocked.
    """
    if not recent_trades or max_consecutive_losses <= 0:
        return False, None

    streak = 0
    last_loss_time = None
    for trade in reversed(recent_trades):
        if trade.pnl < 0:
            streak += 1
            if last_loss_time is None:
                last_loss_time = trade.closed_at
        else:
            break

    if streak < max_consecutive_losses or last_loss_time is None:
        return False, None

    cooldown_ends = last_loss_time + cooldown_bars * bar_duration
    if current_time < cooldown_ends:
        return True, (
            f"Consecutive-loss cooldown active: {streak} losses in a row "
            f"(>= {max_consecutive_losses}); cooldown until {cooldown_ends.isoformat()}."
        )
    return False, None


def max_trades_per_day_exceeded(
    recent_trades: Sequence[TradeOutcome],
    current_time: datetime,
    max_trades_per_day: int,
) -> tuple[bool, Optional[str]]:
    """
    Check whether the rolling-24h trade count for this symbol has hit
    the cap. Counts trades CLOSED in the trailing 24h as a simple proxy
    for "opened" (callers logging opens can pass open timestamps instead
    via the same TradeOutcome.closed_at field — name kept generic on
    purpose so either convention works).
    """
    if max_trades_per_day <= 0:
        return False, None

    window_start = current_time - timedelta(hours=24)
    count = sum(1 for t in recent_trades if window_start <= t.closed_at <= current_time)

    if count >= max_trades_per_day:
        return True, (
            f"Max trades per day reached ({count}/{max_trades_per_day} in trailing 24h)."
        )
    return False, None


def check_protections(
    recent_trades: Sequence[TradeOutcome],
    current_time: datetime,
    max_consecutive_losses: int,
    cooldown_bars: int,
    bar_duration: timedelta,
    max_trades_per_day: int,
) -> tuple[bool, Optional[str]]:
    """Convenience wrapper: run both checks, return the first that blocks."""
    blocked, reason = consecutive_loss_cooldown_active(
        recent_trades, current_time, max_consecutive_losses, cooldown_bars, bar_duration,
    )
    if blocked:
        return True, reason

    blocked, reason = max_trades_per_day_exceeded(recent_trades, current_time, max_trades_per_day)
    if blocked:
        return True, reason

    return False, None
