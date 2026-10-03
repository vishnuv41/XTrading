"""
risk_engine package

    stoploss.py       - ATR-based fixed stops + Supertrend-based trailing
                        stop for open positions
    takeprofit.py     - risk-multiple take-profit (primary) + ATR
                        projection (sanity check)
    risk_reward.py    - computes R:R and downgrades trades below a
                        minimum ratio to HOLD before they reach sizing
    position_size.py  - fixed-fractional risk sizing, Kelly criterion
                        (fractional), and confidence/volatility-adjusted
                        final sizing
    portfolio_risk.py - correlation across open positions + aggregate
                        portfolio heat cap, gating new trades against
                        both

Typical pipeline (continuing on from strategy/):

    from strategy import generate_entry_signal
    from risk_engine.stoploss import calculate_stop_loss_series
    from risk_engine.takeprofit import calculate_take_profit_series
    from risk_engine.risk_reward import filter_by_risk_reward
    from risk_engine.position_size import calculate_adjusted_position_size
    from risk_engine.portfolio_risk import check_new_trade_allowed

    df = generate_entry_signal(df)                     # strategy/
    df = calculate_stop_loss_series(df)
    df = calculate_take_profit_series(df)
    df = filter_by_risk_reward(df, min_ratio=1.5)       # sets 'final_signal'
    # ... for each row where final_signal != 'HOLD':
    #     size = calculate_adjusted_position_size(...)
    #     decision = check_new_trade_allowed(open_positions, ...)
"""

from .stoploss import calculate_stop_loss, calculate_stop_loss_series, calculate_supertrend_stop
from .takeprofit import calculate_take_profit, calculate_take_profit_atr, calculate_take_profit_series
from .risk_reward import calculate_risk_reward, meets_minimum_risk_reward, filter_by_risk_reward
from .position_size import (
    calculate_kelly_fraction, calculate_fixed_fractional_size, calculate_adjusted_position_size,
)
from .portfolio_risk import (
    calculate_return_correlation, get_correlated_symbols,
    calculate_portfolio_heat, check_new_trade_allowed,
)
from .protections import TradeOutcome, check_protections, consecutive_loss_cooldown_active, max_trades_per_day_exceeded

__all__ = [
    "calculate_stop_loss", "calculate_stop_loss_series", "calculate_supertrend_stop",
    "calculate_take_profit", "calculate_take_profit_atr", "calculate_take_profit_series",
    "calculate_risk_reward", "meets_minimum_risk_reward", "filter_by_risk_reward",
    "calculate_kelly_fraction", "calculate_fixed_fractional_size", "calculate_adjusted_position_size",
    "calculate_return_correlation", "get_correlated_symbols",
    "calculate_portfolio_heat", "check_new_trade_allowed",
    "TradeOutcome", "check_protections", "consecutive_loss_cooldown_active", "max_trades_per_day_exceeded",
]