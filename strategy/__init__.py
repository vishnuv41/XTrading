"""
strategy package

    trend.py            - combines EMA/MACD/Supertrend/Ichimoku into a
                           single trend_bias + trend_strength
    entry.py             - regime-aware BUY/SELL/HOLD entry rules
                           (trend-following in trending regimes,
                           mean-reversion in ranging regimes, HOLD in
                           mixed/uncertain regimes)
    exit.py              - signal-based exit conditions for open long/short
                           positions (trend flip, momentum exhaustion,
                           regime deterioration) — separate from hard
                           SL/TP price levels, which live in risk_engine/
    signal.py            - orchestrates the above into one unified
                           per-candle signal with a rule-based confidence
                           score
    multi_timeframe.py   - confirms a lower-timeframe signal against
                           higher-timeframe trend bias before it's acted on

Typical single-timeframe pipeline:
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy import generate_signals

    df = calculate_all_indicators(df)
    df = calculate_market_state(df)
    df = generate_signals(df)

Add multi-timeframe confirmation on top when higher-timeframe data is
available (see multi_timeframe.py's __main__ block for a full example).
"""

from .trend import calculate_trend_bias
from .entry import generate_entry_signal
from .exit import calculate_exit_signals, get_exit_decision
from .signal import generate_signals, summarize_latest_signal
from .multi_timeframe import confirm_multi_timeframe
from .breakout import generate_breakout_signal
from .momentum import generate_momentum_signal
from .pullback import generate_pullback_signal
from .cross_asset import generate_cross_asset_signal
from .registry import STRATEGY_REGISTRY, run_all_strategies, latest_signals
from .fusion import FusedSignal, fuse_signals, fuse_from_dataframe

__all__ = [
    "calculate_trend_bias",
    "generate_entry_signal",
    "calculate_exit_signals", "get_exit_decision",
    "generate_signals", "summarize_latest_signal",
    "confirm_multi_timeframe",
    "generate_breakout_signal",
    "generate_momentum_signal",
    "generate_pullback_signal",
    "generate_cross_asset_signal",
    "STRATEGY_REGISTRY", "run_all_strategies", "latest_signals",
    "FusedSignal", "fuse_signals", "fuse_from_dataframe",
]