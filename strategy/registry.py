"""
strategy/registry.py
------------------------
Common interface + registry over all rule-based strategy modules
(entry.py's trend/mean-reversion branches, breakout.py, momentum.py, ...
future additions). This is what strategy/fusion.py iterates over, and
what any research/backtest script should use instead of importing each
strategy module by hand — adding a new strategy means adding one entry
here, not touching the fusion logic or callers.

Reuses the signal-dict shape strategy/signal.py's summarize_latest_signal
already established ({signal, confidence, reason, strategy, ...}) rather
than inventing a new representation — see that module's docstring for
the original rationale (transparent rule-count confidence, not the ML
model's calibrated probability).
"""

from __future__ import annotations

import pandas as pd

from .entry import generate_entry_signal
from .breakout import generate_breakout_signal
from .momentum import generate_momentum_signal
from .pullback import generate_pullback_signal


def _entry_adapter(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_signal" not in df.columns:
        df = generate_entry_signal(df)
    return df


def _breakout_adapter(df: pd.DataFrame) -> pd.DataFrame:
    if "breakout_signal" not in df.columns:
        df = generate_breakout_signal(df)
    return df


def _momentum_adapter(df: pd.DataFrame) -> pd.DataFrame:
    if "momentum_signal" not in df.columns:
        df = generate_momentum_signal(df)
    return df


def _pullback_adapter(df: pd.DataFrame) -> pd.DataFrame:
    if "pullback_signal" not in df.columns:
        df = generate_pullback_signal(df)
    return df


# name -> (df-mutating generator, signal column, reason column)
# 'regime_adaptive' covers both of entry.py's branches (trend-following
# in trending regimes, mean-reversion in ranging regimes) since they
# share one column pair and are mutually exclusive per-row by construction.
#
# 'cross_asset_confirmation' (strategy/cross_asset.py) is deliberately
# NOT in this registry: it needs OTHER assets' dataframes as input
# (reference_names, optionally reference_trend_bias), which doesn't fit
# the single-df generator signature every other entry here shares. Call
# it directly wherever pooled multi-asset data is available (e.g. the
# pooled BTC/ETH/SOL training/inference path), not through this registry.
STRATEGY_REGISTRY = {
    "regime_adaptive": (_entry_adapter, "entry_signal", "entry_reason"),
    "volatility_breakout": (_breakout_adapter, "breakout_signal", "breakout_reason"),
    "momentum_confirmation": (_momentum_adapter, "momentum_signal", "momentum_reason"),
    "pullback_trend": (_pullback_adapter, "pullback_signal", "pullback_reason"),
}


def run_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Run every registered strategy's generator over df, adding all their columns."""
    for _, (generator, _, _) in STRATEGY_REGISTRY.items():
        df = generator(df)
    return df


def latest_signals(df: pd.DataFrame) -> dict[str, dict]:
    """
    Run every registered strategy and return each one's signal for the
    LAST row in the common dict shape:
        {"signal": "BUY"/"SELL"/"HOLD", "confidence": None, "strategy": name, "reason": [...]}

    confidence is left None here — these rule strategies emit reason
    lists, not a calibrated probability; strategy/signal.py's rule-count
    confidence heuristic only applies to the 'regime_adaptive' entry
    (which already has its own trend_strength-based scoring). Callers
    that want a comparable confidence across strategies should derive
    one from len(reason) the same way strategy/signal.py does for
    mean-reversion entries, rather than this module guessing at it.
    """
    df = run_all_strategies(df)
    latest = df.iloc[-1]

    result = {}
    for name, (_, signal_col, reason_col) in STRATEGY_REGISTRY.items():
        result[name] = {
            "signal": latest[signal_col],
            "confidence": None,
            "strategy": name,
            "reason": latest[reason_col],
        }
    return result


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state

    n = 400
    rng = np.random.default_rng(11)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.random(n) * 1.5,
        "low": close - rng.random(n) * 1.5,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = calculate_market_state(df)

    signals = latest_signals(df)
    for name, sig in signals.items():
        print(f"{name:25s} -> {sig['signal']:5s} {sig['reason']}")
