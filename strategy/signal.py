"""
Signal

The orchestrator: runs trend bias, entry signal, and exit signal
generation in the right order and combines them into one unified
per-candle signal, with a simple rule-based confidence score.

This is intentionally a THIN layer — it does not invent new trading
logic, it composes trend.py + entry.py + exit.py (which each already
have their own docstrings explaining the reasoning) into a single
structured output that the risk_engine and prediction API can consume
without needing to know about the internal rule-based scoring.

The confidence score here is a simple, transparent rule-count ratio —
NOT the same thing as the ML model's calibrated probability from
Phase 9 in the roadmap. Once the ML layer exists, treat this rule-based
confidence as one input feature to that model (or as a fallback signal
source when the model is still being trained/validated), not as the
final number shown to users.
"""

import pandas as pd

from .trend import calculate_trend_bias
from .entry import generate_entry_signal
from .exit import calculate_exit_signals


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full strategy pipeline and return df enriched with a single
    unified signal per row.

    Args:
        df: DataFrame with all indicators (calculate_all_indicators) and
            market_state (calculate_market_state) already computed.
            Trend bias, entry, and exit signals are computed here if not
            already present.

    Returns:
        df with new columns:
          'trend_bias', 'trend_strength'          (from trend.py, if not
                                                     already present)
          'entry_signal', 'entry_reason',
          'entry_rule_set'                         (from entry.py)
          'exit_long', 'exit_long_reason',
          'exit_short', 'exit_short_reason'        (from exit.py)
          'signal_confidence'                      - 0-1 rule-based score:
                                                       entry rows use
                                                       trend_strength (or a
                                                       fixed value for
                                                       mean-reversion
                                                       entries, which don't
                                                       have a trend_strength
                                                       equivalent); HOLD
                                                       rows get 0.
    """
    if "market_state" not in df.columns:
        raise ValueError("'market_state' column missing — run calculate_market_state() first")

    if "trend_bias" not in df.columns:
        df = calculate_trend_bias(df)

    if "entry_signal" not in df.columns:
        df = generate_entry_signal(df)

    if "exit_long" not in df.columns:
        df = calculate_exit_signals(df)

    def confidence(row):
        if row["entry_signal"] == "HOLD":
            return 0.0
        if row["entry_rule_set"] == "trend_following":
            return round(float(row["trend_strength"]), 3)
        if row["entry_rule_set"] == "mean_reversion":
            # No trend_strength equivalent for mean-reversion; use the
            # count of fired reasons out of the 3 possible mean-reversion
            # checks in entry.py's _mean_reversion_entry as a stand-in.
            return round(min(len(row["entry_reason"]) / 3, 1.0), 3)
        return 0.0

    df["signal_confidence"] = df.apply(confidence, axis=1)
    return df


def summarize_latest_signal(df: pd.DataFrame) -> dict:
    """
    Convenience function for the prediction API: return the most recent
    row's signal as a plain dict, ready to be JSON-serialized in a
    response like:

        {
          "signal": "BUY",
          "confidence": 0.8,
          "reasons": ["RSI > 55 confirms bullish momentum", ...],
          "rule_set": "trend_following",
          "market_state": "trending_low_vol"
        }
    """
    if len(df) == 0:
        raise ValueError("Empty dataframe")

    latest = df.iloc[-1]
    return {
        "signal": latest["entry_signal"],
        "confidence": float(latest["signal_confidence"]),
        "reasons": latest["entry_reason"],
        "rule_set": latest["entry_rule_set"],
        "market_state": latest["market_state"],
    }


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state

    n = 400
    rng = np.random.default_rng(21)
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
    df = generate_signals(df)

    print(df[["close", "market_state", "entry_signal", "signal_confidence"]].tail(15))
    print()
    print(summarize_latest_signal(df))