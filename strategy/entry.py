"""
Entry Signal

Generates BUY / SELL / HOLD entry signals. The key design decision here
(flagged repeatedly in earlier planning) is that a single rule set does
not work across all market conditions: trend-following rules lose money
in ranging markets, mean-reversion rules lose money in strong trends.

So this module branches on 'market_state' (from regime/market_state.py):

  trending_* states  -> trend-following entry rules
                         (EMA/MACD/Supertrend alignment + momentum
                         confirmation from trend.py's trend_bias)

  ranging_* states    -> mean-reversion entry rules
                         (Bollinger/Keltner band touch + RSI/Stochastic
                         extreme, betting on reversion to the mean)

  mixed_* states      -> no entries (HOLD). 'mixed' means trend_regime's
                         two independent signals (ADX, Hurst) disagreed —
                         deliberately the most conservative case, since
                         neither rule set has a real edge when the
                         underlying regime call itself is uncertain.

Each row's decision includes 'entry_reason', a list of the specific
conditions that fired, so the confidence-score layer (Phase 9 in the
original plan) and any human reviewing signals can see *why*, not just
the label.
"""

import pandas as pd

from .trend import calculate_trend_bias


TREND_REQUIRED_COLUMNS = [
    "trend_bias", "trend_strength", "RSI14", "MACD_trend", "supertrend_direction",
]
RANGE_REQUIRED_COLUMNS = [
    "close", "BB_upper", "BB_lower", "RSI14", "stoch_k", "stoch_d",
]


def _trend_following_entry(row) -> tuple[str, list[str]]:
    """Rules for trending_* market states."""
    reasons = []

    if row["trend_bias"] == "bullish" and row["trend_strength"] >= 0.6:
        if row["RSI14"] > 55:
            reasons.append("RSI > 55 confirms bullish momentum")
        if row["MACD_trend"] == "bullish":
            reasons.append("MACD bullish crossover")
        if row["supertrend_direction"] == 1:
            reasons.append("Supertrend in uptrend")
        if len(reasons) >= 2:
            return "BUY", reasons

    if row["trend_bias"] == "bearish" and row["trend_strength"] >= 0.6:
        if row["RSI14"] < 45:
            reasons.append("RSI < 45 confirms bearish momentum")
        if row["MACD_trend"] == "bearish":
            reasons.append("MACD bearish crossover")
        if row["supertrend_direction"] == -1:
            reasons.append("Supertrend in downtrend")
        if len(reasons) >= 2:
            return "SELL", reasons

    return "HOLD", []


def _mean_reversion_entry(row) -> tuple[str, list[str]]:
    """Rules for ranging_* market states."""
    reasons = []

    near_lower_band = row["close"] <= row["BB_lower"] * 1.002
    near_upper_band = row["close"] >= row["BB_upper"] * 0.998

    if near_lower_band:
        reasons.append("Price at/below lower Bollinger Band")
        if row["RSI14"] < 30:
            reasons.append("RSI oversold (<30)")
        if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
            reasons.append("Stochastic oversold and turning up")
        if len(reasons) >= 2:
            return "BUY", reasons

    reasons = []
    if near_upper_band:
        reasons.append("Price at/above upper Bollinger Band")
        if row["RSI14"] > 70:
            reasons.append("RSI overbought (>70)")
        if row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
            reasons.append("Stochastic overbought and turning down")
        if len(reasons) >= 2:
            return "SELL", reasons

    return "HOLD", []


def generate_entry_signal(df: pd.DataFrame, ensure_trend_bias: bool = True) -> pd.DataFrame:
    """
    Add entry signal columns to the dataframe.

    Args:
        df: DataFrame that must already have 'market_state' (from
            regime.calculate_market_state) plus the indicator columns
            each rule branch needs (see TREND_REQUIRED_COLUMNS /
            RANGE_REQUIRED_COLUMNS).
        ensure_trend_bias: If True (default) and 'trend_bias' isn't
            already present, calls calculate_trend_bias() automatically.
            Set False if you've already computed it and want to avoid
            recomputation.

    Returns:
        df with new columns:
          'entry_signal' - 'BUY', 'SELL', or 'HOLD'
          'entry_reason' - list[str] of the specific conditions that
                           fired (empty list for HOLD)
          'entry_rule_set' - which rule branch was used: 'trend_following',
                             'mean_reversion', or 'none' (mixed regime)
    """
    if "market_state" not in df.columns:
        raise ValueError("'market_state' column missing — run calculate_market_state() first")

    if ensure_trend_bias and "trend_bias" not in df.columns:
        df = calculate_trend_bias(df)

    missing_trend = [c for c in TREND_REQUIRED_COLUMNS if c not in df.columns]
    missing_range = [c for c in RANGE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_trend:
        raise ValueError(f"Missing columns for trend-following rules: {missing_trend}")
    if missing_range:
        raise ValueError(f"Missing columns for mean-reversion rules: {missing_range}")

    signals = []
    reasons_list = []
    rule_sets = []

    for _, row in df.iterrows():
        state = row["market_state"]

        if pd.isna(state):
            signals.append("HOLD")
            reasons_list.append([])
            rule_sets.append("none")
            continue

        if state.startswith("trending"):
            signal, reasons = _trend_following_entry(row)
            rule_sets.append("trend_following")
        elif state.startswith("ranging"):
            signal, reasons = _mean_reversion_entry(row)
            rule_sets.append("mean_reversion")
        else:  # mixed_*
            signal, reasons = "HOLD", []
            rule_sets.append("none")

        signals.append(signal)
        reasons_list.append(reasons)

    df["entry_signal"] = signals
    df["entry_reason"] = reasons_list
    df["entry_rule_set"] = rule_sets
    return df


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state

    n = 400
    rng = np.random.default_rng(9)
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
    df = generate_entry_signal(df)

    print(df[["close", "market_state", "entry_rule_set", "entry_signal"]].tail(15))
    print()
    print(df["entry_signal"].value_counts())
    non_hold = df[df["entry_signal"] != "HOLD"]
    if len(non_hold):
        print()
        print("Sample non-HOLD reasons:")
        print(non_hold[["entry_signal", "entry_reason"]].head(3).to_string())