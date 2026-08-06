"""
Trend Bias

Combines several already-computed trend indicators into one directional
bias, rather than letting entry.py check EMA/ADX/Supertrend/Ichimoku
separately in an ad-hoc way scattered across the codebase. Every
downstream module (entry, exit, multi-timeframe confirmation) should
read trend_bias / trend_strength from here instead of re-deriving it.

Confirmations checked (each contributes +1 bullish or +1 bearish):
    1. EMA20 vs EMA50            - fast/slow crossover
    2. EMA50 vs EMA200           - "golden cross" style long-term bias
    3. MACD_trend                - bullish/bearish crossover state
    4. Supertrend_dir             - +1/-1 trend flip line
    5. Ichimoku_cloud_bullish     - price/cloud relationship (see note below)

trend_strength is the confirmation count normalized to 0-1 (agreement
ratio), NOT a statement about how strong the underlying price move is —
5/5 confirmations agreeing means high-conviction consensus, not
necessarily a large move. This distinction matters for entry.py, which
uses trend_strength as a confidence gate.
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "EMA20", "EMA50", "EMA200",
    "MACD_trend",
    "supertrend_direction",
    "ichimoku_tenkan", "ichimoku_kijun",
]


def calculate_trend_bias(df: pd.DataFrame, min_confirmations: int = 3) -> pd.DataFrame:
    """
    Add trend bias columns to the dataframe.

    Args:
        df: DataFrame that already has the indicator columns listed in
            REQUIRED_COLUMNS (i.e. calculate_all_indicators() has been
            run). Ichimoku's cloud check uses 'Ichimoku_cloud_bullish' as
            a proxy for "price above a bullish cloud" — for a stricter
            check you'd also want close > max(span_a, span_b), but that
            requires the forward-shifted cloud columns to be non-NaN at
            the current bar, which they often aren't near the most recent
            candles (Ichimoku projects forward). This looser proxy avoids
            losing the most recent (and most decision-relevant) rows to
            NaN.
        min_confirmations: How many of the 5 signals must agree in the
            same direction for trend_bias to be 'bullish'/'bearish'
            rather than 'neutral' (default 3, i.e. a simple majority).

    Returns:
        df with new columns:
          'trend_bullish_count' - how many of the 5 signals are bullish
          'trend_bearish_count' - how many of the 5 signals are bearish
          'trend_bias'          - 'bullish', 'bearish', or 'neutral'
          'trend_strength'      - 0-1, the winning side's agreement ratio
                                   (e.g. 4 of 5 bullish -> 0.8)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} — run calculate_all_indicators() first"
        )

    # Ichimoku cloud bullish proxy: tenkan > kijun, rather than close vs.
    # the forward-shifted senkou spans, which are NaN near the most
    # recent (and most decision-relevant) bars — see module docstring.
    ichimoku_cloud_bullish = df["ichimoku_tenkan"] > df["ichimoku_kijun"]

    bullish_votes = pd.DataFrame({
        "ema_fast_slow": df["EMA20"] > df["EMA50"],
        "ema_long_term": df["EMA50"] > df["EMA200"],
        "macd": df["MACD_trend"] == "bullish",
        "supertrend": df["supertrend_direction"] == 1,
        "ichimoku": ichimoku_cloud_bullish,
    })
    bearish_votes = ~bullish_votes

    bullish_count = bullish_votes.sum(axis=1)
    bearish_count = bearish_votes.sum(axis=1)
    total_signals = bullish_votes.shape[1]

    df["trend_bullish_count"] = bullish_count
    df["trend_bearish_count"] = bearish_count

    def classify(b, s):
        if b >= min_confirmations:
            return "bullish"
        if s >= min_confirmations:
            return "bearish"
        return "neutral"

    df["trend_bias"] = [
        classify(b, s) for b, s in zip(bullish_count, bearish_count)
    ]

    df["trend_strength"] = np.where(
        df["trend_bias"] == "bullish", bullish_count / total_signals,
        np.where(df["trend_bias"] == "bearish", bearish_count / total_signals, 0.5)
    )
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators

    n = 300
    rng = np.random.default_rng(5)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.random(n) * 1.5,
        "low": close - rng.random(n) * 1.5,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = calculate_trend_bias(df)
    print(df[["close", "trend_bullish_count", "trend_bearish_count",
              "trend_bias", "trend_strength"]].tail(10))
    print(df["trend_bias"].value_counts())