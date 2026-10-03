"""
strategy/momentum.py
------------------------
Momentum Confirmation Strategy (new — see roadmap in the architecture
review). Hypothesis: momentum signals derived from independent
mechanisms (price-oscillator, trend-of-averages, moving-average
crossover-state, volume-flow) agreeing simultaneously is a stronger
signal than any single one of them firing alone — each is individually
noisy and prone to false positives; requiring 3-of-4 agreement filters
most of that noise out at the cost of trading less often.

Distinct from entry.py's trend-following rules (which require an
established regime + trend_bias consensus) and breakout.py (which
requires a compression setup): this strategy is a pure momentum-
agreement filter with no regime or compression precondition, intended
to catch continuation moves that aren't at a fresh breakout and aren't
necessarily in a "confirmed" trending regime yet (e.g. early in a move,
before ADX/Hurst have caught up).

Required columns (all already produced upstream):
    close
    RSI14                (indicators/momentum/rsi.py)
    MACD_trend            (indicators/momentum/macd.py)
    EMA20                (indicators/trend/ema.py)
    OBV                  (indicators/volume/obv.py)
"""

import pandas as pd

REQUIRED_COLUMNS = ["close", "RSI14", "MACD_trend", "EMA20", "OBV"]


def _prepare(df: pd.DataFrame, roc_period: int = 10, ema_slope_period: int = 5,
             obv_slope_period: int = 10, min_agreement: int = 3) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["roc"] = df["close"].pct_change(roc_period) * 100
    out["ema_slope"] = df["EMA20"].diff(ema_slope_period)
    out["obv_slope"] = df["OBV"].diff(obv_slope_period)

    out["bull_votes"] = (
        (df["RSI14"] > 55).astype(int)
        + (df["MACD_trend"] == "bullish").astype(int)
        + (out["roc"] > 0).astype(int)
        + (out["ema_slope"] > 0).astype(int)
        + (out["obv_slope"] > 0).astype(int)
    )
    out["bear_votes"] = (
        (df["RSI14"] < 45).astype(int)
        + (df["MACD_trend"] == "bearish").astype(int)
        + (out["roc"] < 0).astype(int)
        + (out["ema_slope"] < 0).astype(int)
        + (out["obv_slope"] < 0).astype(int)
    )
    out["min_agreement"] = min_agreement
    return out


_VOTE_LABELS = {
    "rsi_bull": "RSI14 > 55 (bullish momentum)", "rsi_bear": "RSI14 < 45 (bearish momentum)",
    "macd": "MACD_trend agrees", "roc_bull": "Rate of change positive", "roc_bear": "Rate of change negative",
    "ema_slope_bull": "EMA20 sloping up", "ema_slope_bear": "EMA20 sloping down",
    "obv_bull": "OBV (volume flow) rising", "obv_bear": "OBV (volume flow) falling",
}


def generate_momentum_signal(df: pd.DataFrame, min_agreement: int = 3, **kwargs) -> pd.DataFrame:
    """
    Add momentum-confirmation entry-signal columns.

    Args:
        min_agreement: how many of the 5 independent momentum checks
            (RSI, MACD, ROC, EMA slope, OBV slope) must agree in the
            same direction to fire a signal (default 3-of-5).

    Returns:
        df with new columns:
          'momentum_signal' - 'BUY', 'SELL', or 'HOLD'
          'momentum_reason' - list[str] of the specific checks that agreed
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for momentum strategy: {missing}")

    flags = _prepare(df, min_agreement=min_agreement, **kwargs)

    signals = []
    reasons_list = []

    for i in df.index:
        row = df.loc[i]
        f = flags.loc[i]

        if f["bull_votes"] >= min_agreement:
            reasons = []
            if row["RSI14"] > 55:
                reasons.append(_VOTE_LABELS["rsi_bull"])
            if row["MACD_trend"] == "bullish":
                reasons.append(_VOTE_LABELS["macd"])
            if f["roc"] > 0:
                reasons.append(_VOTE_LABELS["roc_bull"])
            if f["ema_slope"] > 0:
                reasons.append(_VOTE_LABELS["ema_slope_bull"])
            if f["obv_slope"] > 0:
                reasons.append(_VOTE_LABELS["obv_bull"])
            signals.append("BUY")
            reasons_list.append(reasons)
        elif f["bear_votes"] >= min_agreement:
            reasons = []
            if row["RSI14"] < 45:
                reasons.append(_VOTE_LABELS["rsi_bear"])
            if row["MACD_trend"] == "bearish":
                reasons.append(_VOTE_LABELS["macd"])
            if f["roc"] < 0:
                reasons.append(_VOTE_LABELS["roc_bear"])
            if f["ema_slope"] < 0:
                reasons.append(_VOTE_LABELS["ema_slope_bear"])
            if f["obv_slope"] < 0:
                reasons.append(_VOTE_LABELS["obv_bear"])
            signals.append("SELL")
            reasons_list.append(reasons)
        else:
            signals.append("HOLD")
            reasons_list.append([])

    df = df.copy()
    df["momentum_signal"] = signals
    df["momentum_reason"] = reasons_list
    return df


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators

    n = 400
    rng = np.random.default_rng(5)
    close = np.cumsum(rng.normal(0.05, 1, n)) + 100
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.random(n) * 1.5,
        "low": close - rng.random(n) * 1.5,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = generate_momentum_signal(df)
    print(df["momentum_signal"].value_counts())
    non_hold = df[df["momentum_signal"] != "HOLD"]
    if len(non_hold):
        print(non_hold[["close", "momentum_signal", "momentum_reason"]].head(5).to_string())
