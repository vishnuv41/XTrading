"""
strategy/pullback.py
------------------------
Pullback Trend Strategy (new — see roadmap in the architecture review).
Hypothesis: chasing a fresh breakout means paying the worst price of the
move and risking a full-size stop right as volatility peaks. A shallow,
controlled pullback toward the fast moving average within an ALREADY
established trend, followed by momentum turning back in the trend's
direction, offers a better entry price with a tighter, more defensible
stop (just below/above the pullback low/high) than either chasing the
breakout or waiting for a full mean-reversion setup.

    established trend (trend_bias + trend_strength)
      -> controlled pullback (RSI cools off, price returns toward EMA20,
         but doesn't break down structurally)
      -> momentum recovery (RSI turns back up, MACD histogram troughs
         and turns, price reclaims EMA20)
      -> entry

Distinct from entry.py's trend-following rules: those fire on momentum
CONFIRMING an already-moving trend (RSI>55 + MACD bullish + Supertrend
up, all at once) — i.e. they catch continuation while the trend is
actively pushing. This strategy instead waits for the trend to PAUSE
(pullback) and looks for the specific moment it resumes, which by
construction fires at different bars than entry.py's rules (often
immediately after a stretch where entry.py would have been HOLD because
momentum wasn't confirming).

Required columns (all already produced upstream):
    close
    EMA20, trend_bias, trend_strength   (strategy/trend.py; run first)
    RSI14                                 (indicators/momentum/rsi.py)
    MACD_hist                             (indicators/momentum/macd.py)
    ATR14                                 (indicators/volatility/atr.py)
"""

import pandas as pd

from .trend import calculate_trend_bias

REQUIRED_COLUMNS = ["close", "EMA20", "trend_bias", "trend_strength", "RSI14", "MACD_hist", "ATR14"]


def _prepare(df: pd.DataFrame, rsi_lookback: int = 5, ema_distance_atr_mult: float = 1.5,
             min_trend_strength: float = 0.6, rsi_pullback_low: float = 40.0,
             rsi_pullback_high: float = 60.0) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["established_bull"] = (df["trend_bias"] == "bullish") & (df["trend_strength"] >= min_trend_strength)
    out["established_bear"] = (df["trend_bias"] == "bearish") & (df["trend_strength"] >= min_trend_strength)

    rsi_recent_min = df["RSI14"].rolling(rsi_lookback).min()
    rsi_recent_max = df["RSI14"].rolling(rsi_lookback).max()

    # "Cooled off but not broken": RSI dipped into the neutral band
    # recently (a real pullback happened) without the current reading
    # being extreme in either direction (which would suggest either no
    # pullback occurred yet, or it's turned into a full reversal).
    out["pulled_back_bull"] = (rsi_recent_min < rsi_pullback_high) & df["RSI14"].between(rsi_pullback_low, rsi_pullback_high + 10)
    out["pulled_back_bear"] = (rsi_recent_max > rsi_pullback_low) & df["RSI14"].between(rsi_pullback_low - 10, rsi_pullback_high)

    out["near_ema"] = (df["close"] - df["EMA20"]).abs() <= df["ATR14"] * ema_distance_atr_mult

    out["rsi_turning_up"] = df["RSI14"] > df["RSI14"].shift(1)
    out["rsi_turning_down"] = df["RSI14"] < df["RSI14"].shift(1)
    out["macd_hist_turning_up"] = df["MACD_hist"] > df["MACD_hist"].shift(1)
    out["macd_hist_turning_down"] = df["MACD_hist"] < df["MACD_hist"].shift(1)
    out["price_above_ema"] = df["close"] > df["EMA20"]
    out["price_below_ema"] = df["close"] < df["EMA20"]

    return out


def generate_pullback_signal(df: pd.DataFrame, ensure_trend_bias: bool = True, **kwargs) -> pd.DataFrame:
    """
    Add pullback entry-signal columns.

    Returns:
        df with new columns:
          'pullback_signal' - 'BUY', 'SELL', or 'HOLD'
          'pullback_reason' - list[str] of the specific conditions that fired
    """
    if ensure_trend_bias and "trend_bias" not in df.columns:
        df = calculate_trend_bias(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for pullback strategy: {missing}")

    flags = _prepare(df, **kwargs)

    signals = []
    reasons_list = []

    for i in df.index:
        f = flags.loc[i]
        reasons = []

        if bool(f["established_bull"]) and bool(f["pulled_back_bull"]) and bool(f["near_ema"]):
            reasons.append("Established uptrend with a controlled pullback toward EMA20")
            if bool(f["rsi_turning_up"]):
                reasons.append("RSI turning back up")
            if bool(f["macd_hist_turning_up"]):
                reasons.append("MACD histogram turning up")
            if bool(f["price_above_ema"]):
                reasons.append("Price reclaimed EMA20")
            if len(reasons) >= 3:
                signals.append("BUY")
                reasons_list.append(reasons)
                continue

        reasons = []
        if bool(f["established_bear"]) and bool(f["pulled_back_bear"]) and bool(f["near_ema"]):
            reasons.append("Established downtrend with a controlled pullback toward EMA20")
            if bool(f["rsi_turning_down"]):
                reasons.append("RSI turning back down")
            if bool(f["macd_hist_turning_down"]):
                reasons.append("MACD histogram turning down")
            if bool(f["price_below_ema"]):
                reasons.append("Price rejected below EMA20")
            if len(reasons) >= 3:
                signals.append("SELL")
                reasons_list.append(reasons)
                continue

        signals.append("HOLD")
        reasons_list.append([])

    df = df.copy()
    df["pullback_signal"] = signals
    df["pullback_reason"] = reasons_list
    return df


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators

    n = 500
    rng = np.random.default_rng(17)
    drift = np.linspace(0, 40, n)
    wiggle = 3 * np.sin(np.linspace(0, 18, n)) + rng.normal(0, 0.4, n)
    close = 100 + drift + wiggle
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.random(n) * 1.0,
        "low": close - rng.random(n) * 1.0,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = generate_pullback_signal(df)
    print(df["pullback_signal"].value_counts())
    non_hold = df[df["pullback_signal"] != "HOLD"]
    if len(non_hold):
        print(non_hold[["close", "pullback_signal", "pullback_reason"]].head(5).to_string())
