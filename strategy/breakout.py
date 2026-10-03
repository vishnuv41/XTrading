"""
strategy/breakout.py
------------------------
Volatility Breakout Strategy (new — see roadmap in the architecture
review). Hypothesis: price compression (low ATR/Donchian width relative
to recent history) tends to precede a directional expansion once
volume and trend confirmation show up; entering on the confirmed
breakout captures that expansion rather than guessing direction during
the quiet period.

    compression -> breakout of Donchian channel -> volume + ADX
    confirmation -> trend direction agreement -> entry

Distinct from entry.py's trend-following rules: those require an
ALREADY-established trend (trend_bias + trend_strength >= 0.6). This
strategy instead looks for the trend just starting — a Donchian breakout
out of a low-ATR compression window, which by construction fires before
trend_bias would have accumulated enough confirmations. Running both
side by side is intentional: this one catches the start of a move,
entry.py's trend-following rules catch continuation.

Distinct from entry.py's mean-reversion rules: those explicitly require
a RANGING regime and bet on reversion. This strategy is regime-agnostic
by design — compression can happen inside a ranging regime right before
it stops ranging, which is exactly the signal we want, so gating on
market_state the way entry.py does would filter out the highest-value
setups.

Required columns (all already produced upstream):
    close, volume
    donchian_upper, donchian_lower  (indicators/volatility/donchian.py)
    ATR14                            (indicators/volatility/atr.py)
    ADX14                            (indicators/trend/adx.py)
    EMA20, EMA50                     (indicators/trend/ema.py)
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "close", "volume", "donchian_upper", "donchian_lower",
    "ATR14", "ADX14", "EMA20", "EMA50",
]


def _prepare(df: pd.DataFrame, atr_compression_window: int = 50,
             atr_compression_pct: float = 0.8, volume_window: int = 20,
             volume_expansion_mult: float = 1.5, adx_threshold: float = 20.0) -> pd.DataFrame:
    """Vectorized precompute of the compression/expansion/confirmation flags."""
    out = pd.DataFrame(index=df.index)

    # Compression is measured on the PRIOR bar's ATR relative to its own
    # recent history — using the current bar's ATR would already include
    # the volatility expansion the breakout itself causes, which biases
    # the "was compressed" check toward always being true right after a move.
    atr_ma = df["ATR14"].rolling(atr_compression_window).mean()
    out["was_compressed"] = (df["ATR14"].shift(1) < atr_ma.shift(1) * atr_compression_pct)

    # Breakout: close beyond the PRIOR bar's Donchian channel (the channel
    # itself is already backward-looking, but shifting one more bar avoids
    # the channel having been updated by the current bar's own high/low).
    out["breaks_upper"] = df["close"] > df["donchian_upper"].shift(1)
    out["breaks_lower"] = df["close"] < df["donchian_lower"].shift(1)

    vol_ma = df["volume"].rolling(volume_window).mean()
    out["volume_expansion"] = df["volume"] > vol_ma * volume_expansion_mult

    out["adx_confirms"] = df["ADX14"] > adx_threshold
    out["ema_bullish"] = df["EMA20"] > df["EMA50"]
    out["ema_bearish"] = df["EMA20"] < df["EMA50"]

    return out


def generate_breakout_signal(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Add breakout entry-signal columns to the dataframe.

    Returns:
        df with new columns:
          'breakout_signal' - 'BUY', 'SELL', or 'HOLD'
          'breakout_reason' - list[str] of the specific conditions that fired
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for breakout strategy: {missing}")

    flags = _prepare(df, **kwargs)

    signals = []
    reasons_list = []

    for i in df.index:
        f = flags.loc[i]
        reasons = []

        if bool(f["was_compressed"]) and bool(f["breaks_upper"]):
            reasons.append("Price broke above Donchian upper band after ATR compression")
            if bool(f["volume_expansion"]):
                reasons.append("Volume expansion confirms breakout")
            if bool(f["adx_confirms"]):
                reasons.append("ADX confirms directional strength")
            if bool(f["ema_bullish"]):
                reasons.append("EMA20 > EMA50 agrees with breakout direction")
            if len(reasons) >= 3:
                signals.append("BUY")
                reasons_list.append(reasons)
                continue

        reasons = []
        if bool(f["was_compressed"]) and bool(f["breaks_lower"]):
            reasons.append("Price broke below Donchian lower band after ATR compression")
            if bool(f["volume_expansion"]):
                reasons.append("Volume expansion confirms breakout")
            if bool(f["adx_confirms"]):
                reasons.append("ADX confirms directional strength")
            if bool(f["ema_bearish"]):
                reasons.append("EMA20 < EMA50 agrees with breakout direction")
            if len(reasons) >= 3:
                signals.append("SELL")
                reasons_list.append(reasons)
                continue

        signals.append("HOLD")
        reasons_list.append([])

    df = df.copy()
    df["breakout_signal"] = signals
    df["breakout_reason"] = reasons_list
    return df


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators

    n = 400
    rng = np.random.default_rng(3)
    # simulate a compression phase followed by a directional break
    close = np.concatenate([
        100 + np.cumsum(rng.normal(0, 0.15, 150)),
        100 + np.cumsum(rng.normal(0.25, 0.6, 250)),
    ])
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.random(n) * 1.2,
        "low": close - rng.random(n) * 1.2,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = generate_breakout_signal(df)
    print(df["breakout_signal"].value_counts())
    non_hold = df[df["breakout_signal"] != "HOLD"]
    if len(non_hold):
        print(non_hold[["close", "breakout_signal", "breakout_reason"]].head(5).to_string())
