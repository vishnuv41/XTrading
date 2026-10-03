"""
Stop Loss

Calculates where to place a stop-loss, using ATR (Average True Range)
as the volatility yardstick rather than a fixed percentage or fixed pip
distance. This matters because a fixed 2% stop is tight in a calm market
and dangerously tight in a volatile one — it gets you stopped out by
ordinary noise far more often than a volatility-scaled stop does.

Two placement methods are provided:

1. ATR-based fixed stop: entry -/+ (multiplier * ATR). Simple, static
   once set, good default.

2. Supertrend-based trailing stop: uses the Supertrend line itself
   (already ATR-based internally — see indicators/trend/supertrend.py)
   as a dynamic stop that moves with price, locking in gains as a trend
   develops rather than sitting at a fixed level for the whole trade.

Both are exposed so callers can choose per strategy: ATR-fixed suits
mean-reversion entries (a static target range), Supertrend-trailing
suits trend-following entries (let winners run, exit on trend flip).
"""

import pandas as pd


def calculate_stop_loss(entry_price: float, atr: float, side: str,
                         multiplier: float = 2.0) -> float:
    """
    Calculate a single ATR-based stop-loss price.

    Args:
        entry_price: The trade's entry price.
        atr: Current ATR value at entry (e.g. from df['ATR14']).
        side: 'long' or 'short'.
        multiplier: How many ATRs away to place the stop (default 2.0 —
            common range is 1.5-3.0; tighter risks more noise-outs,
            wider risks a worse risk:reward ratio).

    Returns:
        Stop-loss price.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if atr <= 0:
        raise ValueError("ATR must be positive")

    if side == "long":
        return entry_price - multiplier * atr
    return entry_price + multiplier * atr


def calculate_stop_loss_series(df: pd.DataFrame, entry_col: str = "close",
                                atr_col: str = "ATR14", side_col: str = "entry_signal",
                                multiplier: float = 2.0) -> pd.DataFrame:
    """
    Vectorized ATR stop-loss for every row of a dataframe that has an
    entry signal (from strategy/entry.py).

    Args:
        df: DataFrame with entry price, ATR, and signal columns.
        entry_col: Column to treat as entry price (default 'close', i.e.
            assume entry at the candle's close — adjust if your execution
            model fills at next-open instead).
        atr_col: ATR column name (default 'ATR14').
        side_col: Column holding 'BUY'/'SELL'/'HOLD' (default
            'entry_signal' from entry.py).
        multiplier: ATR multiplier (default 2.0).

    Returns:
        df with a new column 'stop_loss' (NaN where side_col is 'HOLD').
    """
    for col in (entry_col, atr_col, side_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    def compute(row):
        if row[side_col] == "BUY":
            return calculate_stop_loss(row[entry_col], row[atr_col], "long", multiplier)
        if row[side_col] == "SELL":
            return calculate_stop_loss(row[entry_col], row[atr_col], "short", multiplier)
        return float("nan")

    df["stop_loss"] = df.apply(compute, axis=1)
    return df


def calculate_supertrend_stop(row, side: str) -> float:
    """
    Use the Supertrend line value itself as a trailing stop for an open
    position. Intended to be called each new candle for an open trade
    (the stop naturally moves as Supertrend updates) rather than once at
    entry.

    Args:
        row: A dataframe row (or dict) with a 'supertrend' column already
            computed (indicators/trend/supertrend.py).
        side: 'long' or 'short'. Only meaningful when the Supertrend
            direction agrees with the position side — see the guard
            below, which raises rather than silently returning a stale
            value if the trend has already flipped (at that point exit.py
            should have already fired, not this function).

    Returns:
        Current Supertrend value to use as the trailing stop.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    expected_dir = 1 if side == "long" else -1
    if row["supertrend_direction"] != expected_dir:
        raise ValueError(
            f"Supertrend direction ({row['supertrend_direction']}) disagrees with "
            f"position side '{side}' — check exit.py's exit_{side} signal "
            f"before continuing to hold this position."
        )
    return row["supertrend"]


if __name__ == "__main__":
    print("Single-trade examples:")
    print("Long stop @ entry=100, ATR=2:", calculate_stop_loss(100, 2, "long"))
    print("Short stop @ entry=100, ATR=2:", calculate_stop_loss(100, 2, "short"))

    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy import generate_entry_signal

    n = 300
    rng = np.random.default_rng(42)
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
    df = calculate_stop_loss_series(df)

    non_hold = df[df["entry_signal"] != "HOLD"]
    print(f"\n{len(non_hold)} trades generated, sample stop-loss placements:")
    print(non_hold[["close", "ATR14", "entry_signal", "stop_loss"]].head(5))