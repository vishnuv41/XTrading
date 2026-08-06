"""
Take Profit

Calculates where to take profit, driven off the stop-loss distance
rather than set independently — this keeps risk:reward the primary
design variable of the trade (see risk_reward.py), instead of picking
an SL and TP separately and hoping the ratio works out.

Two methods:

1. Risk-multiple TP (primary): take_profit = entry +/- (R * stop_distance)
   where R is the desired reward multiple of risk (e.g. R=2 means "make
   2x what I'm risking"). This is the standard approach and is what
   risk_reward.py's minimum-R:R filter assumes was used to generate the
   take-profit level.

2. ATR-projection TP (secondary/sanity-check): entry +/- (multiplier *
   ATR), independent of the stop distance. Useful as a cross-check —
   if the risk-multiple TP lands far beyond a realistic ATR-based price
   projection for the lookahead window, the target may be unrealistic
   for the timeframe being traded.
"""

import pandas as pd


def calculate_take_profit(entry_price: float, stop_loss: float, side: str,
                           risk_reward_ratio: float = 2.0) -> float:
    """
    Calculate take-profit price as a multiple of the risk (entry-to-stop
    distance).

    Args:
        entry_price: The trade's entry price.
        stop_loss: The trade's stop-loss price (from stoploss.py).
        side: 'long' or 'short'.
        risk_reward_ratio: Desired reward multiple of risk (default 2.0,
            i.e. targeting 2x the distance being risked).

    Returns:
        Take-profit price.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio must be positive")

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance == 0:
        raise ValueError("entry_price and stop_loss cannot be equal (zero risk distance)")

    if side == "long":
        return entry_price + risk_reward_ratio * risk_distance
    return entry_price - risk_reward_ratio * risk_distance


def calculate_take_profit_atr(entry_price: float, atr: float, side: str,
                               multiplier: float = 3.0) -> float:
    """
    Secondary ATR-projection take-profit, independent of the stop
    distance. Use as a sanity check against the risk-multiple TP above,
    not as the primary target — mixing the two as if they were the same
    thing defeats the point of designing the trade around risk:reward.

    Args:
        entry_price: The trade's entry price.
        atr: Current ATR value at entry.
        side: 'long' or 'short'.
        multiplier: ATR multiplier (default 3.0, i.e. one ATR further out
            than the default 2x-ATR stop-loss, a rough 1.5:1 R:R
            equivalent when stop multiplier=2).

    Returns:
        Take-profit price.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if atr <= 0:
        raise ValueError("ATR must be positive")

    if side == "long":
        return entry_price + multiplier * atr
    return entry_price - multiplier * atr


def calculate_take_profit_series(df: pd.DataFrame, entry_col: str = "close",
                                  stop_col: str = "stop_loss",
                                  side_col: str = "entry_signal",
                                  risk_reward_ratio: float = 2.0) -> pd.DataFrame:
    """
    Vectorized take-profit for every row that has a stop-loss already
    computed (see stoploss.calculate_stop_loss_series).

    Returns:
        df with a new column 'take_profit' (NaN where side_col is 'HOLD').
    """
    for col in (entry_col, stop_col, side_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    def compute(row):
        if row[side_col] == "BUY":
            return calculate_take_profit(row[entry_col], row[stop_col], "long", risk_reward_ratio)
        if row[side_col] == "SELL":
            return calculate_take_profit(row[entry_col], row[stop_col], "short", risk_reward_ratio)
        return float("nan")

    df["take_profit"] = df.apply(compute, axis=1)
    return df


if __name__ == "__main__":
    print("Long: entry=100, stop=96, R:R=2 ->", calculate_take_profit(100, 96, "long", 2.0))
    print("Short: entry=100, stop=104, R:R=2 ->", calculate_take_profit(100, 104, "short", 2.0))
    print("ATR-projection long: entry=100, ATR=2 ->", calculate_take_profit_atr(100, 2, "long"))

    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy import generate_entry_signal
    from risk_engine.stoploss import calculate_stop_loss_series

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
    df = calculate_take_profit_series(df)

    non_hold = df[df["entry_signal"] != "HOLD"]
    print(f"\n{len(non_hold)} trades, sample entry/stop/target:")
    print(non_hold[["close", "entry_signal", "stop_loss", "take_profit"]].head(5))