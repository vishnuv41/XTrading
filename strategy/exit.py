"""
Exit Signal

Generates exit signals for currently-open positions. This is
deliberately separate from entry.py: entry and exit are not mirror
images of each other in a good trading system — you typically want to
exit a winning trend trade faster than you'd want to fade it, and you
want to exit on early warning signs (momentum exhaustion, regime
deterioration) rather than waiting for a full opposite entry signal to
fire.

This module computes exit conditions for BOTH a hypothetical open long
and open short position on every row (as 'exit_long'/'exit_short`
booleans with reasons), rather than requiring the caller to pass in
their current position state. The caller (signal.py, or a live
position manager) then just reads whichever column matches the position
they're actually in. This keeps the module stateless and easy to
backtest against historical data without needing to simulate a position
tracker just to test the exit logic itself.

Exit triggers checked, independent of entry rule set used:
    1. Trend flip       - Supertrend direction reverses against the position
    2. EMA cross        - EMA20 crosses to the wrong side of EMA50
    3. Momentum reversal - MACD trend flips against the position
    4. Momentum exhaustion - RSI crosses back through 50 from an extreme
                             (was >70 now <60 for longs; was <30 now >40
                             for shorts) — designed to catch fading
                             momentum before a full reversal confirms
    5. Regime deterioration - market_state shifts into 'ranging_high_vol'
                               or any 'mixed_*' state, which the regime
                               module flags as the least reliable/most
                               dangerous conditions to hold a directional
                               position through

Note: this module does NOT calculate stop-loss / take-profit hits — that
is price-level risk management and belongs in risk_engine/stoploss.py
and risk_engine/takeprofit.py. This module only covers *signal-based*
discretionary-style exits (i.e. "the setup that justified this trade no
longer holds"), which should be checked in addition to, not instead of,
hard SL/TP levels.
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "supertrend_direction", "EMA20", "EMA50", "MACD_trend", "RSI14", "market_state",
]

DETERIORATED_STATES = {"ranging_high_vol", "mixed_low_vol", "mixed_medium_vol", "mixed_high_vol"}


def _exit_long_conditions(row, prev_row) -> list[str]:
    reasons = []

    if row["supertrend_direction"] == -1 and prev_row["supertrend_direction"] == 1:
        reasons.append("Supertrend flipped to downtrend")

    if row["EMA20"] < row["EMA50"] and prev_row["EMA20"] >= prev_row["EMA50"]:
        reasons.append("EMA20 crossed below EMA50")

    if row["MACD_trend"] == "bearish" and prev_row["MACD_trend"] == "bullish":
        reasons.append("MACD flipped bearish")

    if prev_row["RSI14"] > 70 and row["RSI14"] < 60:
        reasons.append("RSI momentum exhaustion (fell from overbought)")

    if row["market_state"] in DETERIORATED_STATES:
        reasons.append(f"Regime deteriorated to '{row['market_state']}'")

    return reasons


def _exit_short_conditions(row, prev_row) -> list[str]:
    reasons = []

    if row["supertrend_direction"] == 1 and prev_row["supertrend_direction"] == -1:
        reasons.append("Supertrend flipped to uptrend")

    if row["EMA20"] > row["EMA50"] and prev_row["EMA20"] <= prev_row["EMA50"]:
        reasons.append("EMA20 crossed above EMA50")

    if row["MACD_trend"] == "bullish" and prev_row["MACD_trend"] == "bearish":
        reasons.append("MACD flipped bullish")

    if prev_row["RSI14"] < 30 and row["RSI14"] > 40:
        reasons.append("RSI momentum exhaustion (rose from oversold)")

    if row["market_state"] in DETERIORATED_STATES:
        reasons.append(f"Regime deteriorated to '{row['market_state']}'")

    return reasons


def calculate_exit_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exit signal columns for both hypothetical long and short positions.

    Args:
        df: DataFrame with the indicator/regime columns in
            REQUIRED_COLUMNS already computed (calculate_all_indicators
            + calculate_market_state).

    Returns:
        df with new columns:
          'exit_long'         - bool, True if an open long should exit here
          'exit_long_reason'  - list[str] of triggered conditions
          'exit_short'        - bool, True if an open short should exit here
          'exit_short_reason' - list[str] of triggered conditions

        The first row is always False/[] for both (no previous row to
        compare against for cross/flip detection).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")

    exit_long = [False]
    exit_long_reason = [[]]
    exit_short = [False]
    exit_short_reason = [[]]

    rows = df.to_dict("records")
    for i in range(1, len(rows)):
        row, prev_row = rows[i], rows[i - 1]

        long_reasons = _exit_long_conditions(row, prev_row)
        short_reasons = _exit_short_conditions(row, prev_row)

        exit_long.append(len(long_reasons) > 0)
        exit_long_reason.append(long_reasons)
        exit_short.append(len(short_reasons) > 0)
        exit_short_reason.append(short_reasons)

    df["exit_long"] = exit_long
    df["exit_long_reason"] = exit_long_reason
    df["exit_short"] = exit_short
    df["exit_short_reason"] = exit_short_reason
    return df


def get_exit_decision(df: pd.DataFrame, index: int, position_side: str) -> tuple[bool, list]:
    """
    Convenience accessor for a live position manager: given the current
    row index and which side is open ('long' or 'short'), return whether
    to exit and why.

    Args:
        df: DataFrame with calculate_exit_signals() already run.
        index: Row index (position in the dataframe, e.g. -1 for latest).
        position_side: 'long' or 'short'.

    Returns:
        (should_exit: bool, reasons: list[str])
    """
    if position_side not in ("long", "short"):
        raise ValueError("position_side must be 'long' or 'short'")

    row = df.iloc[index]
    if position_side == "long":
        return bool(row["exit_long"]), row["exit_long_reason"]
    return bool(row["exit_short"]), row["exit_short_reason"]


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state

    n = 400
    rng = np.random.default_rng(13)
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
    df = calculate_exit_signals(df)

    print(df[["close", "market_state", "exit_long", "exit_short"]].tail(15))
    print()
    print("Exit-long triggers:", df["exit_long"].sum(), "/", len(df))
    print("Exit-short triggers:", df["exit_short"].sum(), "/", len(df))

    should_exit, reasons = get_exit_decision(df, -1, "long")
    print(f"\nLatest bar, if holding long -> exit={should_exit}, reasons={reasons}")