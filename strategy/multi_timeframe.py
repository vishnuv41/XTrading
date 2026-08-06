"""
Multi-Timeframe Confirmation

A signal generated on a lower timeframe (e.g. 15m) is far more reliable
when the higher timeframes (1h, 4h) agree with its direction. This
module implements that check: it does NOT generate new signals, it only
confirms or vetoes signals already produced by entry.py/signal.py on the
lower timeframe, using trend_bias already computed on the higher
timeframe dataframes.

Why this matters: a 15m EMA crossover can fire constantly in a market
that is, on the 4h chart, in a strong opposing trend — this is the
single most common way naive lower-timeframe strategies get chopped up.
Requiring higher-timeframe alignment cuts the number of trades taken but
meaningfully raises the win rate of the ones that remain, since you are
now trading *with* the dominant trend rather than against short-term
noise.

Timeframe alignment uses pandas.merge_asof (backward direction): for
each 15m candle, we look up the most recent 1h/4h candle that has
already CLOSED at or before that timestamp. This avoids lookahead bias
— a common and serious bug where a backtest accidentally uses a higher
timeframe candle that technically hadn't closed yet at the time of the
lower-timeframe decision, making backtest results look far better than
what live trading would actually achieve.
"""

import pandas as pd


def _confirm_against_higher_tf(lower_df: pd.DataFrame, higher_df: pd.DataFrame,
                                label: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Merge a single higher-timeframe trend_bias onto the lower-timeframe
    dataframe via an as-of backward join, and add a boolean column
    showing whether entry_signal direction agrees with that higher
    timeframe's trend_bias.
    """
    for col in (timestamp_col, "trend_bias"):
        if col not in higher_df.columns:
            raise ValueError(f"higher_df ({label}) missing required column '{col}'")
    if timestamp_col not in lower_df.columns:
        raise ValueError(f"lower_df missing required column '{timestamp_col}'")
    if "entry_signal" not in lower_df.columns:
        raise ValueError("lower_df missing 'entry_signal' — run generate_entry_signal() first")

    lower_sorted = lower_df.sort_values(timestamp_col)
    higher_sorted = higher_df[[timestamp_col, "trend_bias"]].sort_values(timestamp_col)

    merged = pd.merge_asof(
        lower_sorted, higher_sorted,
        on=timestamp_col, direction="backward",
        suffixes=("", f"_{label}"),
    )

    bias_col = f"trend_bias_{label}" if "trend_bias" in lower_df.columns else "trend_bias"
    # merge_asof only suffixes when the column name collides in both
    # frames; lower_df typically doesn't have 'trend_bias' itself in this
    # module's expected usage (it has entry_signal instead), so guard for
    # both cases explicitly.
    if bias_col not in merged.columns:
        bias_col = "trend_bias"
    merged = merged.rename(columns={bias_col: f"htf_{label}_trend_bias"})

    agree_col = f"htf_{label}_agrees"
    htf_bias = merged[f"htf_{label}_trend_bias"]
    merged[agree_col] = (
        ((merged["entry_signal"] == "BUY") & (htf_bias == "bullish")) |
        ((merged["entry_signal"] == "SELL") & (htf_bias == "bearish")) |
        (merged["entry_signal"] == "HOLD")  # HOLD trivially "agrees" (nothing to confirm)
    )
    return merged


def confirm_multi_timeframe(df_lower: pd.DataFrame, higher_timeframes: dict[str, pd.DataFrame],
                             require_all: bool = True,
                             timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Confirm a lower-timeframe entry signal against one or more higher
    timeframes' trend bias.

    Args:
        df_lower: The lower-timeframe dataframe (e.g. 15m) with
            'entry_signal' already computed (entry.py) and a timestamp
            column.
        higher_timeframes: Dict mapping a label to a dataframe, e.g.
            {"1h": df_1h, "4h": df_4h}. Each must have 'trend_bias'
            already computed (trend.py) and a timestamp column.
        require_all: If True (default), the final 'mtf_confirmed' column
            is True only when ALL higher timeframes agree. If False, it's
            True when ANY higher timeframe agrees (a looser filter).
        timestamp_col: Name of the timestamp column in all dataframes
            (default 'timestamp'). Must be sortable/comparable and
            consistent in type across all frames (e.g. all pandas
            datetime64, not a mix of datetime and string).

    Returns:
        df_lower with new columns:
          'htf_<label>_trend_bias' and 'htf_<label>_agrees' for each
              higher timeframe passed in
          'mtf_confirmed' - final combined bool per require_all logic
          'final_signal'  - 'entry_signal' where mtf_confirmed is True,
                             else 'HOLD' (i.e. the signal to actually act
                             on downstream)
    """
    if not higher_timeframes:
        raise ValueError("higher_timeframes dict must contain at least one timeframe")

    merged = df_lower
    agree_cols = []
    for label, higher_df in higher_timeframes.items():
        merged = _confirm_against_higher_tf(merged, higher_df, label, timestamp_col=timestamp_col)
        agree_cols.append(f"htf_{label}_agrees")

    if require_all:
        merged["mtf_confirmed"] = merged[agree_cols].all(axis=1)
    else:
        merged["mtf_confirmed"] = merged[agree_cols].any(axis=1)

    merged["final_signal"] = merged["entry_signal"].where(merged["mtf_confirmed"], "HOLD")
    return merged


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy.trend import calculate_trend_bias
    from strategy.entry import generate_entry_signal

    def make_df(n, freq, seed):
        rng = np.random.default_rng(seed)
        close = np.cumsum(rng.normal(0, 1, n)) + 100
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq),
            "open": close + rng.normal(0, 0.3, n),
            "high": close + rng.random(n) * 1.5,
            "low": close - rng.random(n) * 1.5,
            "close": close,
            "volume": rng.integers(100, 5000, n),
        })
        return df

    # Simulate 15m, 1h, 4h data over the same ~10 day window
    df_15m = make_df(960, "15min", seed=1)   # 15m * 960 ~= 10 days
    df_1h = make_df(240, "1h", seed=1)       # 1h * 240 = 10 days
    df_4h = make_df(60, "4h", seed=1)        # 4h * 60 = 10 days

    df_15m = calculate_all_indicators(df_15m)
    df_15m = calculate_market_state(df_15m)
    df_15m = generate_entry_signal(df_15m)

    df_1h = calculate_all_indicators(df_1h)
    df_1h = calculate_trend_bias(df_1h)

    df_4h = calculate_all_indicators(df_4h)
    df_4h = calculate_trend_bias(df_4h)

    result = confirm_multi_timeframe(
        df_15m, {"1h": df_1h, "4h": df_4h}, require_all=True
    )

    print(result[["timestamp", "close", "entry_signal",
                   "htf_1h_trend_bias", "htf_4h_trend_bias",
                   "mtf_confirmed", "final_signal"]].tail(15))
    print()
    print("Raw entry signals:", result["entry_signal"].value_counts().to_dict())
    print("Final (MTF-confirmed) signals:", result["final_signal"].value_counts().to_dict())