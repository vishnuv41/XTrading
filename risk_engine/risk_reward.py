"""
Risk:Reward

Computes the risk:reward ratio for a trade (distance to target divided
by distance to stop) and provides a filter to reject trades that don't
meet a minimum acceptable ratio.

This is one of the simplest but most impactful checks in the whole
pipeline: even a strategy with a mediocre win rate can be profitable if
every trade has a good R:R, and even a strategy with a great win rate
can bleed money if R:R is consistently poor. Filtering out sub-par R:R
setups before they ever reach position_size.py means capital isn't risked
on trades that don't clear the bar, regardless of what the entry signal
said.
"""

import pandas as pd


def calculate_risk_reward(entry_price: float, stop_loss: float, take_profit: float) -> float:
    """
    Calculate the risk:reward ratio of a trade.

    Args:
        entry_price: Entry price.
        stop_loss: Stop-loss price.
        take_profit: Take-profit price.

    Returns:
        Risk:reward ratio (e.g. 2.0 means the reward is 2x the risk).
        Works for both long and short since it's purely distance-based.
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk == 0:
        raise ValueError("Risk distance is zero (entry_price == stop_loss)")

    return reward / risk


def meets_minimum_risk_reward(entry_price: float, stop_loss: float, take_profit: float,
                               min_ratio: float = 1.5) -> bool:
    """
    Check whether a trade clears a minimum risk:reward bar.

    Args:
        entry_price, stop_loss, take_profit: Trade prices.
        min_ratio: Minimum acceptable R:R (default 1.5 — a common floor;
            many systematic traders use 1.5-2.0 as a hard minimum since
            below that, a strategy needs an unusually high win rate to
            stay profitable after fees/slippage).

    Returns:
        True if the trade's R:R >= min_ratio.
    """
    return calculate_risk_reward(entry_price, stop_loss, take_profit) >= min_ratio


def filter_by_risk_reward(df: pd.DataFrame, entry_col: str = "close",
                           stop_col: str = "stop_loss", target_col: str = "take_profit",
                           side_col: str = "entry_signal", min_ratio: float = 1.5) -> pd.DataFrame:
    """
    Add a risk:reward ratio column and downgrade any entry signal that
    fails the minimum ratio to 'HOLD'. This is meant to run AFTER
    stoploss/takeprofit have been computed and BEFORE position_size.py,
    so undersized-reward trades never reach the sizing/execution stage.

    Args:
        df: DataFrame with entry, stop, target, and side columns.
        min_ratio: Minimum acceptable R:R (default 1.5).

    Returns:
        df with new columns:
          'risk_reward_ratio' - computed ratio (NaN where side_col is 'HOLD')
          'final_signal'      - side_col's value, downgraded to 'HOLD' if
                                 risk_reward_ratio < min_ratio. If a
                                 'final_signal' column already exists
                                 (e.g. from strategy/multi_timeframe.py),
                                 this filter is applied on top of it
                                 rather than overwriting the upstream
                                 filtering decision.
    """
    for col in (entry_col, stop_col, target_col, side_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    source_col = "final_signal" if "final_signal" in df.columns else side_col

    def compute_ratio(row):
        if row[side_col] == "HOLD":
            return float("nan")
        return calculate_risk_reward(row[entry_col], row[stop_col], row[target_col])

    df["risk_reward_ratio"] = df.apply(compute_ratio, axis=1)

    def apply_filter(row):
        signal = row[source_col]
        if signal == "HOLD":
            return "HOLD"
        ratio = row["risk_reward_ratio"]
        if pd.isna(ratio) or ratio < min_ratio:
            return "HOLD"
        return signal

    df["final_signal"] = df.apply(apply_filter, axis=1)
    return df


if __name__ == "__main__":
    print("R:R for entry=100, stop=96, target=108:",
          calculate_risk_reward(100, 96, 108))
    print("Meets 1.5 minimum?", meets_minimum_risk_reward(100, 96, 108, 1.5))
    print("Meets 3.0 minimum?", meets_minimum_risk_reward(100, 96, 108, 3.0))

    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy import generate_entry_signal
    from risk_engine.stoploss import calculate_stop_loss_series
    from risk_engine.takeprofit import calculate_take_profit_series

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
    df = filter_by_risk_reward(df, min_ratio=1.5)

    print(f"\nRaw entry signals: {(df['entry_signal'] != 'HOLD').sum()}")
    print(f"After R:R >= 1.5 filter: {(df['final_signal'] != 'HOLD').sum()}")
    non_hold = df[df["entry_signal"] != "HOLD"]
    print(non_hold[["entry_signal", "risk_reward_ratio", "final_signal"]].head(8))