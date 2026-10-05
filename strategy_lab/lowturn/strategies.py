"""
strategy_lab/lowturn/strategies.py
----------------------------------
Frozen Strategy Grid implementations for Phase 17 (T4).

Grid:
1. Buy & Hold (Benchmark)
2. Trend Following Single EMA: EMA 20, EMA 50, EMA 100, EMA 200
3. Trend Following Crossover: EMA 20/100, EMA 50/200
4. Time-Series Momentum (TSMOM): 30d, 60d, 90d, 180d, 365d
5. Volatility-Targeted Momentum: TSMOM 90d & 180d with 20% & 40% vol target
"""

from typing import Dict, Callable
import numpy as np
import pandas as pd


# 1. Buy & Hold
def generate_buy_and_hold_weights(df: pd.DataFrame) -> pd.Series:
    """Always fully invested in the asset."""
    return pd.Series(1.0, index=df.index)


# 2. Trend Following (Single EMA)
def generate_single_ema_weights(df: pd.DataFrame, span: int) -> pd.Series:
    """Long if Close > EMA(span), else Cash (0.0)."""
    ema = df["close"].ewm(span=span, adjust=False).mean()
    weights = (df["close"] > ema).astype(float)
    return weights


# 3. Trend Following (EMA Crossover)
def generate_ema_crossover_weights(df: pd.DataFrame, fast_span: int, slow_span: int) -> pd.Series:
    """Long if Fast EMA > Slow EMA, else Cash (0.0)."""
    fast_ema = df["close"].ewm(span=fast_span, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_span, adjust=False).mean()
    weights = (fast_ema > slow_ema).astype(float)
    return weights


# 4. Time-Series Momentum (TSMOM)
def generate_tsmom_weights(df: pd.DataFrame, lookback_bars: int) -> pd.Series:
    """Long if Return over lookback > 0, else Cash (0.0)."""
    momentum = df["close"].pct_change(lookback_bars)
    weights = (momentum > 0.0).astype(float)
    return weights


# 5. Volatility-Targeted Momentum
def generate_vol_targeted_tsmom_weights(
    df: pd.DataFrame,
    lookback_bars: int,
    target_annual_vol: float = 0.20,
    vol_lookback_bars: int = 30,
    periods_per_year: int = 365,
    max_leverage: float = 1.5,
) -> pd.Series:
    """
    Long momentum signal scaled by inverse realized volatility to target constant volatility.
    """
    # 1. Binary momentum signal
    mom = df["close"].pct_change(lookback_bars)
    signal = (mom > 0.0).astype(float)

    # 2. Realized daily return volatility
    daily_returns = df["close"].pct_change().fillna(0.0)
    realized_daily_vol = daily_returns.rolling(window=vol_lookback_bars, min_periods=10).std()
    annualized_vol = realized_daily_vol * np.sqrt(periods_per_year)

    # Sizing factor: target_vol / annualized_vol
    sizing = (target_annual_vol / annualized_vol.replace(0.0, np.nan)).fillna(1.0)
    sizing = sizing.clip(lower=0.0, upper=max_leverage)

    weights = (signal * sizing).fillna(0.0)
    return weights


# Registry of all strategies in frozen grid
def build_frozen_strategy_grid(timeframe: str = "1d") -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """
    Returns dictionary of {strategy_id: strategy_weight_generator}.
    Timeframe scaling: If 4H, lookbacks are multiplied by 6.
    """
    multiplier = 6 if timeframe == "4h" else 1
    grid = {}

    # 1. Buy & Hold
    grid["Buy_and_Hold"] = generate_buy_and_hold_weights

    # 2. Single EMA
    for span in [20, 50, 100, 200]:
        adj_span = span * multiplier
        grid[f"Trend_EMA_{span}"] = lambda df, s=adj_span: generate_single_ema_weights(df, span=s)

    # 3. EMA Crossover
    for fast, slow in [(20, 100), (50, 200)]:
        adj_fast, adj_slow = fast * multiplier, slow * multiplier
        grid[f"Crossover_EMA_{fast}_{slow}"] = (
            lambda df, f=adj_fast, sl=adj_slow: generate_ema_crossover_weights(df, fast_span=f, slow_span=sl)
        )

    # 4. TSMOM
    for lb in [30, 60, 90, 180, 365]:
        adj_lb = lb * multiplier
        grid[f"TSMOM_{lb}d"] = lambda df, l=adj_lb: generate_tsmom_weights(df, lookback_bars=l)

    # 5. Vol-Targeted Momentum
    periods_yr = 365 * multiplier
    for mom_lb in [90, 180]:
        for vol_tgt in [0.20, 0.40]:
            adj_mom = mom_lb * multiplier
            adj_vol_lb = 30 * multiplier
            tgt_pct = int(vol_tgt * 100)
            grid[f"VolTarget_{tgt_pct}pct_TSMOM_{mom_lb}d"] = (
                lambda df, m=adj_mom, v=vol_tgt, vlb=adj_vol_lb, py=periods_yr: generate_vol_targeted_tsmom_weights(
                    df, lookback_bars=m, target_annual_vol=v, vol_lookback_bars=vlb, periods_per_year=py
                )
            )

    return grid
