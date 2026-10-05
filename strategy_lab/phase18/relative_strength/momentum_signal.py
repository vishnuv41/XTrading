"""
strategy_lab/phase18/relative_strength/signal.py
--------------------------------------------------
Hypothesis H18-B: Cross-Asset Relative Strength Momentum Strategy.
Ranks universe of 9 assets by momentum over lookback L, selecting top K assets.
Combines with trend filter (close > EMA50) to allow cash preservation during broad market drawdowns.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_relative_strength_weights(
    prices_df: pd.DataFrame,
    lookback_days: int = 14,
    top_k: int = 3,
    rebalance_interval_days: int = 1,
    ema_trend_filter_period: int = 50,
) -> pd.DataFrame:
    """
    Computes portfolio target weights for H18-B Relative Strength strategy.

    Args:
        prices_df: Close prices (index=DatetimeIndex, columns=symbols).
        lookback_days: Momentum lookback L in days.
        top_k: Number of highest momentum assets to allocate to (K).
        rebalance_interval_days: Rebalance frequency in days (1 = daily, 7 = weekly).
        ema_trend_filter_period: EMA lookback for trend gate.

    Returns:
        DataFrame of target weights (index=DatetimeIndex, columns=symbols).
    """
    # 1. Momentum: simple return over lookback_days
    mom_df = prices_df.pct_change(lookback_days)

    # 2. Trend filter: close > EMA(period)
    ema_df = prices_df.ewm(span=ema_trend_filter_period, adjust=False).mean()
    trend_filter = prices_df > ema_df

    weights_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    weight_per_asset = 1.0 / float(top_k)

    dates = prices_df.index
    n_dates = len(dates)

    last_rebal_idx = -rebalance_interval_days

    for i in range(n_dates):
        dt = dates[i]
        
        # Check rebalance interval
        if (i - last_rebal_idx) >= rebalance_interval_days:
            last_rebal_idx = i
            
            # Rank momentum across assets on this bar
            row_mom = mom_df.loc[dt]
            row_trend = trend_filter.loc[dt]

            # Drop NaNs
            valid_mom = row_mom.dropna()
            if len(valid_mom) >= top_k:
                top_assets = valid_mom.nlargest(top_k).index
                
                for sym in prices_df.columns:
                    if sym in top_assets:
                        # Allocate only if asset passes trend filter, else hold cash
                        weights_df.loc[dt, sym] = weight_per_asset if row_trend.get(sym, False) else 0.0
                    else:
                        weights_df.loc[dt, sym] = 0.0
            else:
                # Insufficient history
                weights_df.loc[dt, :] = 0.0
        else:
            # Hold previous weights until next rebalance date
            weights_df.loc[dt, :] = weights_df.iloc[i - 1]

    return weights_df
