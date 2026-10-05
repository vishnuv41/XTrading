"""
strategy_lab/phase19/regime_engine.py
--------------------------------------
Causal Market Regime Calculation Engine for Phase 19.
Computes:
1. Cross-Asset Market Breadth (fraction of 9 assets above EMA50).
2. Asset-level Trend Quality (EMA20 vs EMA50, Slope).
3. Volatility State (rolling 60-day percentile of 14-day ATR).
Strictly Causal: Uses only data available at bar close t.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_market_breadth(prices_df: pd.DataFrame, ema_period: int = 50) -> pd.Series:
    """
    Computes fraction of universe assets with Close > EMA(period) at bar t.
    Returns Series with values in [0.0, 1.0].
    """
    ema_df = prices_df.ewm(span=ema_period, adjust=False).mean()
    trending_df = prices_df > ema_df
    breadth = trending_df.mean(axis=1)
    return breadth


def compute_asset_atr_percentile(prices_df: pd.DataFrame, atr_period: int = 14, lookback_days: int = 60) -> pd.DataFrame:
    """
    Computes rolling percentile rank of 14-day ATR over lookback_days for each asset.
    Percentile values in [0.0, 100.0].
    """
    pct_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    for sym in prices_df.columns:
        p = prices_df[sym]
        # Daily return magnitude as ATR approximation
        tr = p.pct_change().abs()
        atr14 = tr.rolling(window=atr_period).mean()

        def calc_pct_rank(window):
            if len(window) < lookback_days or np.isnan(window[-1]):
                return 50.0  # default normal volatility if insufficient history
            val = window[-1]
            return (np.sum(window <= val) / len(window)) * 100.0

        pct_df[sym] = atr14.rolling(window=lookback_days).apply(calc_pct_rank, raw=True).fillna(50.0)

    return pct_df


def compute_trend_quality(prices_df: pd.DataFrame, fast_span: int = 20, slow_span: int = 50) -> pd.DataFrame:
    """
    Returns boolean DataFrame: True if EMA20 > EMA50 for each asset.
    """
    ema_fast = prices_df.ewm(span=fast_span, adjust=False).mean()
    ema_slow = prices_df.ewm(span=slow_span, adjust=False).mean()
    return ema_fast > ema_slow
