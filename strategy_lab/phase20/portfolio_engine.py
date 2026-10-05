"""
strategy_lab/phase20/portfolio_engine.py
-----------------------------------------
Causal Portfolio Construction & Sizing Engine for Phase 20.
Implements:
1. Volatility-Inverse Risk Parity Weighting (1 / ATR_pct).
2. Continuous Market Breadth Exposure Scaling.
Strictly Causal: Uses only data available at bar close t.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_normalized_atr_volatility(prices_df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Computes normalized daily ATR volatility fraction (ATR / Close) for each asset.
    """
    vol_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for sym in prices_df.columns:
        p = prices_df[sym]
        tr = p.pct_change().abs()
        atr = tr.rolling(window=atr_period).mean().fillna(0.03)  # default 3% daily vol fallback
        atr_clean = atr.replace(0.0, 0.03)
        vol_df[sym] = atr_clean
    return vol_df


def compute_continuous_breadth_exposure(prices_df: pd.DataFrame, ema_period: int = 50, saturation_breadth: float = 0.60) -> pd.Series:
    """
    Computes continuous portfolio exposure scaling E_t = clip(Breadth / saturation_breadth, 0.20, 1.0).
    """
    ema_df = prices_df.ewm(span=ema_period, adjust=False).mean()
    trending_df = prices_df > ema_df
    breadth = trending_df.mean(axis=1)

    # Continuous scaling: at Breadth=0.60 exposure is 1.0; below 0.60 scales down to 0.20
    scaled_exposure = np.clip(breadth / saturation_breadth, 0.20, 1.0)
    return scaled_exposure
