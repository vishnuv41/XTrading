"""
strategy_lab/phase19/regime_signals.py
---------------------------------------
Signal generation for the 3 pre-registered regime filter candidates:
1. Regime_Breadth_Filter (Breadth >= 0.30)
2. Regime_Strong_Breadth_Filter (Breadth >= 0.60)
3. Regime_Composite_Filter (Close > EMA50 and EMA20 > EMA50 and Breadth >= 0.30 and ATR_Percentile <= 80%)
"""

from typing import Dict
import numpy as np
import pandas as pd

from .regime_engine import compute_market_breadth, compute_asset_atr_percentile, compute_trend_quality


def generate_regime_weights(
    prices_df: pd.DataFrame,
    rule_id: str = "Regime_Breadth_Filter",
    ema_period: int = 50,
) -> pd.DataFrame:
    """
    Computes portfolio target weights for the specified regime rule.
    """
    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)

    ema50 = prices_df.ewm(span=ema_period, adjust=False).mean()
    base_trend_long = prices_df > ema50
    breadth = compute_market_breadth(prices_df, ema_period=ema_period)

    weights_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    if rule_id == "Regime_Breadth_Filter":
        # Enter/Hold if asset Close > EMA50 AND Market Breadth >= 0.30
        favorable_breadth = breadth >= 0.30
        for sym in prices_df.columns:
            long_cond = base_trend_long[sym] & favorable_breadth
            weights_df[sym] = np.where(long_cond, base_weight, 0.0)

    elif rule_id == "Regime_Strong_Breadth_Filter":
        # Enter/Hold if asset Close > EMA50 AND Market Breadth >= 0.60
        favorable_breadth = breadth >= 0.60
        for sym in prices_df.columns:
            long_cond = base_trend_long[sym] & favorable_breadth
            weights_df[sym] = np.where(long_cond, base_weight, 0.0)

    elif rule_id == "Regime_Composite_Filter":
        # Enter/Hold if Close > EMA50 AND EMA20 > EMA50 AND Breadth >= 0.30 AND ATR_Percentile <= 80%
        trend_qual = compute_trend_quality(prices_df, fast_span=20, slow_span=50)
        atr_pct = compute_asset_atr_percentile(prices_df, atr_period=14, lookback_days=60)
        favorable_breadth = breadth >= 0.30

        for sym in prices_df.columns:
            long_cond = (
                base_trend_long[sym]
                & trend_qual[sym]
                & favorable_breadth
                & (atr_pct[sym] <= 80.0)
            )
            weights_df[sym] = np.where(long_cond, base_weight, 0.0)
    else:
        raise ValueError(f"Unknown rule_id: {rule_id}")

    return weights_df
