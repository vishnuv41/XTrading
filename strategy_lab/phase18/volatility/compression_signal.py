"""
strategy_lab/phase18/volatility/compression_signal.py
------------------------------------------------------
Hypothesis H18-C: Pre-Entry Volatility Compression Filter.
Evaluates whether filtering Trend_EMA_50 entries to bars following a volatility compression
(historical Bandwidth or ATR percentile < 20th) improves risk-adjusted returns and drawdown.

Strict Isolation:
The baseline strategy remains the exact 9-asset equal-weighted Trend_EMA_50.
The filter only regulates the entry trigger. Positions are held until Close <= EMA50.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_volatility_compression_weights(
    prices_df: pd.DataFrame,
    compression_type: str = "bandwidth_percentile",
    lookback_days: int = 60,
    compression_percentile_cutoff: float = 20.0,
    ema_period: int = 50,
) -> pd.DataFrame:
    """
    Computes portfolio target weights for H18-C Volatility Compression filter.

    Args:
        prices_df: Close prices (index=DatetimeIndex, columns=symbols).
        compression_type: 'bandwidth_percentile' or 'atr_ratio'.
        lookback_days: Lookback window for percentile calculation (30 or 60 days).
        compression_percentile_cutoff: Cutoff percentile for compression (e.g. 20.0 = bottom 20%).
        ema_period: EMA lookback period (50 days).

    Returns:
        DataFrame of target weights (index=DatetimeIndex, columns=symbols).
    """
    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)

    # 1. Base Trend Indicator
    ema_df = prices_df.ewm(span=ema_period, adjust=False).mean()
    trend_long = prices_df > ema_df

    # 2. Volatility Compression Metric per Asset
    weights_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    for sym in prices_df.columns:
        p = prices_df[sym]
        
        if compression_type == "bandwidth_percentile":
            # 20-day Bollinger Bandwidth: (Upper - Lower) / Middle
            roll_mean = p.rolling(window=20).mean()
            roll_std = p.rolling(window=20).std(ddof=1)
            bb_upper = roll_mean + 2.0 * roll_std
            bb_lower = roll_mean - 2.0 * roll_std
            bandwidth = (bb_upper - bb_lower) / roll_mean
            
            # Rolling percentile rank over lookback_days
            # Rank bandwidth value at bar t against past lookback_days
            def calc_pct_rank(window):
                if len(window) < lookback_days or np.isnan(window[-1]):
                    return np.nan
                val = window[-1]
                return (np.sum(window <= val) / len(window)) * 100.0

            vol_pct = bandwidth.rolling(window=lookback_days).apply(calc_pct_rank, raw=True)
            
        elif compression_type == "atr_ratio":
            # ATR Ratio: Short-term (14d) volatility / Long-term (lookback_days) volatility
            # Approximate daily true range from daily returns
            ret_abs = p.pct_change().abs()
            short_vol = ret_abs.rolling(window=14).mean()
            long_vol = ret_abs.rolling(window=lookback_days).mean()
            vol_ratio = short_vol / long_vol

            def calc_pct_rank_ratio(window):
                if len(window) < lookback_days or np.isnan(window[-1]):
                    return np.nan
                val = window[-1]
                return (np.sum(window <= val) / len(window)) * 100.0

            vol_pct = vol_ratio.rolling(window=lookback_days).apply(calc_pct_rank_ratio, raw=True)
        else:
            raise ValueError(f"Unknown compression_type: {compression_type}")

        # State tracking: entry permitted if trend_long is True AND compression was active at or immediately before entry
        in_pos = False
        sym_trend = trend_long[sym].fillna(False)
        sym_vol_pct = vol_pct.fillna(100.0)

        for i in range(len(prices_df)):
            dt = prices_df.index[i]
            is_trend = sym_trend.iloc[i]
            prev_trend = sym_trend.iloc[i - 1] if i > 0 else False
            curr_pct = sym_vol_pct.iloc[i]

            # Trigger condition: New trend crossover (prev_trend == False, is_trend == True)
            # Entry allowed only if volatility was compressed (curr_pct <= compression_percentile_cutoff)
            if not in_pos:
                if is_trend and not prev_trend:
                    if curr_pct <= compression_percentile_cutoff:
                        in_pos = True
                elif is_trend and in_pos:
                    # Maintain existing position
                    pass
            else:
                # Exit when trend is lost
                if not is_trend:
                    in_pos = False

            if in_pos:
                weights_df.loc[dt, sym] = base_weight
            else:
                weights_df.loc[dt, sym] = 0.0

    return weights_df
