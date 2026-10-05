"""
strategy_lab/phase18/funding/funding_signal.py
------------------------------------------------
Hypothesis H18-A: Continuous Funding Rate Overlay Strategy.
Evaluates whether continuous 8h perpetual funding-rate information (rolling z-score)
provides incremental predictive or risk-adjusted value beyond Trend_EMA_50 after 30 bps friction.

Strict Causality:
At daily decision timestamp T (00:00 UTC), only funding rates settled strictly <= T are used.
Rolling mean and std are computed causally with no lookahead.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

FUNDING_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "funding_rates"


def load_daily_funding_features(
    symbol: str,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Loads raw 8h funding rates and converts them causally to daily features at 00:00 UTC.
    """
    clean_name = symbol.replace("/", "_")
    csv_file = FUNDING_DATA_DIR / f"{clean_name}_funding.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing funding CSV for {symbol} at {csv_file}")

    df = pd.read_csv(csv_file)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.set_index("ts").sort_index()

    # Daily aggregation: funding observations available at or before 00:00 UTC each day
    # Resample daily (using right closed interval for bar close at 00:00 UTC)
    daily_funding = df["funding_rate"].resample("1D").mean().ffill()

    # Rolling causal statistics (L days)
    roll_mean = daily_funding.rolling(window=lookback_days).mean()
    roll_std = daily_funding.rolling(window=lookback_days).std(ddof=1)
    
    zscore = (daily_funding - roll_mean) / roll_std
    zscore = zscore.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    res = pd.DataFrame({
        "daily_funding": daily_funding,
        "funding_zscore": zscore,
    }, index=daily_funding.index)
    return res


def compute_funding_overlay_weights(
    prices_df: pd.DataFrame,
    lookback_days: int = 30,
    overlay_weight: float = 0.25,
    overlay_mode: str = "contrarian",  # 'contrarian' (dampen on high funding) or 'momentum' (boost on high funding)
    ema_period: int = 50,
) -> pd.DataFrame:
    """
    Computes portfolio target weights for H18-A Continuous Funding Overlay.

    Args:
        prices_df: Close prices (index=DatetimeIndex, columns=symbols).
        lookback_days: Lookback window for funding z-score (14 or 30 days).
        overlay_weight: Scaling coefficient (0.25 or 0.50).
        overlay_mode: 'contrarian' or 'momentum'.
        ema_period: EMA lookback period (50 days).

    Returns:
        DataFrame of target weights (index=DatetimeIndex, columns=symbols).
    """
    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)

    # 1. Base Trend Filter
    ema_df = prices_df.ewm(span=ema_period, adjust=False).mean()
    trend_long = prices_df > ema_df

    weights_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    for sym in prices_df.columns:
        funding_feats = load_daily_funding_features(sym, lookback_days=lookback_days)
        zscore_aligned = funding_feats["funding_zscore"].reindex(prices_df.index).ffill().fillna(0.0)

        # Non-linear squash of z-score using tanh
        z_squashed = np.tanh(zscore_aligned)

        if overlay_mode == "contrarian":
            # High positive funding (crowded longs) -> reduce position size
            # High negative funding (crowded shorts) -> increase position size
            multiplier = 1.0 - (overlay_weight * z_squashed)
        elif overlay_mode == "momentum":
            # High positive funding (strong bullish momentum) -> increase position size
            multiplier = 1.0 + (overlay_weight * z_squashed)
        else:
            raise ValueError(f"Unknown overlay_mode: {overlay_mode}")

        multiplier = np.clip(multiplier, 0.0, 2.0)

        sym_weights = np.where(trend_long[sym], base_weight * multiplier, 0.0)
        weights_df[sym] = sym_weights

    # Normalize across assets so total allocation <= 1.0
    w_sum = weights_df.sum(axis=1)
    scale = np.where(w_sum > 1.0, 1.0 / w_sum, 1.0)
    weights_df = weights_df.mul(scale, axis=0)

    return weights_df
