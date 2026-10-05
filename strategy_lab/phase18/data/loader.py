"""
strategy_lab/phase18/data/loader.py
------------------------------------
Phase 18 isolated historical data loader.
Enforces:
1. Strict firewall checking against forward ledger dates.
2. Verified common historical universe alignment.
3. Clean OHLCV panel loading.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
from datetime import datetime, timezone

from strategy_lab.lowturn.panel_loader import load_symbol_ohlcv, UNIVERSE_SYMBOLS
from strategy_lab.phase18.manifest_validator import check_dataset_boundaries, load_manifest


def load_phase18_price_panel(
    start_ts: str = "2020-01-01 00:00:00+00:00",
    end_ts: str = "2026-09-30 23:59:59+00:00",
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Loads close price panel for the 9-asset universe, aligned by timestamp.
    Enforces boundary checking.
    """
    check_dataset_boundaries(start_ts, end_ts)
    manifest = load_manifest()
    symbols = manifest["universe"]

    start_dt = datetime.fromisoformat(start_ts) if isinstance(start_ts, str) else start_ts
    end_dt = datetime.fromisoformat(end_ts) if isinstance(end_ts, str) else end_ts

    price_series = {}
    for sym in symbols:
        df = load_symbol_ohlcv(sym, timeframe=timeframe, start_ts=start_dt, end_ts=end_dt)
        price_series[sym] = df["close"]

    prices_df = pd.DataFrame(price_series)
    # Forward fill gaps and drop leading NaNs across common universe
    prices_df = prices_df.ffill().dropna()
    return prices_df


def load_phase18_baseline_weights(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates target weights for the reference Trend_EMA_50 equal-weighted baseline.
    Weight = 1/9 for asset if close > EMA50, else 0.
    """
    ema50 = prices_df.ewm(span=50, adjust=False).mean()
    long_mask = prices_df > ema50
    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)
    
    weights_df = long_mask.astype(float) * base_weight
    return weights_df
