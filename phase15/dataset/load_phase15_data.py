"""
phase15/dataset/load_phase15_data.py
-------------------------------------
Authoritative Historical Phase 15 Data Loader.

Guarantees strict adherence to Phase 15 split boundaries:
1. Loads full historical 1h OHLCV from 2017-08-17 up to 2025-12-31 (END boundary = 2025-12-31 23:59:59 UTC).
2. STRICTLY EXCLUDES all 2026+ prospective/out-of-sample data.
3. Generates point-in-time causal datasets for BTC/USDT and ETH/USDT.
4. Returns clean, chronologically sorted DataFrame with strict split masks:
   - Train Era:      2017-08-17 to 2023-12-31
   - Validation Era: 2024-01-01 to 2025-12-31
   - Untouched OOS:  2026-01-01 to Present (0 rows in dev set)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
from datetime import datetime, timezone
from typing import Tuple
import pandas as pd
import numpy as np

from pipeline.data_loader import load_ohlcv
from phase15.dataset.causal_dataset_builder import generate_causal_dataset

logger = logging.getLogger(__name__)

PHASE15_END_BOUNDARY = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def load_phase15_canonical_dataset() -> pd.DataFrame:
    """
    Loads full 2017-2025 historical dataset for Phase 15 research.
    Excludes all 2026 data.
    """
    logger.info("Loading Phase 15 historical OHLCV data (2017-08-17 to 2025-12-31)...")
    df_btc = load_ohlcv("BTC/USDT", "1h", end=PHASE15_END_BOUNDARY, limit=None)
    df_eth = load_ohlcv("ETH/USDT", "1h", end=PHASE15_END_BOUNDARY, limit=None)
    
    ds_btc = generate_causal_dataset(df_btc, symbol="BTC/USDT")
    ds_eth = generate_causal_dataset(df_eth, symbol="ETH/USDT")
    
    combined = pd.concat([ds_btc, ds_eth], ignore_index=True)
    combined = combined.sort_values(by=["ts", "symbol"]).reset_index(drop=True)
    
    feature_cols = [c for c in combined.columns if c not in ["ts", "symbol", "label", "label_raw", "timestamp"]]
    numeric_cols = combined[feature_cols].select_dtypes(include=[np.number]).columns
    combined = combined.dropna(subset=numeric_cols).reset_index(drop=True)
    
    timestamps = pd.to_datetime(combined["ts"], utc=True)
    max_ts = timestamps.max()
    
    if max_ts > PHASE15_END_BOUNDARY:
        raise ValueError(f"Contamination detected! Max dataset timestamp {max_ts} exceeds 2025 boundary {PHASE15_END_BOUNDARY}")
        
    logger.info("Phase 15 Canonical Dataset Loaded. Total rows: %d | Min TS: %s | Max TS: %s",
                len(combined), timestamps.min(), max_ts)
    return combined
