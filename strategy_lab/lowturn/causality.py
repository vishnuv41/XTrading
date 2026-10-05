"""
strategy_lab/lowturn/causality.py
---------------------------------
Causality test (Rule R1).
Verifies that strategy signals at index t depend exclusively on observations <= t,
and execution occurs strictly at t+1.
"""

import numpy as np
import pandas as pd
from typing import Callable


def verify_causality_r1(
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    test_bar_idx: int = 200,
) -> bool:
    """
    Verifies Rule R1:
    Modifying any bar at t > test_bar_idx must NOT alter the signal at test_bar_idx.
    """
    base_signal = strategy_fn(df)
    
    # Create corrupted future dataframe
    df_corrupted = df.copy()
    future_mask = np.arange(len(df_corrupted)) > test_bar_idx
    df_corrupted.loc[future_mask, "close"] = df_corrupted.loc[future_mask, "close"] * 1.5
    df_corrupted.loc[future_mask, "high"] = df_corrupted.loc[future_mask, "high"] * 1.5
    df_corrupted.loc[future_mask, "low"] = df_corrupted.loc[future_mask, "low"] * 1.5
    df_corrupted.loc[future_mask, "open"] = df_corrupted.loc[future_mask, "open"] * 1.5

    corrupted_signal = strategy_fn(df_corrupted)

    # Check that signals up to test_bar_idx are identical
    is_causal = np.allclose(
        base_signal.iloc[: test_bar_idx + 1].fillna(0).values,
        corrupted_signal.iloc[: test_bar_idx + 1].fillna(0).values,
        atol=1e-8,
    )
    return bool(is_causal)
