"""
strategy_lab/phase20/portfolio_signals.py
------------------------------------------
Portfolio target weight generators for Phase 20 pre-registered mechanisms:
1. Risk_Parity_Sizing
2. Breadth_Scaled_Exposure
3. Composite_Risk_Portfolio
"""

from typing import Dict
import numpy as np
import pandas as pd

from .portfolio_engine import compute_normalized_atr_volatility, compute_continuous_breadth_exposure


def generate_portfolio_weights(
    prices_df: pd.DataFrame,
    mechanism_id: str = "Risk_Parity_Sizing",
    ema_period: int = 50,
) -> pd.DataFrame:
    """
    Computes portfolio target weights for the specified Phase 20 mechanism.
    """
    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)

    # Base Trend Signals (Long if Close > EMA50)
    ema50 = prices_df.ewm(span=ema_period, adjust=False).mean()
    trend_long = prices_df > ema50

    vol_df = compute_normalized_atr_volatility(prices_df, atr_period=14)
    breadth_exp = compute_continuous_breadth_exposure(prices_df, ema_period=ema_period, saturation_breadth=0.60)

    weights_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    if mechanism_id == "Risk_Parity_Sizing":
        # Weight proportional to 1 / vol for all trending assets
        inv_vol = 1.0 / vol_df
        # Mask non-trending assets
        masked_inv_vol = inv_vol.where(trend_long, 0.0)

        for i in range(len(prices_df)):
            dt = prices_df.index[i]
            row_inv = masked_inv_vol.iloc[i]
            total_inv = row_inv.sum()
            n_open = trend_long.iloc[i].sum()

            if total_inv > 1e-12 and n_open > 0:
                # Target exposure equals (n_open / n_assets)
                target_exp = float(n_open) / float(n_assets)
                weights_df.iloc[i] = (row_inv / total_inv) * target_exp
            else:
                weights_df.iloc[i] = 0.0

    elif mechanism_id == "Breadth_Scaled_Exposure":
        # Equal-weight open signals scaled by continuous breadth exposure
        for i in range(len(prices_df)):
            dt = prices_df.index[i]
            is_long = trend_long.iloc[i]
            n_open = is_long.sum()
            e_t = breadth_exp.iloc[i]

            if n_open > 0:
                # Raw equal weight = 1/9 * e_t for each open position
                w_per_asset = base_weight * e_t
                weights_df.iloc[i] = np.where(is_long, w_per_asset, 0.0)
            else:
                weights_df.iloc[i] = 0.0

    elif mechanism_id == "Composite_Risk_Portfolio":
        # Risk parity weighting scaled by continuous breadth exposure
        inv_vol = 1.0 / vol_df
        masked_inv_vol = inv_vol.where(trend_long, 0.0)

        for i in range(len(prices_df)):
            dt = prices_df.index[i]
            row_inv = masked_inv_vol.iloc[i]
            total_inv = row_inv.sum()
            n_open = trend_long.iloc[i].sum()
            e_t = breadth_exp.iloc[i]

            if total_inv > 1e-12 and n_open > 0:
                target_exp = (float(n_open) / float(n_assets)) * e_t
                weights_df.iloc[i] = (row_inv / total_inv) * target_exp
            else:
                weights_df.iloc[i] = 0.0
    else:
        raise ValueError(f"Unknown mechanism_id: {mechanism_id}")

    # Bounds check: sum <= 1.0
    w_sum = weights_df.sum(axis=1)
    scale = np.where(w_sum > 1.0, 1.0 / w_sum, 1.0)
    weights_df = weights_df.mul(scale, axis=0)

    return weights_df
