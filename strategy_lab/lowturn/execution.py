"""
strategy_lab/lowturn/execution.py
---------------------------------
Execution & Portfolio Simulation Engine (Rule R2 & R3).
Enforces next-bar execution lag (t+1) and transaction costs.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .cost_model import LowTurnoverCostModel, CANONICAL_ROUND_TRIP_BPS


def simulate_single_asset_strategy(
    df: pd.DataFrame,
    target_weights: pd.Series,
    round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """
    Simulate a single asset strategy given target position weights computed at bar close t.
    Execution occurs at bar t+1.
    """
    cost_model = LowTurnoverCostModel(round_trip_bps=round_trip_bps)

    # Align target weights to dataframe index
    weights = target_weights.reindex(df.index).fillna(0.0)

    # Execution lag: position held during bar t is target_weights computed at bar t-1
    pos_held = weights.shift(1).fillna(0.0)

    # Asset simple return
    asset_ret = df["close"].pct_change().fillna(0.0)

    # Turnover occurs when position held changes: |pos_held_t - pos_held_{t-1}|
    turnover = (pos_held - pos_held.shift(1).fillna(0.0)).abs()

    gross_ret = pos_held * asset_ret
    friction = turnover * cost_model.one_way_pct
    net_ret = gross_ret - friction

    res = pd.DataFrame(
        {
            "close": df["close"],
            "target_weight": weights,
            "pos_held": pos_held,
            "turnover": turnover,
            "asset_return": asset_ret,
            "gross_return": gross_ret,
            "friction": friction,
            "net_return": net_ret,
        },
        index=df.index,
    )

    res["gross_equity"] = (1.0 + res["gross_return"]).cumprod()
    res["net_equity"] = (1.0 + res["net_return"]).cumprod()

    return res


def simulate_portfolio_strategy(
    panel: Dict[str, pd.DataFrame],
    target_weights_dict: Dict[str, pd.Series],
    round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """
    Simulates an equal-weighted multi-asset portfolio strategy across all universe assets.
    """
    cost_model = LowTurnoverCostModel(round_trip_bps=round_trip_bps)
    
    asset_results = {}
    for sym, df in panel.items():
        weights = target_weights_dict.get(sym, pd.Series(0.0, index=df.index))
        res = simulate_single_asset_strategy(df, weights, round_trip_bps=round_trip_bps)
        asset_results[sym] = res

    # Combine net returns across assets (equal-weighted)
    all_net_rets = pd.DataFrame({sym: res["net_return"] for sym, res in asset_results.items()})
    all_gross_rets = pd.DataFrame({sym: res["gross_return"] for sym, res in asset_results.items()})
    all_turnovers = pd.DataFrame({sym: res["turnover"] for sym, res in asset_results.items()})
    all_weights = pd.DataFrame({sym: res["pos_held"] for sym, res in asset_results.items()})

    port_net_ret = all_net_rets.mean(axis=1).fillna(0.0)
    port_gross_ret = all_gross_rets.mean(axis=1).fillna(0.0)
    port_turnover = all_turnovers.mean(axis=1).fillna(0.0)
    port_exposure = all_weights.mean(axis=1).fillna(0.0)

    port_df = pd.DataFrame(
        {
            "gross_return": port_gross_ret,
            "net_return": port_net_ret,
            "turnover": port_turnover,
            "exposure": port_exposure,
        },
        index=port_net_ret.index,
    )

    port_df["gross_equity"] = (1.0 + port_df["gross_return"]).cumprod()
    port_df["net_equity"] = (1.0 + port_df["net_return"]).cumprod()

    return port_df
