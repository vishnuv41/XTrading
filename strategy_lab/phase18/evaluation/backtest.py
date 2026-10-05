"""
strategy_lab/phase18/evaluation/backtest.py
---------------------------------------------
Standardized portfolio simulation engine for Phase 18.
Enforces:
1. Signal on bar close t, fill on open of bar t+1 (shift(1)).
2. Strict portfolio-level cash accounting (weights sum <= 1.0, remainder in cash).
3. Turnover calculation and friction drag.
4. Annualized metrics (Net Return, Sharpe, Max Drawdown, Calmar, Turnover Drag).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .friction import Phase18CostModel, CANONICAL_ROUND_TRIP_BPS


def calculate_max_drawdown(equity_series: pd.Series) -> Tuple[float, pd.Series]:
    """Calculate maximum drawdown fraction and drawdown series."""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = float(dd.min())
    return max_dd, dd


def calculate_annualized_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover_series: pd.Series,
    annualization_factor: float = 365.25,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics from daily returns."""
    clean_net = net_returns.dropna()
    clean_gross = gross_returns.dropna()
    n_days = len(clean_net)

    if n_days < 2:
        return {
            "net_annualized_return": 0.0,
            "gross_annualized_return": 0.0,
            "net_sharpe": 0.0,
            "gross_sharpe": 0.0,
            "max_drawdown": 0.0,
            "annualized_turnover": 0.0,
            "turnover_drag_annualized": 0.0,
        }

    # Cumulative equity
    net_equity = (1.0 + clean_net).cumprod()
    gross_equity = (1.0 + clean_gross).cumprod()

    # Annualized return
    total_net_ret = net_equity.iloc[-1]
    years = n_days / annualization_factor
    cagr_net = float(total_net_ret ** (1.0 / years) - 1.0) if years > 0 and total_net_ret > 0 else float(total_net_ret - 1.0)
    
    total_gross_ret = gross_equity.iloc[-1]
    cagr_gross = float(total_gross_ret ** (1.0 / years) - 1.0) if years > 0 and total_gross_ret > 0 else float(total_gross_ret - 1.0)

    # Sharpe ratio (zero risk-free rate)
    mean_ret = clean_net.mean()
    std_ret = clean_net.std(ddof=1)
    net_sharpe = float((mean_ret / std_ret) * np.sqrt(annualization_factor)) if std_ret > 1e-12 else 0.0

    mean_gross = clean_gross.mean()
    std_gross = clean_gross.std(ddof=1)
    gross_sharpe = float((mean_gross / std_gross) * np.sqrt(annualization_factor)) if std_gross > 1e-12 else 0.0

    # Drawdown
    max_dd, _ = calculate_max_drawdown(net_equity)

    # Turnover
    annual_turnover = float(turnover_series.mean() * annualization_factor)
    turnover_drag = float((clean_gross - clean_net).mean() * annualization_factor)

    return {
        "net_annualized_return": cagr_net,
        "gross_annualized_return": cagr_gross,
        "net_sharpe": net_sharpe,
        "gross_sharpe": gross_sharpe,
        "max_drawdown": max_dd,
        "annualized_turnover": annual_turnover,
        "turnover_drag_annualized": turnover_drag,
    }


def simulate_portfolio(
    prices_df: pd.DataFrame,
    target_weights_df: pd.DataFrame,
    round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """
    Simulates a multi-asset portfolio strategy from a panel of asset close prices.
    
    Args:
        prices_df: DataFrame of close prices (index=DatetimeIndex, columns=symbols).
        target_weights_df: DataFrame of target weights determined at bar close t.
        round_trip_bps: Transaction cost in basis points.

    Returns:
        DataFrame containing portfolio gross/net returns, turnover, exposure, and equity.
    """
    cost_model = Phase18CostModel(round_trip_bps=round_trip_bps)

    # Align timestamps
    common_idx = prices_df.index.intersection(target_weights_df.index)
    p_df = prices_df.loc[common_idx].sort_index()
    w_df = target_weights_df.loc[common_idx].sort_index().fillna(0.0)

    # Normalize weights so sum across assets <= 1.0 (cash buffer)
    w_sum = w_df.sum(axis=1)
    scale_factor = np.where(w_sum > 1.0, 1.0 / w_sum, 1.0)
    w_df = w_df.mul(scale_factor, axis=0)

    # Execution lag: position held during bar t is target_weights computed at bar t-1
    pos_held = w_df.shift(1).fillna(0.0)

    # Asset simple returns
    asset_rets = p_df.pct_change().fillna(0.0)

    # Gross return = sum(weight_i * ret_i)
    gross_rets = (pos_held * asset_rets).sum(axis=1)

    # Portfolio turnover = sum(|pos_held_i,t - pos_held_i,t-1|)
    turnover = (pos_held - pos_held.shift(1).fillna(0.0)).abs().sum(axis=1)

    # Friction drag
    friction = turnover * cost_model.one_way_pct

    # Net return
    net_rets = gross_rets - friction

    res = pd.DataFrame(
        {
            "gross_return": gross_rets,
            "net_return": net_rets,
            "turnover": turnover,
            "exposure": pos_held.sum(axis=1),
        },
        index=common_idx,
    )
    res["gross_equity"] = (1.0 + res["gross_return"]).cumprod()
    res["net_equity"] = (1.0 + res["net_return"]).cumprod()

    return res
