"""
strategy_lab/lowturn/metrics.py
-------------------------------
Comprehensive performance and risk metrics suite for Phase 17.
Evaluates:
- Sharpe Ratio (Gross / Net) + 95% Date-Cluster CI
- Max Drawdown & Calmar Ratio
- Annualized Turnover
- Bull / Bear Market Split (BTC > 200d SMA vs <= 200d SMA)
- Cost Sensitivity across [0, 10, 20, 30, 40, 50] bps
- Exposure-matched Null p-value
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .bootstrap import calendar_week_block_bootstrap_sharpe_ci, compute_annualized_sharpe
from .cost_model import SENSITIVITY_GRID_BPS
from .execution import simulate_portfolio_strategy
from .exposure_null import evaluate_exposure_matched_null


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Computes maximum peak-to-trough drawdown."""
    peaks = equity_curve.cummax()
    drawdowns = (equity_curve - peaks) / peaks
    return float(drawdowns.min())


def compute_annual_turnover(turnover_series: pd.Series, periods_per_year: int = 365) -> float:
    """Computes annualized turnover (sum of weight changes annualized)."""
    return float(turnover_series.mean() * periods_per_year)


def evaluate_regime_split(
    port_net_ret: pd.Series,
    btc_df: pd.DataFrame,
    periods_per_year: int = 365,
) -> Dict[str, float]:
    """
    Splits net returns into Bull (BTC > 200d SMA) and Bear (BTC <= 200d SMA) regimes.
    """
    btc_close = btc_df["close"].reindex(port_net_ret.index).ffill()
    sma200 = btc_close.rolling(window=200, min_periods=50).mean()
    bull_mask = btc_close > sma200
    bear_mask = ~bull_mask

    bull_rets = port_net_ret[bull_mask]
    bear_rets = port_net_ret[bear_mask]

    return {
        "bull_sharpe": compute_annualized_sharpe(bull_rets, periods_per_year) if len(bull_rets) > 30 else 0.0,
        "bear_sharpe": compute_annualized_sharpe(bear_rets, periods_per_year) if len(bear_rets) > 30 else 0.0,
        "bull_bars": int(bull_mask.sum()),
        "bear_bars": int(bear_mask.sum()),
    }


def evaluate_cost_sensitivity(
    panel: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.Series],
    periods_per_year: int = 365,
) -> Dict[str, float]:
    """Evaluates strategy Net Sharpe across friction grid [0, 10, 20, 30, 40, 50] bps."""
    sensitivity = {}
    for bps in SENSITIVITY_GRID_BPS:
        res = simulate_portfolio_strategy(panel, weights_dict, round_trip_bps=bps)
        sensitivity[f"sharpe_{int(bps)}bps"] = compute_annualized_sharpe(res["net_return"], periods_per_year)
    return sensitivity


def compute_full_cell_metrics(
    strategy_name: str,
    panel: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.Series],
    btc_df: pd.DataFrame,
    periods_per_year: int = 365,
    n_bootstraps: int = 2000,
    n_null_sims: int = 1000,
) -> Dict[str, Any]:
    """
    Computes all standard Phase 17 metrics for a strategy cell.
    """
    # 1. Base Portfolio Simulation (at canonical 30 bps)
    sim_res = simulate_portfolio_strategy(panel, weights_dict, round_trip_bps=30.0)
    net_ret = sim_res["net_return"]
    gross_ret = sim_res["gross_return"]
    turnover = sim_res["turnover"]
    net_equity = sim_res["net_equity"]

    # 2. Return & Sharpe Stats
    ann_net_ret = float(net_ret.mean() * periods_per_year)
    ann_gross_ret = float(gross_ret.mean() * periods_per_year)
    ann_net_vol = float(net_ret.std() * np.sqrt(periods_per_year))
    gross_sharpe = compute_annualized_sharpe(gross_ret, periods_per_year)
    
    # 3. Calendar-Week Date-Cluster Bootstrap
    net_sharpe, ci_lower, ci_upper = calendar_week_block_bootstrap_sharpe_ci(
        net_ret, n_bootstraps=n_bootstraps, periods_per_year=periods_per_year
    )

    # 4. Drawdown & Calmar
    max_dd = compute_max_drawdown(net_equity)
    calmar = float(ann_net_ret / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

    # 5. Turnover
    ann_turnover = compute_annual_turnover(turnover, periods_per_year)

    # 6. Regime Split
    regime = evaluate_regime_split(net_ret, btc_df, periods_per_year)

    # 7. Cost Sensitivity
    cost_sens = evaluate_cost_sensitivity(panel, weights_dict, periods_per_year)

    # 8. Exposure-Matched Null Test
    p_val, null_mean, null_std = evaluate_exposure_matched_null(
        panel, weights_dict, net_sharpe, n_simulations=n_null_sims
    )

    return {
        "strategy": strategy_name,
        "ann_net_return": ann_net_ret,
        "ann_gross_return": ann_gross_ret,
        "ann_net_vol": ann_net_vol,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "annual_turnover": ann_turnover,
        "exposure_pct": float(sim_res["exposure"].mean() * 100.0),
        "bull_sharpe": regime["bull_sharpe"],
        "bear_sharpe": regime["bear_sharpe"],
        "cost_sensitivity": cost_sens,
        "null_p_value": p_val,
        "null_mean_sharpe": null_mean,
    }
