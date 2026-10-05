"""
strategy_lab/phase18/evaluation/gates.py
-----------------------------------------
Acceptance gate evaluator for Phase 18.
Checks candidate performance against all 6 pre-registered gates in manifest.json.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from strategy_lab.phase18.manifest_validator import load_manifest
from .backtest import calculate_annualized_metrics, simulate_portfolio
from .bootstrap import paired_bootstrap_delta_sharpe


def evaluate_candidate_gates(
    prices_df: pd.DataFrame,
    candidate_weights_df: pd.DataFrame,
    baseline_weights_df: pd.DataFrame,
    manifest: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Evaluates candidate strategy against baseline across all 6 pre-registered gates.
    """
    if manifest is None:
        manifest = load_manifest()

    gates_spec = manifest["acceptance_gates"]
    seed = manifest["reproducibility"]["random_seed"]
    n_bootstraps = manifest["reproducibility"]["bootstrap_iterations"]

    # 1. Canonical Backtests (30 bps)
    cand_30 = simulate_portfolio(prices_df, candidate_weights_df, round_trip_bps=30.0)
    base_30 = simulate_portfolio(prices_df, baseline_weights_df, round_trip_bps=30.0)

    # 2. Performance Metrics
    cand_metrics_30 = calculate_annualized_metrics(cand_30["net_return"], cand_30["gross_return"], cand_30["turnover"])
    base_metrics_30 = calculate_annualized_metrics(base_30["net_return"], base_30["gross_return"], base_30["turnover"])

    # 3. Sensitivity Backtest (45 bps)
    cand_45 = simulate_portfolio(prices_df, candidate_weights_df, round_trip_bps=45.0)
    cand_metrics_45 = calculate_annualized_metrics(cand_45["net_return"], cand_45["gross_return"], cand_45["turnover"])

    # 4. Paired Bootstrap
    boot_res = paired_bootstrap_delta_sharpe(
        cand_30["net_return"],
        base_30["net_return"],
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Gate 1: Incremental Net Return
    delta_return = cand_metrics_30["net_annualized_return"] - base_metrics_30["net_annualized_return"]
    pass_g1 = bool(delta_return > gates_spec["gate_1_incremental_net_return"]["threshold"])

    # Gate 2: Incremental Sharpe
    delta_sharpe = cand_metrics_30["net_sharpe"] - base_metrics_30["net_sharpe"]
    pass_g2 = bool(delta_sharpe >= gates_spec["gate_2_incremental_sharpe"]["threshold"])

    # Gate 3: Paired Bootstrap CI Lower Bound > 0
    ci_lower = boot_res["ci_lower"]
    pass_g3 = bool(ci_lower > gates_spec["gate_3_paired_bootstrap_ci"]["threshold"])

    # Gate 4: Friction Robustness (Sharpe > 0 at 45 bps)
    sharpe_45 = cand_metrics_45["net_sharpe"]
    pass_g4 = bool(sharpe_45 > gates_spec["gate_4_friction_robustness"]["threshold"])

    # Gate 5: Turnover Efficiency (Friction Drag <= 50% of Gross Excess Return)
    gross_delta_ret = cand_metrics_30["gross_annualized_return"] - base_metrics_30["gross_annualized_return"]
    annual_drag = cand_metrics_30["turnover_drag_annualized"]
    drag_ratio = (annual_drag / gross_delta_ret) if gross_delta_ret > 1e-6 else 999.0
    pass_g5 = bool(drag_ratio <= gates_spec["gate_5_turnover_efficiency"]["threshold"])

    # Gate 6: Regime Stability (Sub-period Calendar Years)
    cand_30["year"] = cand_30.index.year
    g6_spec = gates_spec["gate_6_regime_stability"]
    yearly_results = {}
    pass_g6 = True

    for yr, yr_group in cand_30.groupby("year"):
        yr_m = calculate_annualized_metrics(yr_group["net_return"], yr_group["gross_return"], yr_group["turnover"])
        yr_sharpe = yr_m["net_sharpe"]
        yr_dd_pct = abs(yr_m["max_drawdown"]) * 100.0
        yr_ret_pct = yr_m["net_annualized_return"] * 100.0

        yr_pass = (
            yr_sharpe >= g6_spec["min_annual_sharpe"]
            and yr_dd_pct <= g6_spec["max_annual_drawdown_pct"]
            and yr_ret_pct >= g6_spec["min_annual_net_return_pct"]
        )
        if not yr_pass:
            pass_g6 = False

        yearly_results[int(yr)] = {
            "net_sharpe": yr_sharpe,
            "max_drawdown_pct": yr_dd_pct,
            "net_return_pct": yr_ret_pct,
            "pass": yr_pass,
        }

    all_passed = pass_g1 and pass_g2 and pass_g3 and pass_g4 and pass_g5 and pass_g6

    return {
        "candidate_metrics": cand_metrics_30,
        "baseline_metrics": base_metrics_30,
        "bootstrap": boot_res,
        "gate_results": {
            "gate_1_incremental_net_return": {"delta_return": delta_return, "pass": pass_g1},
            "gate_2_incremental_sharpe": {"delta_sharpe": delta_sharpe, "pass": pass_g2},
            "gate_3_paired_bootstrap_ci": {"ci_lower": ci_lower, "ci_upper": boot_res["ci_upper"], "pass": pass_g3},
            "gate_4_friction_robustness": {"sharpe_45bps": sharpe_45, "pass": pass_g4},
            "gate_5_turnover_efficiency": {"drag_ratio": drag_ratio, "annual_drag": annual_drag, "pass": pass_g5},
            "gate_6_regime_stability": {"yearly_breakdown": yearly_results, "pass": pass_g6},
        },
        "all_gates_passed": all_passed,
    }
