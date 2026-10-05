"""
strategy_lab/phase19/evaluation.py
-----------------------------------
Evaluator for Phase 19: Causal Market Regime Classifier.
Runs pre-registered candidate filters against the frozen 6 acceptance gates and 0–50 bps friction sweep.
"""

import os
import sys
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.phase18.data.loader import load_phase18_price_panel, load_phase18_baseline_weights
from strategy_lab.phase18.evaluation.backtest import simulate_portfolio, calculate_annualized_metrics
from strategy_lab.phase18.evaluation.bootstrap import paired_bootstrap_delta_sharpe
from strategy_lab.phase19.regime_signals import generate_regime_weights

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.json")


def load_phase19_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_phase19_evaluation() -> dict:
    manifest = load_phase19_manifest()
    rules = manifest["regime_rules"]
    friction_sweep = manifest["friction_sweep_bps"]
    canonical_bps = manifest["canonical_friction_bps"]
    gates_spec = manifest["acceptance_gates"]
    seed = gates_spec["bootstrap_seed"]
    n_boot = gates_spec["bootstrap_iterations"]

    print("=================================================================")
    print("  PHASE 19 ALPHA EXPERIMENT: Causal Market Regime Classifier     ")
    print("=================================================================")
    print(f"Universe: {len(manifest['universe'])} assets | Canonical Friction: {canonical_bps} bps")

    # Load Development Panel
    dev_bounds = manifest["dataset_boundaries"]
    prices_df = load_phase18_price_panel(
        start_ts=dev_bounds["development_start"],
        end_ts=dev_bounds["development_end"],
    )
    print(f"Loaded Development Panel: {len(prices_df)} daily bars ({prices_df.index[0]} to {prices_df.index[-1]})")

    baseline_weights = load_phase18_baseline_weights(prices_df)
    base_30 = simulate_portfolio(prices_df, baseline_weights, round_trip_bps=canonical_bps)
    base_m30 = calculate_annualized_metrics(base_30["net_return"], base_30["gross_return"], base_30["turnover"])

    candidate_results = []

    for rule_info in rules:
        rule_id = rule_info["id"]
        print(f"\n--> Evaluating Candidate: {rule_id}...")

        cand_weights = generate_regime_weights(prices_df, rule_id=rule_id)

        # 1. Canonical 30 bps backtest
        cand_30 = simulate_portfolio(prices_df, cand_weights, round_trip_bps=canonical_bps)
        cand_m30 = calculate_annualized_metrics(cand_30["net_return"], cand_30["gross_return"], cand_30["turnover"])

        # 2. Friction Sweep (0 to 50 bps)
        sweep_results = {}
        for bps in friction_sweep:
            sim_bps = simulate_portfolio(prices_df, cand_weights, round_trip_bps=bps)
            m_bps = calculate_annualized_metrics(sim_bps["net_return"], sim_bps["gross_return"], sim_bps["turnover"])
            sweep_results[f"{int(bps)}bps"] = {
                "net_sharpe": m_bps["net_sharpe"],
                "net_return_pct": m_bps["net_annualized_return"] * 100.0,
                "max_drawdown_pct": abs(m_bps["max_drawdown"]) * 100.0,
            }

        # 3. Paired Bootstrap
        boot_res = paired_bootstrap_delta_sharpe(
            cand_30["net_return"],
            base_30["net_return"],
            n_bootstraps=n_boot,
            seed=seed,
        )

        # 4. Yearly breakdown
        cand_30["year"] = cand_30.index.year
        yearly_results = {}
        pass_g6 = True

        for yr, yr_group in cand_30.groupby("year"):
            yr_m = calculate_annualized_metrics(yr_group["net_return"], yr_group["gross_return"], yr_group["turnover"])
            yr_sharpe = yr_m["net_sharpe"]
            yr_dd_pct = abs(yr_m["max_drawdown"]) * 100.0
            yr_ret_pct = yr_m["net_annualized_return"] * 100.0

            yr_pass = (
                yr_sharpe >= gates_spec["min_annual_sharpe"]
                and yr_dd_pct <= gates_spec["max_annual_drawdown_pct"]
                and yr_ret_pct >= gates_spec["min_annual_net_return_pct"]
            )
            if not yr_pass:
                pass_g6 = False

            yearly_results[int(yr)] = {
                "net_sharpe": yr_sharpe,
                "max_drawdown_pct": yr_dd_pct,
                "net_return_pct": yr_ret_pct,
                "pass": yr_pass,
            }

        # Gate Checks
        delta_return = cand_m30["net_annualized_return"] - base_m30["net_annualized_return"]
        delta_sharpe = cand_m30["net_sharpe"] - base_m30["net_sharpe"]
        ci_lower = boot_res["ci_lower"]

        pass_g1 = bool(delta_return > 0.0)
        pass_g2 = bool(delta_sharpe >= gates_spec["min_delta_sharpe"])
        pass_g3 = bool(ci_lower > 0.0)
        pass_g4 = bool(sweep_results["45bps"]["net_sharpe"] > 0.0 and sweep_results["50bps"]["net_sharpe"] > 0.0)
        
        gross_delta_ret = cand_m30["gross_annualized_return"] - base_m30["gross_annualized_return"]
        annual_drag = cand_m30["turnover_drag_annualized"]
        drag_ratio = (annual_drag / gross_delta_ret) if gross_delta_ret > 1e-6 else 999.0
        pass_g5 = bool(drag_ratio <= gates_spec["max_turnover_drag_ratio"])

        all_passed = pass_g1 and pass_g2 and pass_g3 and pass_g4 and pass_g5 and pass_g6

        print(f"    Net Sharpe: {cand_m30['net_sharpe']:.3f} vs Baseline: {base_m30['net_sharpe']:.3f} (dSharpe: {delta_sharpe:+.3f})")
        print(f"    Net Return: {cand_m30['net_annualized_return']*100:.2f}% vs Baseline: {base_m30['net_annualized_return']*100:.2f}% (dReturn: {delta_return*100:+.2f}%)")
        print(f"    Paired 95% Bootstrap CI: [{boot_res['ci_lower']:+.3f}, {boot_res['ci_upper']:+.3f}] | p-val: {boot_res['p_value_one_tailed']:.4f}")
        print(f"    Friction Sweep Net Sharpe: 0bps={sweep_results['0bps']['net_sharpe']:.3f} | 30bps={sweep_results['30bps']['net_sharpe']:.3f} | 50bps={sweep_results['50bps']['net_sharpe']:.3f}")
        print(f"    Max Drawdown: {cand_m30['max_drawdown']*100:.2f}% vs Baseline: {base_m30['max_drawdown']*100:.2f}%")
        print(f"    GATES PASSED: {all_passed} [G1: {pass_g1} | G2: {pass_g2} | G3: {pass_g3} | G4: {pass_g4} | G5: {pass_g5} | G6: {pass_g6}]")

        candidate_results.append({
            "rule_id": rule_id,
            "description": rule_info["description"],
            "metrics_30bps": cand_m30,
            "delta_sharpe": delta_sharpe,
            "delta_return": delta_return,
            "friction_sweep": sweep_results,
            "bootstrap": boot_res,
            "yearly_breakdown": yearly_results,
            "gates": {
                "gate_1_incremental_net_return": {"delta_return": delta_return, "pass": pass_g1},
                "gate_2_incremental_sharpe": {"delta_sharpe": delta_sharpe, "pass": pass_g2},
                "gate_3_paired_bootstrap_ci": {"ci_lower": ci_lower, "ci_upper": boot_res["ci_upper"], "pass": pass_g3},
                "gate_4_severe_friction_survival": {"sharpe_50bps": sweep_results["50bps"]["net_sharpe"], "pass": pass_g4},
                "gate_5_turnover_efficiency": {"drag_ratio": drag_ratio, "pass": pass_g5},
                "gate_6_subperiod_stability": {"pass": pass_g6},
            },
            "all_gates_passed": all_passed,
        })

    summary = {
        "experiment": "Phase 19 Causal Market Regime Classifier",
        "dataset_window": "Development (2020-08-11 to 2024-12-31)",
        "baseline_metrics_30bps": base_m30,
        "results": candidate_results,
    }

    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "phase19_regime_evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {report_file}")

    return summary


if __name__ == "__main__":
    run_phase19_evaluation()
