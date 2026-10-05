import os
import sys
from typing import Dict, Any, List
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategy_lab.phase18.manifest_validator import load_manifest
from strategy_lab.phase18.data.loader import load_phase18_price_panel, load_phase18_baseline_weights
from strategy_lab.phase18.relative_strength.momentum_signal import compute_relative_strength_weights
from strategy_lab.phase18.evaluation.gates import evaluate_candidate_gates


def run_h18_b_grid_evaluation() -> Dict[str, Any]:
    """
    Evaluates all pre-registered parameter combinations for Hypothesis H18-B.
    """
    manifest = load_manifest()
    h18_b_spec = manifest["hypotheses"]["H18_B_RelativeStrength"]
    param_grid = h18_b_spec["parameter_grid"]

    print("=================================================================")
    print("  PHASE 18 ALPHA EXPERIMENT: H18-B (Relative Strength Momentum)  ")
    print("=================================================================")
    print(f"Universe: {len(manifest['universe'])} assets | Canonical Friction: 30.0 bps")

    # 1. Load Data
    dev_bounds = manifest["dataset_boundaries"]
    prices_df = load_phase18_price_panel(
        start_ts=dev_bounds["development_start"],
        end_ts=dev_bounds["development_end"],
    )
    print(f"Loaded Development Panel: {len(prices_df)} daily bars ({prices_df.index[0]} to {prices_df.index[-1]})")

    baseline_weights = load_phase18_baseline_weights(prices_df)

    results = []
    
    top_k_list = param_grid["top_k_assets"]
    lookback_list = param_grid["lookback_window_days"]
    rebalance_list = param_grid["rebalance_interval_days"]

    for k in top_k_list:
        for l in lookback_list:
            for r in rebalance_list:
                cand_id = f"H18B_Top{k}_Lookback{l}d_Rebal{r}d"
                print(f"\n--> Evaluating {cand_id}...")

                cand_weights = compute_relative_strength_weights(
                    prices_df=prices_df,
                    lookback_days=l,
                    top_k=k,
                    rebalance_interval_days=r,
                )

                eval_res = evaluate_candidate_gates(
                    prices_df=prices_df,
                    candidate_weights_df=cand_weights,
                    baseline_weights_df=baseline_weights,
                    manifest=manifest,
                )

                g = eval_res["gate_results"]
                cand_m = eval_res["candidate_metrics"]
                base_m = eval_res["baseline_metrics"]
                boot = eval_res["bootstrap"]

                print(f"    Net Sharpe: {cand_m['net_sharpe']:.3f} vs Baseline: {base_m['net_sharpe']:.3f} (dSharpe: {g['gate_2_incremental_sharpe']['delta_sharpe']:+.3f})")
                print(f"    Net Return: {cand_m['net_annualized_return']*100:.2f}% vs Baseline: {base_m['net_annualized_return']*100:.2f}% (dReturn: {g['gate_1_incremental_net_return']['delta_return']*100:+.2f}%)")
                print(f"    Paired 95% Bootstrap CI: [{boot['ci_lower']:+.3f}, {boot['ci_upper']:+.3f}] | p-val: {boot['p_value_one_tailed']:.4f}")
                print(f"    Turnover Drag: {cand_m['turnover_drag_annualized']*100:.2f}% | Max DD: {cand_m['max_drawdown']*100:.2f}%")
                print(f"    GATES PASSED: {eval_res['all_gates_passed']} "
                      f"[G1: {g['gate_1_incremental_net_return']['pass']} | "
                      f"G2: {g['gate_2_incremental_sharpe']['pass']} | "
                      f"G3: {g['gate_3_paired_bootstrap_ci']['pass']} | "
                      f"G4: {g['gate_4_friction_robustness']['pass']} | "
                      f"G5: {g['gate_5_turnover_efficiency']['pass']} | "
                      f"G6: {g['gate_6_regime_stability']['pass']}]")

                results.append({
                    "candidate_id": cand_id,
                    "top_k": k,
                    "lookback_days": l,
                    "rebalance_interval_days": r,
                    "metrics": cand_m,
                    "delta_sharpe": g['gate_2_incremental_sharpe']['delta_sharpe'],
                    "delta_return": g['gate_1_incremental_net_return']['delta_return'],
                    "bootstrap": boot,
                    "gates": g,
                    "all_gates_passed": eval_res["all_gates_passed"],
                })

    summary = {
        "hypothesis": "H18_B_RelativeStrength",
        "dataset_window": "Development (2020–2025)",
        "baseline_metrics": base_m,
        "results": results,
    }

    report_path = "strategy_lab/phase18/reports/h18_b_relative_strength_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {report_path}")

    return summary


if __name__ == "__main__":
    run_h18_b_grid_evaluation()
