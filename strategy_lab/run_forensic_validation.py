from __future__ import annotations
"""
strategy_lab/run_forensic_validation.py
----------------------------------------
Executes the 5-Dimension Forensic Suite on exploratory candidates:
1. Candidate B: Momentum Breakout (BTC/USDT 4H)
2. Candidate D: ML Regime Router (ETH/USDT 4H)

PURELY OFFLINE RESEARCH — ZERO LIVE EXECUTION, ZERO PHASE 13 CONTAMINATION.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.data_loader import load_ohlcv
from strategy_lab.run_benchmark_matrix import resample_ohlcv
from strategy_lab.strategies.breakout import MomentumBreakoutStrategy
from strategy_lab.strategies.ml_regime_filter import MLRegimeFilteredStrategy
from strategy_lab.forensic_suite import (
    run_full_history_audit,
    run_walk_forward_validation,
    run_regime_stratification,
    run_cost_sensitivity_analysis,
    run_parameter_perturbation
)

def df_to_md(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table without external dependencies."""
    cols = list(df.columns)
    md = "| " + " | ".join(cols) + " |\n"
    md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join([str(row[c]) for c in cols]) + " |\n"
    return md

def main():
    print("=========================================================================")
    print("      STRATEGY LAB V2 — 5-DIMENSION FORENSIC VALIDATION SUITE")
    print("=========================================================================")
    print("Target Candidates:")
    print("  1. Candidate B: Donchian Volatility Breakout (BTC/USDT 4H)")
    print("  2. Candidate D: ML Regime Router (ETH/USDT 4H)")
    print("=========================================================================\n")

    # Load 4H data
    btc_1h = load_ohlcv("BTC/USDT", "1h", limit=2000)
    btc_4h = resample_ohlcv(btc_1h, "4h")

    eth_1h = load_ohlcv("ETH/USDT", "1h", limit=2000)
    eth_4h = resample_ohlcv(eth_1h, "4h")

    # --- CANDIDATE B: BTC 4H BREAKOUT ---
    print("--- RUNNING FORENSIC SUITE ON CANDIDATE B (BTC 4H BREAKOUT) ---")
    btc_strat = MomentumBreakoutStrategy("BTC/USDT", "4h", params={"channel_period": 20})
    
    # Test 1: Full History Audit
    t1_btc = run_full_history_audit(btc_strat, btc_4h)
    
    # Test 2: Walk-Forward Validation
    t2_btc = run_walk_forward_validation(MomentumBreakoutStrategy, btc_4h, "BTC/USDT", "4h", base_params={"channel_period": 20}, n_windows=3)

    # Test 3: Regime Stratification
    t3_btc = run_regime_stratification(btc_strat, btc_4h)

    # Test 4: Cost Sensitivity Analysis
    t4_btc = run_cost_sensitivity_analysis(btc_strat, btc_4h)

    # Test 5: Parameter Perturbation (Channel Period: 16, 18, 20, 22, 24)
    t5_btc = run_parameter_perturbation(MomentumBreakoutStrategy, "BTC/USDT", "4h", btc_4h, {"channel_period": 20}, "channel_period", [16, 18, 20, 22, 24])


    # --- CANDIDATE D: ETH 4H ML REGIME ROUTER ---
    print("--- RUNNING FORENSIC SUITE ON CANDIDATE D (ETH 4H ML REGIME ROUTER) ---")
    eth_strat = MLRegimeFilteredStrategy("ETH/USDT", "4h")

    # Test 1: Full History Audit
    t1_eth = run_full_history_audit(eth_strat, eth_4h)

    # Test 2: Walk-Forward Validation
    t2_eth = run_walk_forward_validation(MLRegimeFilteredStrategy, eth_4h, "ETH/USDT", "4h", n_windows=3)

    # Test 3: Regime Stratification
    t3_eth = run_regime_stratification(eth_strat, eth_4h)

    # Test 4: Cost Sensitivity Analysis
    t4_eth = run_cost_sensitivity_analysis(eth_strat, eth_4h)

    # Test 5: Parameter Perturbation (SL ATR Multiplier: 1.0, 1.25, 1.5, 1.75, 2.0)
    t5_eth = run_parameter_perturbation(MLRegimeFilteredStrategy, "ETH/USDT", "4h", eth_4h, {"sl_atr_mult": 1.5}, "sl_atr_mult", [1.0, 1.25, 1.5, 1.75, 2.0])

    # Build Markdown Report
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "forensic_validation_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Strategy Lab V2 — Forensic Validation Report\n\n")
        f.write("*Purely offline research — Zero Phase 13 live execution contamination.*\n\n")

        # ----------------------------------------------------
        # CANDIDATE B SECTION
        # ----------------------------------------------------
        f.write("## 1. Candidate B: Donchian Volatility Breakout (BTC/USDT 4H)\n\n")
        
        f.write("### Test 1 — Full History & Causal Parity Audit\n")
        f.write(f"- **Period**: `{t1_btc['start_timestamp']}` to `{t1_btc['end_timestamp']}`\n")
        f.write(f"- **Trades ($N$)**: `{t1_btc['total_trades']}`\n")
        f.write(f"- **Win Rate**: `{t1_btc['win_rate_pct']:.1f}%`\n")
        f.write(f"- **Profit Factor**: `{t1_btc['profit_factor']:.2f}`\n")
        f.write(f"- **Net Return**: `{t1_btc['net_pnl_pct']:+.2f}%` (${t1_btc['net_pnl_usd']:+,.2f})\n")
        f.write(f"- **Max Drawdown**: `{t1_btc['max_drawdown_pct']:.2f}%`\n\n")

        f.write("### Test 2 — Walk-Forward Out-of-Sample (OOS) Validation\n\n")
        wf_btc_rows = []
        for w in t2_btc:
            wf_btc_rows.append({
                "Window": f"Window {w['window_idx']}",
                "Start": w["start_ts"][:10],
                "End": w["end_ts"][:10],
                "Trades (N)": w["total_trades"],
                "Win Rate %": f"{w['win_rate_pct']:.1f}%",
                "Profit Factor": f"{w['profit_factor']:.2f}",
                "Net P&L %": f"{w['net_pnl_pct']:+.2f}%",
                "Max DD %": f"{w['max_drawdown_pct']:.2f}%"
            })
        df_wf_btc = pd.DataFrame(wf_btc_rows)
        f.write(df_to_md(df_wf_btc) + "\n\n")

        f.write("### Test 3 — Market Regime Stratification\n\n")
        reg_btc_rows = []
        for r_name, r_data in t3_btc["regime_breakdown"].items():
            wr = (r_data["wins"] / r_data["trades"] * 100) if r_data["trades"] > 0 else 0.0
            reg_btc_rows.append({
                "Regime": r_name,
                "Trades (N)": r_data["trades"],
                "Wins": r_data["wins"],
                "Losses": r_data["losses"],
                "Win Rate %": f"{wr:.1f}%",
                "Net P&L ($)": f"${r_data['net_pnl']:+,.2f}"
            })
        df_reg_btc = pd.DataFrame(reg_btc_rows)
        f.write(df_to_md(df_reg_btc) + "\n\n")

        f.write("### Test 4 — Cost & Friction Sensitivity Analysis\n\n")
        cost_btc_rows = []
        for cl in t4_btc:
            cost_btc_rows.append({
                "Cost Scenario": cl["cost_scenario"],
                "RT Friction BPS": f"{cl['roundtrip_bps']} BPS",
                "Trades (N)": cl["total_trades"],
                "Win Rate %": f"{cl['win_rate_pct']:.1f}%",
                "Profit Factor": f"{cl['profit_factor']:.2f}",
                "Net P&L %": f"{cl['net_pnl_pct']:+.2f}%",
            })
        df_cost_btc = pd.DataFrame(cost_btc_rows)
        f.write(df_to_md(df_cost_btc) + "\n\n")

        f.write("### Test 5 — Parameter Perturbation & Stability\n\n")
        param_btc_rows = []
        for p in t5_btc:
            param_btc_rows.append({
                "Channel Period": p["param_value"],
                "Trades (N)": p["total_trades"],
                "Win Rate %": f"{p['win_rate_pct']:.1f}%",
                "Profit Factor": f"{p['profit_factor']:.2f}",
                "Net P&L %": f"{p['net_pnl_pct']:+.2f}%",
                "Max DD %": f"{p['max_drawdown_pct']:.2f}%"
            })
        df_param_btc = pd.DataFrame(param_btc_rows)
        f.write(df_to_md(df_param_btc) + "\n\n")

        # ----------------------------------------------------
        # CANDIDATE D SECTION
        # ----------------------------------------------------
        f.write("---\n\n## 2. Candidate D: ML Regime Router (ETH/USDT 4H)\n\n")

        f.write("### Test 1 — Full History & Causal Parity Audit\n")
        f.write(f"- **Period**: `{t1_eth['start_timestamp']}` to `{t1_eth['end_timestamp']}`\n")
        f.write(f"- **Trades ($N$)**: `{t1_eth['total_trades']}`\n")
        f.write(f"- **Win Rate**: `{t1_eth['win_rate_pct']:.1f}%`\n")
        f.write(f"- **Profit Factor**: `{t1_eth['profit_factor']:.2f}`\n")
        f.write(f"- **Net Return**: `{t1_eth['net_pnl_pct']:+.2f}%` (${t1_eth['net_pnl_usd']:+,.2f})\n")
        f.write(f"- **Max Drawdown**: `{t1_eth['max_drawdown_pct']:.2f}%`\n\n")

        f.write("### Test 2 — Walk-Forward Out-of-Sample (OOS) Validation\n\n")
        wf_eth_rows = []
        for w in t2_eth:
            wf_eth_rows.append({
                "Window": f"Window {w['window_idx']}",
                "Start": w["start_ts"][:10],
                "End": w["end_ts"][:10],
                "Trades (N)": w["total_trades"],
                "Win Rate %": f"{w['win_rate_pct']:.1f}%",
                "Profit Factor": f"{w['profit_factor']:.2f}",
                "Net P&L %": f"{w['net_pnl_pct']:+.2f}%",
                "Max DD %": f"{w['max_drawdown_pct']:.2f}%"
            })
        df_wf_eth = pd.DataFrame(wf_eth_rows)
        f.write(df_to_md(df_wf_eth) + "\n\n")

        f.write("### Test 3 — Market Regime Stratification\n\n")
        reg_eth_rows = []
        for r_name, r_data in t3_eth["regime_breakdown"].items():
            wr = (r_data["wins"] / r_data["trades"] * 100) if r_data["trades"] > 0 else 0.0
            reg_eth_rows.append({
                "Regime": r_name,
                "Trades (N)": r_data["trades"],
                "Wins": r_data["wins"],
                "Losses": r_data["losses"],
                "Win Rate %": f"{wr:.1f}%",
                "Net P&L ($)": f"${r_data['net_pnl']:+,.2f}"
            })
        df_reg_eth = pd.DataFrame(reg_eth_rows)
        f.write(df_to_md(df_reg_eth) + "\n\n")

        f.write("### Test 4 — Cost & Friction Sensitivity Analysis\n\n")
        cost_eth_rows = []
        for cl in t4_eth:
            cost_eth_rows.append({
                "Cost Scenario": cl["cost_scenario"],
                "RT Friction BPS": f"{cl['roundtrip_bps']} BPS",
                "Trades (N)": cl["total_trades"],
                "Win Rate %": f"{cl['win_rate_pct']:.1f}%",
                "Profit Factor": f"{cl['profit_factor']:.2f}",
                "Net P&L %": f"{cl['net_pnl_pct']:+.2f}%",
            })
        df_cost_eth = pd.DataFrame(cost_eth_rows)
        f.write(df_to_md(df_cost_eth) + "\n\n")

        f.write("### Test 5 — Parameter Perturbation & Stability\n\n")
        param_eth_rows = []
        for p in t5_eth:
            param_eth_rows.append({
                "SL ATR Mult": p["param_value"],
                "Trades (N)": p["total_trades"],
                "Win Rate %": f"{p['win_rate_pct']:.1f}%",
                "Profit Factor": f"{p['profit_factor']:.2f}",
                "Net P&L %": f"{p['net_pnl_pct']:+.2f}%",
                "Max DD %": f"{p['max_drawdown_pct']:.2f}%"
            })
        df_param_eth = pd.DataFrame(param_eth_rows)
        f.write(df_to_md(df_param_eth) + "\n\n")

    print("\n=========================================================")
    print(f" Saved full forensic report to: {report_path}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
