"""
strategy_lab/phase21/run_decomposition.py
------------------------------------------
Driver script for Phase 21: Baseline Edge Decomposition & Forensic Attribution.
Executes 8-dimensional forensic diagnosis of Trend_EMA_50 over the full verified timeline (2020-08-11 to 2026-09-30).
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.phase18.data.loader import load_phase18_price_panel
from strategy_lab.phase21.decomposition_engine import run_full_baseline_decomposition

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=================================================================")
    print("  PHASE 21: BASELINE EDGE DECOMPOSITION & FORENSIC ATTRIBUTION   ")
    print("=================================================================")
    print("Strategy: Trend_EMA_50 Equal-Weighted | Universe: 9 Assets")

    # Load Full Timeline (2020-08-11 to 2026-09-30)
    prices_df = load_phase18_price_panel(
        start_ts="2020-08-11 00:00:00+00:00",
        end_ts="2026-09-30 23:59:59+00:00",
    )
    print(f"Loaded Full Historical Panel: {len(prices_df)} daily bars ({prices_df.index[0]} to {prices_df.index[-1]})\n")

    res = run_full_baseline_decomposition(prices_df, canonical_bps=30.0)

    # 1. Print Asset Attribution
    print("--- 1. ASSET-LEVEL ATTRIBUTION ---")
    print(f"{'Symbol':10s} | {'Net Return':10s} | {'Sharpe':6s} | {'Max DD':8s} | {'Trades':6s} | {'Win Rate':8s} | {'Avg Win':8s} | {'Avg Loss':8s} | {'PF':6s}")
    print("-" * 88)
    for sym, m in res["asset_attribution"].items():
        print(f"{sym:10s} | {m['net_annualized_return_pct']:+9.2f}% | {m['net_sharpe']:6.3f} | {m['max_drawdown_pct']:7.2f}% | {m['total_trades']:6d} | {m['win_rate_pct']:7.1f}% | {m['avg_win_pct']:+7.2f}% | {m['avg_loss_pct']:+7.2f}% | {m['profit_factor']:5.2f}")

    # 2. Print Yearly Attribution
    print("\n--- 2. YEAR-BY-YEAR ATTRIBUTION ---")
    print(f"{'Year':6s} | {'Net Return':10s} | {'Sharpe':6s} | {'Max DD':8s} | {'Exposure':8s} | {'Turnover':8s}")
    print("-" * 58)
    for yr, m in res["yearly_attribution"].items():
        print(f"{yr:6d} | {m['net_annualized_return_pct']:+9.2f}% | {m['net_sharpe']:6.3f} | {m['max_drawdown_pct']:7.2f}% | {m['average_exposure_pct']:7.1f}% | {m['annualized_turnover']:8.2f}")

    # 3. Print Market Regime Attribution
    print("\n--- 3. MARKET REGIME DECOMPOSITION ---")
    print(f"{'Regime':22s} | {'Days Count':10s} | {'Share %':8s} | {'Daily bps':10s} | {'Ann. Return':12s} | {'Win Days %':10s}")
    print("-" * 84)
    for reg, m in res["regime_attribution"].items():
        print(f"{reg:22s} | {m['days_count']:10d} | {m['days_pct']:7.1f}% | {m['daily_mean_bps']:+9.2f} | {m['annualized_net_return_pct']:+11.2f}% | {m['win_day_pct']:9.1f}%")

    # 4. Print Trade Expectancy & Tail Concentration
    te = res["trade_expectancy"]
    print("\n--- 4. TRADE EXPECTANCY & TAIL CONCENTRATION ---")
    print(f"  * Total Trades Across Universe: {te['total_trades_across_universe']}")
    print(f"  * Win Rate                    : {te['win_rate_pct']:.1f}%")
    print(f"  * Median Trade Return         : {te['median_trade_pct']:+.2f}%")
    print(f"  * Mean Trade Return           : {te['mean_trade_pct']:+.2f}%")
    print(f"  * Average Winner vs Loser     : {te['average_winner_pct']:+.2f}% vs {te['average_loser_pct']:+.2f}% (Payoff Ratio: {te['payoff_ratio']:.2f})")
    print(f"  * Overall Profit Factor       : {te['overall_profit_factor']:.2f}")
    print(f"  * Top 5 Trades Profit Share   : {te['top_5_trades_profit_share_pct']:.1f}% of total gains")
    print(f"  * Top 10% Trades Profit Share : {te['top_10pct_trades_profit_share_pct']:.1f}% of total gains")

    # 5. Print Friction Sensitivity Curve
    print("\n--- 5. FRICTION BREAK-EVEN SENSITIVITY ---")
    print(f"{'Friction':10s} | {'Net Return':10s} | {'Net Sharpe':10s} | {'Max DD':8s}")
    print("-" * 46)
    for bps_str, m in res["friction_sensitivity"]["friction_curve"].items():
        print(f"{bps_str:10s} | {m['net_annualized_return_pct']:+9.2f}% | {m['net_sharpe']:10.3f} | {m['max_drawdown_pct']:7.2f}%")
    print(f"--> Break-even friction: {res['friction_sensitivity']['break_even_bps']}")

    # 6. Print Benchmark Comparison
    print("\n--- 6. DESCRIPTIVE BENCHMARK LANDSCAPE ---")
    print(f"{'Strategy / Benchmark':26s} | {'Net Return':10s} | {'Net Sharpe':10s} | {'Max DD':8s} | {'Turnover Drag':12s}")
    print("-" * 75)
    for name, m in res["benchmark_comparison"].items():
        print(f"{name:26s} | {m['net_annualized_return_pct']:+9.2f}% | {m['net_sharpe']:10.3f} | {m['max_drawdown_pct']:7.2f}% | {m['turnover_drag_pct']:11.2f}%")

    # Save JSON Report
    json_path = REPORTS_DIR / "phase21_baseline_decomposition_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    # Save Markdown Report
    md_path = REPORTS_DIR / "phase21_baseline_decomposition_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 21: Baseline Edge Decomposition & Forensic Attribution Report\n\n")
        f.write(f"**Timeline**: 2020-08-11 to 2026-09-30 ($N={len(prices_df)}$ Daily Bars)  \n")
        f.write("**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  \n")
        f.write(f"**Canonical Strategy**: `Trend_EMA_50` Equal-Weighted (30.0 bps round-trip friction)  \n\n")

        f.write("## 1. Asset-Level Performance Attribution\n\n")
        f.write("| Symbol | Net Ann. Return | Net Sharpe | Max Drawdown | Total Trades | Win Rate | Avg Win | Avg Loss | Profit Factor |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for sym, m in res["asset_attribution"].items():
            f.write(f"| `{sym}` | {m['net_annualized_return_pct']:+.2f}% | {m['net_sharpe']:.3f} | {m['max_drawdown_pct']:.2f}% | {m['total_trades']} | {m['win_rate_pct']:.1f}% | {m['avg_win_pct']:+.2f}% | {m['avg_loss_pct']:+.2f}% | {m['profit_factor']:.2f} |\n")

        f.write("\n## 2. Year-by-Year Calendar Attribution\n\n")
        f.write("| Year | Net Ann. Return | Net Sharpe | Max Drawdown | Avg Exposure | Annual Turnover |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for yr, m in res["yearly_attribution"].items():
            f.write(f"| `{yr}` | {m['net_annualized_return_pct']:+.2f}% | {m['net_sharpe']:.3f} | {m['max_drawdown_pct']:.2f}% | {m['average_exposure_pct']:.1f}% | {m['annualized_turnover']:.2f} |\n")

        f.write("\n## 3. Market Regime Decomposition\n\n")
        f.write("| Regime | Days Count | Share of Time | Daily Return (bps) | Ann. Return | Win Day % |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for reg, m in res["regime_attribution"].items():
            f.write(f"| `{reg}` | {m['days_count']} | {m['days_pct']:.1f}% | {m['daily_mean_bps']:+.2f} bps | {m['annualized_net_return_pct']:+.2f}% | {m['win_day_pct']:.1f}% |\n")

        f.write("\n## 4. Trade Expectancy & Profit Concentration\n\n")
        f.write(f"- **Total Trades Extracted**: {te['total_trades_across_universe']}\n")
        f.write(f"- **Win Rate**: {te['win_rate_pct']:.1f}%\n")
        f.write(f"- **Median Trade Return**: {te['median_trade_pct']:+.2f}%\n")
        f.write(f"- **Average Winner**: {te['average_winner_pct']:+.2f}% vs **Average Loser**: {te['average_loser_pct']:+.2f}%\n")
        f.write(f"- **Payoff Ratio (Win/Loss Size)**: **{te['payoff_ratio']:.2f}x**\n")
        f.write(f"- **Overall Profit Factor**: **{te['overall_profit_factor']:.2f}**\n")
        f.write(f"- **Top 5 Trades Profit Share**: **{te['top_5_trades_profit_share_pct']:.1f}%** of total gains\n")
        f.write(f"- **Top 10% Trades Profit Share**: **{te['top_10pct_trades_profit_share_pct']:.1f}%** of total gains\n")

        f.write("\n## 5. Friction Sensitivity & Break-Even Analysis\n\n")
        f.write("| Friction Scenario | Net Ann. Return | Net Sharpe | Max Drawdown |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for bps_str, m in res["friction_sensitivity"]["friction_curve"].items():
            f.write(f"| `{bps_str}` | {m['net_annualized_return_pct']:+.2f}% | {m['net_sharpe']:.3f} | {m['max_drawdown_pct']:.2f}% |\n")
        f.write(f"\n**Economic Break-Even Friction**: `{res['friction_sensitivity']['break_even_bps']}`\n")

        f.write("\n## 6. Descriptive Benchmark Comparison\n\n")
        f.write("| Strategy / Benchmark | Net Ann. Return | Net Sharpe | Max Drawdown | Annual Friction Drag |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for name, m in res["benchmark_comparison"].items():
            f.write(f"| `{name}` | {m['net_annualized_return_pct']:+.2f}% | {m['net_sharpe']:.3f} | {m['max_drawdown_pct']:.2f}% | {m['turnover_drag_pct']:.2f}% |\n")

    print(f"\n[OK] Reports saved to {json_path} and {md_path}")
    return res


if __name__ == "__main__":
    main()
