from __future__ import annotations
"""
strategy_lab/run_benchmark_matrix.py
------------------------------------
Strategy Lab Multi-Asset, Multi-Timeframe Benchmark Matrix.
Evaluates Strategy A, B, C, D alongside Phase 13 baseline under identical cost model (10 BPS fees, 5 BPS slippage).

PURELY OFFLINE RESEARCH — ZERO LIVE EXECUTION, ZERO PHASE 13 CONTAMINATION.
"""

import sys
import os
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.data_loader import load_ohlcv
from strategy_lab.evaluator import run_strategy_backtest
from strategy_lab.strategies.trend_following import TrendFollowingStrategy
from strategy_lab.strategies.breakout import MomentumBreakoutStrategy
from strategy_lab.strategies.mean_reversion import MeanReversionStrategy
from strategy_lab.strategies.ml_regime_filter import MLRegimeFilteredStrategy

def resample_ohlcv(df: pd.DataFrame, target_tf: str = "4h") -> pd.DataFrame:
    """Resample 1h DataFrame into 4h OHLCV bars, strictly dropping incomplete trailing bars."""
    df_copy = df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)
    
    counts = df_copy.set_index("timestamp").resample(target_tf)["close"].count()
    resampled = df_copy.set_index("timestamp").resample(target_tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    expected_bars = 4 if target_tf.lower() == "4h" else 1
    resampled = resampled[counts >= expected_bars].reset_index()
    return resampled

def main():
    print("=========================================================================")
    print("          STRATEGY LAB MULTI-FAMILY BENCHMARK MATRIX")
    print("=========================================================================")
    print("Cost Model: 10 BPS Exchange Taker Fee + 5 BPS Slippage per side (30 BPS RT)")
    print("Risk Engine: 1.0% Account Equity Risk per trade scenario")
    print("=========================================================================\n")

    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["1h", "4h"]

    benchmark_results = []

    for sym in symbols:
        # Load full 1h history
        df_1h = load_ohlcv(sym, "1h", limit=2000)
        df_4h = resample_ohlcv(df_1h, "4h")

        tf_map = {"1h": df_1h, "4h": df_4h}

        for tf in timeframes:
            df_target = tf_map[tf]

            # Instantiate Strategy Candidates
            candidates = [
                TrendFollowingStrategy(sym, tf),
                MomentumBreakoutStrategy(sym, tf),
                MeanReversionStrategy(sym, tf),
                MLRegimeFilteredStrategy(sym, tf),
            ]

            for strat in candidates:
                res = run_strategy_backtest(strat, df_target, initial_capital=10000.0, warmup_bars=100)
                benchmark_results.append(res)

    # Convert results to clean summary DataFrame
    summary_rows = []
    for r in benchmark_results:
        summary_rows.append({
            "Strategy": r["strategy"],
            "Asset": r["symbol"],
            "Timeframe": r["timeframe"],
            "Trades (N)": r["total_trades"],
            "Win Rate %": f"{r['win_rate_pct']:.1f}%",
            "Profit Factor": f"{r['profit_factor']:.2f}",
            "Net P&L ($)": f"${r['net_pnl_usd']:+,.2f}",
            "Net P&L %": f"{r['net_pnl_pct']:+.2f}%",
            "Max DD %": f"{r['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{r['sharpe_ratio']:.2f}",
        })

    df_res = pd.DataFrame(summary_rows)

    # Format markdown table manually
    cols = list(df_res.columns)
    md_table = "| " + " | ".join(cols) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, row in df_res.iterrows():
        md_table += "| " + " | ".join([str(row[c]) for c in cols]) + " |\n"

    print("\n### STRATEGY LAB BENCHMARK MATRIX SUMMARY\n")
    print(md_table)

    # Save to Markdown Report in strategy_lab/reports/
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "benchmark_summary.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Strategy Lab Benchmark Summary Report\n\n")
        f.write(md_table)
        f.write("\n\n*Purely offline research — Zero Phase 13 live execution contamination.*\n")

    print(f"\nSaved benchmark report to {report_path}")

if __name__ == "__main__":
    main()
