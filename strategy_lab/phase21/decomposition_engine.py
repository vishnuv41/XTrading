"""
strategy_lab/phase21/decomposition_engine.py
---------------------------------------------
Comprehensive forensic decomposition engine for the canonical Trend_EMA_50 baseline.
Calculates:
1. Asset-Level Attribution
2. Year-by-Year Attribution
3. Market Regime Decomposition
4. Trade-Level Expectancy Distribution
5. Tail Dependence & Concentration Analysis
6. Exposure & Cash Analysis
7. Friction Break-Even Curve (0 to 100 bps)
8. Descriptive Benchmark Comparison
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from strategy_lab.phase18.evaluation.friction import Phase18CostModel
from strategy_lab.phase18.evaluation.backtest import calculate_annualized_metrics, simulate_portfolio, calculate_max_drawdown


def extract_discrete_trades(
    prices: pd.Series,
    weights: pd.Series,
    round_trip_bps: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Extracts individual closed trades from daily price and target weight series.
    Enforces next-bar execution lag (t+1).
    """
    cost_model = Phase18CostModel(round_trip_bps=round_trip_bps)
    pos_held = weights.shift(1).fillna(0.0)

    trades = []
    in_trade = False
    entry_bar_idx = 0
    entry_price = 0.0
    entry_date = None

    dates = prices.index
    n = len(prices)

    for i in range(1, n):
        prev_pos = pos_held.iloc[i - 1]
        curr_pos = pos_held.iloc[i]
        p = prices.iloc[i]
        dt = dates[i]

        # Entry trigger
        if prev_pos == 0.0 and curr_pos > 0.0:
            in_trade = True
            entry_bar_idx = i
            entry_price = p
            entry_date = dt

        # Exit trigger
        elif in_trade and prev_pos > 0.0 and curr_pos == 0.0:
            in_trade = False
            exit_price = p
            exit_date = dt
            holding_bars = i - entry_bar_idx
            
            gross_ret = (exit_price - entry_price) / entry_price
            # Friction: 2 one-way costs
            net_ret = gross_ret - (2.0 * cost_model.one_way_pct)

            trades.append({
                "entry_date": str(entry_date),
                "exit_date": str(exit_date),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "holding_days": int(holding_bars),
                "gross_return_pct": float(gross_ret * 100.0),
                "net_return_pct": float(net_ret * 100.0),
                "is_win": bool(net_ret > 0.0),
            })

    # Close open trade on final bar
    if in_trade:
        exit_price = prices.iloc[-1]
        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - (2.0 * cost_model.one_way_pct)
        trades.append({
            "entry_date": str(entry_date),
            "exit_date": str(dates[-1]),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "holding_days": int(max(1, n - 1 - entry_bar_idx)),
            "gross_return_pct": float(gross_ret * 100.0),
            "net_return_pct": float(net_ret * 100.0),
            "is_win": bool(net_ret > 0.0),
        })

    return trades


def run_full_baseline_decomposition(
    prices_df: pd.DataFrame,
    canonical_bps: float = 30.0,
    friction_sweep_bps: List[float] = None,
) -> Dict[str, Any]:
    """
    Executes full 8-dimensional forensic decomposition of Trend_EMA_50 across the universe.
    """
    if friction_sweep_bps is None:
        friction_sweep_bps = [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 75.0, 100.0]

    n_assets = len(prices_df.columns)
    base_weight = 1.0 / float(n_assets)

    # 1. Generate Baseline EMA50 Signals
    ema50 = prices_df.ewm(span=50, adjust=False).mean()
    long_mask = prices_df > ema50
    baseline_weights = long_mask.astype(float) * base_weight

    # 2. Portfolio Simulation (Canonical 30 bps)
    port_sim = simulate_portfolio(prices_df, baseline_weights, round_trip_bps=canonical_bps)
    port_metrics = calculate_annualized_metrics(port_sim["net_return"], port_sim["gross_return"], port_sim["turnover"])

    # -------------------------------------------------------------
    # DIMENSION 1: Asset-Level Attribution
    # -------------------------------------------------------------
    asset_attribution = {}
    all_asset_trades = []

    for sym in prices_df.columns:
        p_series = prices_df[sym]
        w_series = baseline_weights[sym]
        
        # Single asset backtest (unleveraged, full weight when long)
        single_weights = pd.DataFrame({sym: long_mask[sym].astype(float)}, index=prices_df.index)
        single_sim = simulate_portfolio(pd.DataFrame({sym: p_series}), single_weights, round_trip_bps=canonical_bps)
        single_m = calculate_annualized_metrics(single_sim["net_return"], single_sim["gross_return"], single_sim["turnover"])

        # Extract trades
        sym_trades = extract_discrete_trades(p_series, long_mask[sym].astype(float), round_trip_bps=canonical_bps)
        for t in sym_trades:
            t["symbol"] = sym
        all_asset_trades.extend(sym_trades)

        wins = [t for t in sym_trades if t["is_win"]]
        losses = [t for t in sym_trades if not t["is_win"]]

        win_rate = (len(wins) / len(sym_trades) * 100.0) if sym_trades else 0.0
        avg_win = float(np.mean([t["net_return_pct"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t["net_return_pct"] for t in losses])) if losses else 0.0
        total_gains = sum(t["net_return_pct"] for t in wins)
        total_losses = abs(sum(t["net_return_pct"] for t in losses))
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 999.0

        asset_attribution[sym] = {
            "net_annualized_return_pct": single_m["net_annualized_return"] * 100.0,
            "net_sharpe": single_m["net_sharpe"],
            "max_drawdown_pct": abs(single_m["max_drawdown"]) * 100.0,
            "total_trades": len(sym_trades),
            "win_rate_pct": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "turnover_drag_pct": single_m["turnover_drag_annualized"] * 100.0,
        }

    # -------------------------------------------------------------
    # DIMENSION 2: Year-by-Year Attribution
    # -------------------------------------------------------------
    port_sim["year"] = port_sim.index.year
    yearly_attribution = {}

    for yr, yr_group in port_sim.groupby("year"):
        yr_m = calculate_annualized_metrics(yr_group["net_return"], yr_group["gross_return"], yr_group["turnover"])
        yearly_attribution[int(yr)] = {
            "net_annualized_return_pct": yr_m["net_annualized_return"] * 100.0,
            "net_sharpe": yr_m["net_sharpe"],
            "max_drawdown_pct": abs(yr_m["max_drawdown"]) * 100.0,
            "average_exposure_pct": float(yr_group["exposure"].mean() * 100.0),
            "annualized_turnover": yr_m["annualized_turnover"],
        }

    # -------------------------------------------------------------
    # DIMENSION 3: Market Regime Attribution
    # -------------------------------------------------------------
    # Causal Regime: Breadth = fraction of 9 coins > EMA50
    breadth = long_mask.mean(axis=1)
    ret_series = port_sim["net_return"]

    bull_mask = breadth >= 0.60
    bear_mask = breadth < 0.30
    sideways_mask = (breadth >= 0.30) & (breadth < 0.60)

    regime_attribution = {
        "bull_market_regime": {
            "days_count": int(bull_mask.sum()),
            "days_pct": float(bull_mask.mean() * 100.0),
            "annualized_net_return_pct": float(ret_series[bull_mask].mean() * 365.25 * 100.0) if bull_mask.sum() > 0 else 0.0,
            "daily_mean_bps": float(ret_series[bull_mask].mean() * 10000.0) if bull_mask.sum() > 0 else 0.0,
            "win_day_pct": float((ret_series[bull_mask] > 0).mean() * 100.0) if bull_mask.sum() > 0 else 0.0,
        },
        "sideways_chop_regime": {
            "days_count": int(sideways_mask.sum()),
            "days_pct": float(sideways_mask.mean() * 100.0),
            "annualized_net_return_pct": float(ret_series[sideways_mask].mean() * 365.25 * 100.0) if sideways_mask.sum() > 0 else 0.0,
            "daily_mean_bps": float(ret_series[sideways_mask].mean() * 10000.0) if sideways_mask.sum() > 0 else 0.0,
            "win_day_pct": float((ret_series[sideways_mask] > 0).mean() * 100.0) if sideways_mask.sum() > 0 else 0.0,
        },
        "bear_market_regime": {
            "days_count": int(bear_mask.sum()),
            "days_pct": float(bear_mask.mean() * 100.0),
            "annualized_net_return_pct": float(ret_series[bear_mask].mean() * 365.25 * 100.0) if bear_mask.sum() > 0 else 0.0,
            "daily_mean_bps": float(ret_series[bear_mask].mean() * 10000.0) if bear_mask.sum() > 0 else 0.0,
            "win_day_pct": float((ret_series[bear_mask] > 0).mean() * 100.0) if bear_mask.sum() > 0 else 0.0,
        },
    }

    # -------------------------------------------------------------
    # DIMENSION 4 & 5: Trade Distribution & Tail Concentration
    # -------------------------------------------------------------
    sorted_trades = sorted(all_asset_trades, key=lambda x: x["net_return_pct"], reverse=True)
    n_total_trades = len(sorted_trades)

    all_trade_rets = [t["net_return_pct"] for t in sorted_trades]
    total_net_pnl = sum(all_trade_rets)
    total_gains = sum(r for r in all_trade_rets if r > 0)
    total_losses = abs(sum(r for r in all_trade_rets if r <= 0))

    top_5_trades = sorted_trades[:5]
    top_5_profit = sum(t["net_return_pct"] for t in top_5_trades)
    top_5_profit_pct = (top_5_profit / total_gains * 100.0) if total_gains > 0 else 0.0

    top_10pct_count = max(1, int(n_total_trades * 0.10))
    top_10pct_profit = sum(t["net_return_pct"] for t in sorted_trades[:top_10pct_count])
    top_10pct_profit_pct = (top_10pct_profit / total_gains * 100.0) if total_gains > 0 else 0.0

    bottom_5_trades = sorted_trades[-5:]
    bottom_5_loss = abs(sum(t["net_return_pct"] for t in bottom_5_trades))
    bottom_5_loss_pct = (bottom_5_loss / total_losses * 100.0) if total_losses > 0 else 0.0

    wins = [r for r in all_trade_rets if r > 0]
    losses = [r for r in all_trade_rets if r <= 0]

    trade_expectancy = {
        "total_trades_across_universe": n_total_trades,
        "win_rate_pct": float(len(wins) / n_total_trades * 100.0) if n_total_trades else 0.0,
        "median_trade_pct": float(np.median(all_trade_rets)) if all_trade_rets else 0.0,
        "mean_trade_pct": float(np.mean(all_trade_rets)) if all_trade_rets else 0.0,
        "average_winner_pct": float(np.mean(wins)) if wins else 0.0,
        "average_loser_pct": float(np.mean(losses)) if losses else 0.0,
        "payoff_ratio": float(abs(np.mean(wins) / np.mean(losses))) if losses and np.mean(losses) != 0 else 0.0,
        "overall_profit_factor": float(total_gains / total_losses) if total_losses > 0 else 999.0,
        "top_5_trades_profit_share_pct": top_5_profit_pct,
        "top_10pct_trades_profit_share_pct": top_10pct_profit_pct,
        "bottom_5_trades_loss_share_pct": bottom_5_loss_pct,
        "top_5_trades_list": top_5_trades,
        "bottom_5_trades_list": bottom_5_trades,
    }

    # -------------------------------------------------------------
    # DIMENSION 6: Exposure & Cash Analysis
    # -------------------------------------------------------------
    exp_series = port_sim["exposure"]
    exposure_analysis = {
        "mean_exposure_pct": float(exp_series.mean() * 100.0),
        "pct_time_in_full_cash": float((exp_series == 0.0).mean() * 100.0),
        "pct_time_fully_invested": float((exp_series >= 0.99).mean() * 100.0),
        "pct_time_partially_invested": float(((exp_series > 0.0) & (exp_series < 0.99)).mean() * 100.0),
    }

    # -------------------------------------------------------------
    # DIMENSION 7: Friction Sensitivity & Break-Even Curve
    # -------------------------------------------------------------
    friction_curve = {}
    break_even_bps = None

    for bps in friction_sweep_bps:
        sim_bps = simulate_portfolio(prices_df, baseline_weights, round_trip_bps=bps)
        m_bps = calculate_annualized_metrics(sim_bps["net_return"], sim_bps["gross_return"], sim_bps["turnover"])
        friction_curve[f"{int(bps)}bps"] = {
            "net_sharpe": m_bps["net_sharpe"],
            "net_annualized_return_pct": m_bps["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(m_bps["max_drawdown"]) * 100.0,
        }
        if m_bps["net_sharpe"] <= 0.0 and break_even_bps is None:
            break_even_bps = bps

    if break_even_bps is None:
        break_even_bps = "> 100 bps"

    # -------------------------------------------------------------
    # DIMENSION 8: Benchmark Comparison (Descriptive)
    # -------------------------------------------------------------
    # 1. Buy & Hold (equal weighted, 100% long)
    bh_weights = pd.DataFrame(base_weight, index=prices_df.index, columns=prices_df.columns)
    bh_sim = simulate_portfolio(prices_df, bh_weights, round_trip_bps=canonical_bps)
    bh_m = calculate_annualized_metrics(bh_sim["net_return"], bh_sim["gross_return"], bh_sim["turnover"])

    # 2. Static 45% Cash (55% invested equally across 9 assets)
    static_weights = pd.DataFrame(base_weight * 0.55, index=prices_df.index, columns=prices_df.columns)
    static_sim = simulate_portfolio(prices_df, static_weights, round_trip_bps=canonical_bps)
    static_m = calculate_annualized_metrics(static_sim["net_return"], static_sim["gross_return"], static_sim["turnover"])

    # 3. EMA20 (Fast Trend)
    ema20 = prices_df.ewm(span=20, adjust=False).mean()
    ema20_weights = (prices_df > ema20).astype(float) * base_weight
    ema20_sim = simulate_portfolio(prices_df, ema20_weights, round_trip_bps=canonical_bps)
    ema20_m = calculate_annualized_metrics(ema20_sim["net_return"], ema20_sim["gross_return"], ema20_sim["turnover"])

    # 4. EMA100 (Slow Trend)
    ema100 = prices_df.ewm(span=100, adjust=False).mean()
    ema100_weights = (prices_df > ema100).astype(float) * base_weight
    ema100_sim = simulate_portfolio(prices_df, ema100_weights, round_trip_bps=canonical_bps)
    ema100_m = calculate_annualized_metrics(ema100_sim["net_return"], ema100_sim["gross_return"], ema100_sim["turnover"])

    benchmark_comparison = {
        "Trend_EMA_50 (Baseline)": {
            "net_sharpe": port_metrics["net_sharpe"],
            "net_annualized_return_pct": port_metrics["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(port_metrics["max_drawdown"]) * 100.0,
            "turnover_drag_pct": port_metrics["turnover_drag_annualized"] * 100.0,
        },
        "Buy_and_Hold": {
            "net_sharpe": bh_m["net_sharpe"],
            "net_annualized_return_pct": bh_m["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(bh_m["max_drawdown"]) * 100.0,
            "turnover_drag_pct": bh_m["turnover_drag_annualized"] * 100.0,
        },
        "Static_45pct_Cash": {
            "net_sharpe": static_m["net_sharpe"],
            "net_annualized_return_pct": static_m["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(static_m["max_drawdown"]) * 100.0,
            "turnover_drag_pct": static_m["turnover_drag_annualized"] * 100.0,
        },
        "Trend_EMA_20 (Fast)": {
            "net_sharpe": ema20_m["net_sharpe"],
            "net_annualized_return_pct": ema20_m["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(ema20_m["max_drawdown"]) * 100.0,
            "turnover_drag_pct": ema20_m["turnover_drag_annualized"] * 100.0,
        },
        "Trend_EMA_100 (Slow)": {
            "net_sharpe": ema100_m["net_sharpe"],
            "net_annualized_return_pct": ema100_m["net_annualized_return"] * 100.0,
            "max_drawdown_pct": abs(ema100_m["max_drawdown"]) * 100.0,
            "turnover_drag_pct": ema100_m["turnover_drag_annualized"] * 100.0,
        },
    }

    return {
        "portfolio_summary_30bps": port_metrics,
        "asset_attribution": asset_attribution,
        "yearly_attribution": yearly_attribution,
        "regime_attribution": regime_attribution,
        "trade_expectancy": trade_expectancy,
        "exposure_analysis": exposure_analysis,
        "friction_sensitivity": {
            "friction_curve": friction_curve,
            "break_even_bps": break_even_bps,
        },
        "benchmark_comparison": benchmark_comparison,
    }
