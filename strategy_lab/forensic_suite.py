from __future__ import annotations
"""
strategy_lab/forensic_suite.py
------------------------------
5-Dimension Forensic Stress-Testing Suite for Strategy Lab V2 candidates.

Test 1 — Full History & Causal Parity Audit
Test 2 — Rolling Walk-Forward Out-of-Sample Validation
Test 3 — Market Regime Stratification Analysis
Test 4 — Cost & Friction Sensitivity Analysis (0-50 BPS)
Test 5 — Parameter Perturbation & Stability Testing
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Type
from strategy_lab.base_strategy import BaseStrategy
from strategy_lab.evaluator import run_strategy_backtest

def run_full_history_audit(strategy: BaseStrategy, df: pd.DataFrame) -> dict:
    """Test 1: Full History Causal Audit"""
    start_ts = str(df["timestamp"].iloc[0])
    end_ts = str(df["timestamp"].iloc[-1])
    res = run_strategy_backtest(strategy, df, warmup_bars=100)
    res["start_timestamp"] = start_ts
    res["end_timestamp"] = end_ts
    return res

def run_walk_forward_validation(
    strategy_cls: Type[BaseStrategy],
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_params: dict = None,
    n_windows: int = 4
) -> list[dict]:
    """Test 2: Rolling Walk-Forward Out-of-Sample (OOS) Validation"""
    n = len(df)
    window_size = int(n / (n_windows + 1) * 2)
    step_size = int((n - window_size) / (n_windows - 1)) if n_windows > 1 else window_size

    wf_results = []
    for w in range(n_windows):
        start_idx = w * step_size
        end_idx = min(n, start_idx + window_size)
        df_sub = df.iloc[start_idx:end_idx].reset_index(drop=True)
        if len(df_sub) < 120:
            continue

        strat = strategy_cls(symbol, timeframe, params=base_params)
        res = run_strategy_backtest(strat, df_sub, warmup_bars=50)
        res["window_idx"] = w + 1
        res["start_ts"] = str(df_sub["timestamp"].iloc[0])
        res["end_ts"] = str(df_sub["timestamp"].iloc[-1])
        wf_results.append(res)

    return wf_results

def run_regime_stratification(strategy: BaseStrategy, df: pd.DataFrame) -> dict:
    """Test 3: Market Regime Stratification Analysis"""
    df_copy = df.copy()
    c = df_copy["close"]
    h = df_copy["high"]
    l = df_copy["low"]

    df_copy["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df_copy["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df_copy["tr"] = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    df_copy["atr"] = df_copy["tr"].rolling(14).mean()
    df_copy["atr_pct"] = (df_copy["atr"] / c) * 100.0

    atr_80 = df_copy["atr_pct"].quantile(0.80)
    atr_20 = df_copy["atr_pct"].quantile(0.20)

    # Classify Regimes per bar
    def get_regime(row):
        close_p = row["close"]
        ema20 = row["ema_20"]
        ema50 = row["ema_50"]
        atr_p = row["atr_pct"]

        if atr_p >= atr_80:
            return "HIGH_VOLATILITY"
        elif atr_p <= atr_20:
            return "LOW_VOLATILITY"
        elif close_p > ema20 > ema50:
            return "BULLISH_TREND"
        elif close_p < ema20 < ema50:
            return "BEARISH_TREND"
        else:
            return "SIDEWAYS_RANGING"

    df_copy["regime"] = df_copy.apply(get_regime, axis=1)

    full_res = run_strategy_backtest(strategy, df_copy, warmup_bars=100)
    trades = full_res.get("completed_trades", [])

    regime_breakdown = {}
    for r_name in ["BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS_RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY"]:
        regime_breakdown[r_name] = {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0}

    for t in trades:
        entry_ts = t["entry_ts"]
        matched_row = df_copy[df_copy["timestamp"] == entry_ts]
        if not matched_row.empty:
            r_type = matched_row["regime"].iloc[0]
            regime_breakdown[r_type]["trades"] += 1
            if t["net_pnl"] > 0:
                regime_breakdown[r_type]["wins"] += 1
            else:
                regime_breakdown[r_type]["losses"] += 1
            regime_breakdown[r_type]["net_pnl"] += t["net_pnl"]

    return {
        "strategy": strategy.name,
        "symbol": strategy.symbol,
        "timeframe": strategy.timeframe,
        "regime_breakdown": regime_breakdown
    }

def run_cost_sensitivity_analysis(strategy: BaseStrategy, df: pd.DataFrame) -> list[dict]:
    """Test 4: Cost & Friction Sensitivity Analysis (0 BPS to 50 BPS RT)"""
    cost_levels = [
        {"name": "0 BPS (Zero Fee)", "fee": 0.0000, "slip": 0.0000},
        {"name": "10 BPS (Taker Only)", "fee": 0.0005, "slip": 0.0000},
        {"name": "20 BPS (Fee + Minor Slip)", "fee": 0.0008, "slip": 0.0002},
        {"name": "30 BPS (Standard)", "fee": 0.0010, "slip": 0.0005},
        {"name": "50 BPS (High Friction)", "fee": 0.0015, "slip": 0.0010},
    ]

    cost_results = []
    for cl in cost_levels:
        res = run_strategy_backtest(
            strategy, df, warmup_bars=100,
            taker_fee_pct=cl["fee"], slippage_pct=cl["slip"]
        )
        res["cost_scenario"] = cl["name"]
        res["roundtrip_bps"] = int((cl["fee"] + cl["slip"]) * 2 * 10000)
        cost_results.append(res)

    return cost_results

def run_parameter_perturbation(
    strategy_cls: Type[BaseStrategy],
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    base_params: dict,
    param_name: str,
    param_values: list
) -> list[dict]:
    """Test 5: Parameter Perturbation & Stability Testing"""
    perturbation_results = []
    for val in param_values:
        params_copy = dict(base_params)
        params_copy[param_name] = val

        strat = strategy_cls(symbol, timeframe, params=params_copy)
        res = run_strategy_backtest(strat, df, warmup_bars=100)
        res["param_name"] = param_name
        res["param_value"] = val
        perturbation_results.append(res)

    return perturbation_results
