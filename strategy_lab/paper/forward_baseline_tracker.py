"""
strategy_lab/paper/forward_baseline_tracker.py
---------------------------------------------
Isolated forward paper-trading tracking engine for Phase 17.
Tracks 3 parallel arms with independent $10,000 USD virtual accounts:
1. Arm A: Trend_EMA_50 (Daily, 9 symbols, equal-weighted)
2. Arm B: Buy_and_Hold (Daily, 9 symbols, equal-weighted)
3. Arm C: Static_Exposure_45pct (45% crypto / 55% cash)

Completely isolated from Phase 13 state.
Persistence stored in strategy_lab/paper/state/forward_track_state.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.connection import get_engine
from strategy_lab.lowturn.panel_loader import load_panel_universe, UNIVERSE_SYMBOLS
from strategy_lab.lowturn.cost_model import LowTurnoverCostModel, CANONICAL_ROUND_TRIP_BPS
from strategy_lab.lowturn.strategies import generate_single_ema_weights, generate_buy_and_hold_weights

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
STATE_FILE = os.path.join(STATE_DIR, "forward_track_state.json")


def initialize_or_load_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # Initialize fresh state starting at $10,000 USD each
    return {
        "start_date": datetime.now(timezone.utc).isoformat(),
        "last_updated": None,
        "round_trip_bps": CANONICAL_ROUND_TRIP_BPS,
        "symbols": UNIVERSE_SYMBOLS,
        "arms": {
            "Trend_EMA_50": {
                "initial_cash": 10000.0,
                "current_equity": 10000.0,
                "peak_equity": 10000.0,
                "max_drawdown": 0.0,
                "total_turnover": 0.0,
                "total_friction_usd": 0.0,
                "current_weights": {s: 0.0 for s in UNIVERSE_SYMBOLS},
                "history": [],
            },
            "Buy_and_Hold": {
                "initial_cash": 10000.0,
                "current_equity": 10000.0,
                "peak_equity": 10000.0,
                "max_drawdown": 0.0,
                "total_turnover": 0.0,
                "total_friction_usd": 0.0,
                "current_weights": {s: 1.0 / len(UNIVERSE_SYMBOLS) for s in UNIVERSE_SYMBOLS},
                "history": [],
            },
            "Static_Exposure_45pct": {
                "initial_cash": 10000.0,
                "current_equity": 10000.0,
                "peak_equity": 10000.0,
                "max_drawdown": 0.0,
                "total_turnover": 0.0,
                "total_friction_usd": 0.0,
                "current_weights": {s: 0.45 / len(UNIVERSE_SYMBOLS) for s in UNIVERSE_SYMBOLS},
                "history": [],
            },
        },
    }


def save_state(state: Dict[str, Any]):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def step_daily_tracker(timeframe: str = "1d") -> Dict[str, Any]:
    """
    Executes a single daily update step across the 9-asset panel.
    """
    state = initialize_or_load_state()
    panel = load_panel_universe(timeframe=timeframe)
    cost_model = LowTurnoverCostModel(round_trip_bps=state["round_trip_bps"])

    # Determine latest closed bar timestamp
    latest_ts = next(iter(panel.values())).index[-1].to_pydatetime()
    ts_str = latest_ts.isoformat()

    # Prevent double-processing the same bar
    if state["last_updated"] == ts_str:
        print(f"Bar {ts_str} already processed in forward tracker.")
        return state

    n_syms = len(UNIVERSE_SYMBOLS)

    # 1. Compute today's target weights for each arm
    # Arm A: Trend_EMA_50
    ema50_targets = {}
    for s, df in panel.items():
        w_series = generate_single_ema_weights(df, span=50)
        ema50_targets[s] = float(w_series.iloc[-1]) / n_syms

    # Arm B: Buy & Hold
    bh_targets = {s: 1.0 / n_syms for s in UNIVERSE_SYMBOLS}

    # Arm C: Static 45% Exposure
    static_targets = {s: 0.45 / n_syms for s in UNIVERSE_SYMBOLS}

    arm_targets = {
        "Trend_EMA_50": ema50_targets,
        "Buy_and_Hold": bh_targets,
        "Static_Exposure_45pct": static_targets,
    }

    # 2. Update each arm's daily equity, turnover, friction
    for arm_name, arm_data in state["arms"].items():
        prev_weights = arm_data["current_weights"]
        curr_targets = arm_targets[arm_name]

        # Turnover = sum(|w_t - w_{t-1}|)
        turnover = sum(abs(curr_targets[s] - prev_weights[s]) for s in UNIVERSE_SYMBOLS)
        friction_pct = cost_model.apply_turnover_friction(turnover)

        # Asset returns over current closed bar
        asset_rets = {s: float(panel[s]["close"].pct_change().iloc[-1]) for s in UNIVERSE_SYMBOLS}
        gross_ret = sum(prev_weights[s] * asset_rets[s] for s in UNIVERSE_SYMBOLS)
        net_ret = gross_ret - friction_pct

        # Update equity
        prev_equity = arm_data["current_equity"]
        curr_equity = prev_equity * (1.0 + net_ret)
        friction_usd = prev_equity * friction_pct

        arm_data["current_equity"] = curr_equity
        arm_data["total_turnover"] += turnover
        arm_data["total_friction_usd"] += friction_usd
        arm_data["peak_equity"] = max(arm_data["peak_equity"], curr_equity)
        dd = (curr_equity - arm_data["peak_equity"]) / arm_data["peak_equity"]
        arm_data["max_drawdown"] = min(arm_data["max_drawdown"], dd)
        arm_data["current_weights"] = curr_targets

        arm_data["history"].append({
            "timestamp": ts_str,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "friction_usd": friction_usd,
            "turnover": turnover,
            "equity": curr_equity,
            "drawdown": dd,
        })

    state["last_updated"] = ts_str
    save_state(state)
    print(f"Forward tracker updated successfully for bar {ts_str}.")
    for name, arm in state["arms"].items():
        print(f"  [{name}] Equity: ${arm['current_equity']:.2f} | MaxDD: {arm['max_drawdown']*100:.2f}% | Turnover: {arm['total_turnover']:.2f}")

    return state


if __name__ == "__main__":
    step_daily_tracker()
