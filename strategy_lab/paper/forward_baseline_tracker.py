"""
strategy_lab/paper/forward_baseline_tracker.py
---------------------------------------------
Isolated forward paper-trading tracking engine for Phase 17.
Tracks 3 parallel arms with independent $10,000 USD virtual accounts:
1. Arm A: Trend_EMA_50 (Daily, 9 symbols, equal-weighted, 30 bps round-trip)
2. Arm B: Buy_and_Hold (Daily, 9 symbols, equal-weighted, 100% crypto)
3. Arm C: Static_Exposure_45pct (45% crypto / 55% cash, monthly rebalanced on 1st of month)

Features:
- Idempotent Catch-Up: Processes all missed closed calendar days chronologically.
- Append-Only Log: Records daily bar transitions to strategy_lab/paper/forward_track_ledger.jsonl.
- Complete Segregation: No dependencies on Phase 13 single-position engine or database state.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.lowturn.panel_loader import load_panel_universe, UNIVERSE_SYMBOLS
from strategy_lab.lowturn.cost_model import LowTurnoverCostModel, CANONICAL_ROUND_TRIP_BPS
from strategy_lab.lowturn.strategies import generate_single_ema_weights

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
STATE_FILE = os.path.join(STATE_DIR, "forward_track_state.json")
LEDGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_track_ledger.jsonl")


def initialize_or_load_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    n_syms = len(UNIVERSE_SYMBOLS)
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
            },
            "Buy_and_Hold": {
                "initial_cash": 10000.0,
                "current_equity": 10000.0,
                "peak_equity": 10000.0,
                "max_drawdown": 0.0,
                "total_turnover": 0.0,
                "total_friction_usd": 0.0,
                "current_weights": {s: 1.0 / n_syms for s in UNIVERSE_SYMBOLS},
            },
            "Static_Exposure_45pct": {
                "initial_cash": 10000.0,
                "current_equity": 10000.0,
                "peak_equity": 10000.0,
                "max_drawdown": 0.0,
                "total_turnover": 0.0,
                "total_friction_usd": 0.0,
                "current_weights": {s: 0.45 / n_syms for s in UNIVERSE_SYMBOLS},
            },
        },
    }


def save_state(state: Dict[str, Any]):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def append_to_ledger(record: Dict[str, Any]):
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def compute_target_weights_for_bar(
    panel: Dict[str, pd.DataFrame],
    bar_idx: int,
    ts: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    n_syms = len(UNIVERSE_SYMBOLS)
    
    # 1. Arm A: Trend_EMA_50
    ema50_targets = {}
    for s, df in panel.items():
        sub_df = df.iloc[: bar_idx + 1]
        w_series = generate_single_ema_weights(sub_df, span=50)
        ema50_targets[s] = (float(w_series.iloc[-1]) / n_syms)

    # 2. Arm B: Buy & Hold (always 1/N per asset)
    bh_targets = {s: 1.0 / n_syms for s in UNIVERSE_SYMBOLS}

    # 3. Arm C: Static 45% Exposure (rebalance on 1st day of month, else maintain weight)
    static_targets = {s: 0.45 / n_syms for s in UNIVERSE_SYMBOLS}

    return {
        "Trend_EMA_50": ema50_targets,
        "Buy_and_Hold": bh_targets,
        "Static_Exposure_45pct": static_targets,
    }


def step_daily_tracker(timeframe: str = "1d") -> Dict[str, Any]:
    """
    Executes daily tracker with catch-up: identifies all unprocessed closed bars
    and advances state sequentially.
    """
    state = initialize_or_load_state()
    panel = load_panel_universe(timeframe=timeframe)
    cost_model = LowTurnoverCostModel(round_trip_bps=state["round_trip_bps"])

    sample_df = next(iter(panel.values()))
    all_timestamps = sample_df.index

    # Find the starting index for unprocessed bars
    last_updated_str = state["last_updated"]
    if last_updated_str is not None:
        last_dt = pd.to_datetime(last_updated_str, utc=True)
        unprocessed_mask = all_timestamps > last_dt
    else:
        # First initialization: process the last closed bar
        unprocessed_mask = np.zeros(len(all_timestamps), dtype=bool)
        unprocessed_mask[-1] = True

    unprocessed_indices = np.where(unprocessed_mask)[0]

    if len(unprocessed_indices) == 0:
        print(f"Forward tracker: all bars up to {sample_df.index[-1].isoformat()} are already up to date.")
        return state

    print(f"Forward tracker: processing {len(unprocessed_indices)} pending closed bar(s)...")

    for bar_idx in unprocessed_indices:
        current_ts = all_timestamps[bar_idx]
        ts_str = current_ts.to_pydatetime().isoformat()
        is_first_of_month = (current_ts.day == 1)

        # Asset simple returns on this bar relative to prior bar
        if bar_idx == 0:
            asset_rets = {s: 0.0 for s in UNIVERSE_SYMBOLS}
        else:
            asset_rets = {
                s: float((panel[s]["close"].iloc[bar_idx] / panel[s]["close"].iloc[bar_idx - 1]) - 1.0)
                for s in UNIVERSE_SYMBOLS
            }

        arm_targets = compute_target_weights_for_bar(panel, bar_idx, current_ts)

        ledger_entry = {"timestamp": ts_str, "arms": {}}

        for arm_name, arm_data in state["arms"].items():
            prev_weights = arm_data["current_weights"]
            target_weights = arm_targets[arm_name]

            # Rebalancing logic for Arm C: only rebalance on 1st of month, else drift
            if arm_name == "Static_Exposure_45pct" and not is_first_of_month and state["last_updated"] is not None:
                # Drift with price
                active_weights = prev_weights
                turnover = 0.0
                friction_pct = 0.0
            else:
                active_weights = prev_weights
                turnover = sum(abs(target_weights[s] - prev_weights[s]) for s in UNIVERSE_SYMBOLS)
                friction_pct = cost_model.apply_turnover_friction(turnover)

            gross_ret = sum(active_weights[s] * asset_rets[s] for s in UNIVERSE_SYMBOLS)
            net_ret = gross_ret - friction_pct

            prev_equity = arm_data["current_equity"]
            curr_equity = prev_equity * (1.0 + net_ret)
            friction_usd = prev_equity * friction_pct

            arm_data["current_equity"] = curr_equity
            arm_data["total_turnover"] += turnover
            arm_data["total_friction_usd"] += friction_usd
            arm_data["peak_equity"] = max(arm_data["peak_equity"], curr_equity)
            dd = (curr_equity - arm_data["peak_equity"]) / arm_data["peak_equity"]
            arm_data["max_drawdown"] = min(arm_data["max_drawdown"], dd)
            arm_data["current_weights"] = target_weights

            ledger_entry["arms"][arm_name] = {
                "gross_return": gross_ret,
                "net_return": net_ret,
                "friction_usd": friction_usd,
                "turnover": turnover,
                "equity": curr_equity,
                "drawdown": dd,
            }

        state["last_updated"] = ts_str
        append_to_ledger(ledger_entry)

    save_state(state)
    print(f"Forward tracker updated to: {state['last_updated']}.")
    for name, arm in state["arms"].items():
        print(f"  [{name}] Equity: ${arm['current_equity']:.2f} | MaxDD: {arm['max_drawdown']*100:.2f}% | Turnover: {arm['total_turnover']:.2f}")

    return state


if __name__ == "__main__":
    step_daily_tracker()
