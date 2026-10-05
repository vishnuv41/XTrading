"""
strategy_lab/paper/forward_baseline_tracker.py
---------------------------------------------
Isolated forward paper-trading tracking engine for Phase 17.
Tracks 3 parallel arms with independent $10,000 USD virtual accounts:
1. Arm A: Trend_EMA_50 (Daily, 9 symbols, equal-weighted, 30 bps round-trip)
2. Arm B: Buy_and_Hold (Daily, 9 symbols, equal-weighted, 100% crypto)
3. Arm C: Static_Exposure_45pct (45% crypto / 55% cash, monthly rebalanced on 1st of month)

Features:
- Cryptographic SHA-256 Hash-Chained Ledger: Every row is linked to the previous row's hash.
- Dynamic Genesis Pin: Dynamically resolves git HEAD and hashes of all 5 strategy code files on genesis.
- Code Integrity Audit: On every run, recomputes hashes of the 5 files; if any differ, appends a "code_change" entry.
- Worktree Isolation: Commits and pushes exclusively within strategy_lab/paper/.worktree_ledger/ to origin/forward-ledger.
- Idempotent Catch-Up: Processes all missed closed calendar days chronologically.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_lab.lowturn.panel_loader import load_panel_universe, UNIVERSE_SYMBOLS
from strategy_lab.lowturn.cost_model import LowTurnoverCostModel, CANONICAL_ROUND_TRIP_BPS
from strategy_lab.lowturn.strategies import generate_single_ema_weights

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
STATE_FILE = os.path.join(STATE_DIR, "forward_track_state.json")
LEDGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_track_ledger.jsonl")
WORKTREE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worktree_ledger")

# Official Start Timestamp (first closed bar post-freeze)
OFFICIAL_START_TS = "2026-10-06T00:00:00+00:00"

TRACKED_CODE_FILES = [
    "strategy_lab/paper/forward_baseline_tracker.py",
    "strategy_lab/lowturn/strategies.py",
    "strategy_lab/lowturn/execution.py",
    "strategy_lab/lowturn/cost_model.py",
    "config/costs.py",
]

FROZEN_STRATEGY_PARAMETERS = {
    "protocol": "Phase 17 Trend_EMA_50 Forward Benchmark",
    "universe": UNIVERSE_SYMBOLS,
    "timeframe": "1d",
    "ema_span": 50,
    "round_trip_bps": CANONICAL_ROUND_TRIP_BPS,
    "arm_c_rebalance_rule": "monthly_1st",
    "arm_c_target_crypto_exposure": 0.45,
    "execution_convention": "t+1_open",
}


def get_current_git_head() -> str:
    """Dynamically get current git HEAD commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_HEAD"


def compute_file_sha256(rel_path: str) -> str:
    """Compute SHA-256 hash of a file relative to project root."""
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(full_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_all_tracked_hashes() -> Dict[str, str]:
    """Compute SHA-256 for all 5 core strategy code files."""
    return {f: compute_file_sha256(f) for f in TRACKED_CODE_FILES}


def get_parameters_sha256() -> str:
    canonical_str = json.dumps(FROZEN_STRATEGY_PARAMETERS, sort_keys=True)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def initialize_or_load_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    n_syms = len(UNIVERSE_SYMBOLS)
    current_hashes = compute_all_tracked_hashes()
    current_head = get_current_git_head()

    return {
        "official_start_date": OFFICIAL_START_TS,
        "frozen_code_commit": current_head,
        "tracked_file_hashes": current_hashes,
        "parameters_sha256": get_parameters_sha256(),
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


def get_last_ledger_hash() -> str:
    if not os.path.exists(LEDGER_FILE) or os.path.getsize(LEDGER_FILE) == 0:
        return "0" * 64

    last_line = ""
    with open(LEDGER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()

    if not last_line:
        return "0" * 64

    try:
        data = json.loads(last_line)
        return data.get("entry_hash", hashlib.sha256(last_line.encode("utf-8")).hexdigest())
    except Exception:
        return hashlib.sha256(last_line.encode("utf-8")).hexdigest()


def append_to_hash_chained_ledger(record: Dict[str, Any]):
    prev_hash = get_last_ledger_hash()
    record["prev_hash"] = prev_hash

    # If genesis block, pin the dynamically computed metadata
    if prev_hash == "0" * 64:
        record["genesis_spec"] = {
            "frozen_code_commit": get_current_git_head(),
            "tracked_file_hashes": compute_all_tracked_hashes(),
            "parameters_sha256": get_parameters_sha256(),
            "strategy_parameters": FROZEN_STRATEGY_PARAMETERS,
        }

    # Compute canonical hash of the record payload (excluding entry_hash)
    payload_str = json.dumps(record, sort_keys=True, default=str)
    entry_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    record["entry_hash"] = entry_hash

    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def check_and_record_code_changes(state: Dict[str, Any]):
    """
    Recompute hashes of all 5 core strategy files.
    If any file's hash differs from state, appends a 'code_change' entry to the ledger.
    """
    current_hashes = compute_all_tracked_hashes()
    stored_hashes = state.get("tracked_file_hashes", {})

    for file_path, curr_h in current_hashes.items():
        prev_h = stored_hashes.get(file_path)
        if prev_h is not None and curr_h != prev_h:
            print(f"WARNING: Code change detected in {file_path}! Appending code_change entry to ledger.")
            code_change_entry = {
                "entry_type": "code_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file": file_path,
                "old_sha256": prev_h,
                "new_sha256": curr_h,
            }
            append_to_hash_chained_ledger(code_change_entry)
            stored_hashes[file_path] = curr_h

    state["tracked_file_hashes"] = current_hashes


def compute_target_weights_for_bar(
    panel: Dict[str, pd.DataFrame],
    bar_idx: int,
) -> Dict[str, Dict[str, float]]:
    n_syms = len(UNIVERSE_SYMBOLS)
    
    # 1. Arm A: Trend_EMA_50
    ema50_targets = {}
    for s, df in panel.items():
        sub_df = df.iloc[: bar_idx + 1]
        w_series = generate_single_ema_weights(sub_df, span=50)
        ema50_targets[s] = float(w_series.iloc[-1]) / n_syms

    # 2. Arm B: Buy & Hold
    bh_targets = {s: 1.0 / n_syms for s in UNIVERSE_SYMBOLS}

    # 3. Arm C: Static 45% Exposure
    static_targets = {s: 0.45 / n_syms for s in UNIVERSE_SYMBOLS}

    return {
        "Trend_EMA_50": ema50_targets,
        "Buy_and_Hold": bh_targets,
        "Static_Exposure_45pct": static_targets,
    }


def push_ledger_via_isolated_worktree(last_ts_str: str):
    """
    Safely commits and pushes forward_track_ledger.jsonl from the isolated worktree
    without touching the developer's active working tree branch.
    """
    if not os.path.exists(WORKTREE_DIR):
        return

    try:
        dest_ledger = os.path.join(WORKTREE_DIR, "strategy_lab", "paper", "forward_track_ledger.jsonl")
        os.makedirs(os.path.dirname(dest_ledger), exist_ok=True)
        shutil.copy2(LEDGER_FILE, dest_ledger)

        # Commit and push strictly within the dedicated worktree directory
        subprocess.run(["git", "add", "strategy_lab/paper/forward_track_ledger.jsonl"], cwd=WORKTREE_DIR, check=False)
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"chore(tracker): forward ledger update {last_ts_str}"],
            cwd=WORKTREE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if "nothing to commit" not in commit_res.stdout.lower() and commit_res.returncode == 0:
            push_res = subprocess.run(["git", "push", "origin", "forward-ledger"], cwd=WORKTREE_DIR, capture_output=True, text=True, check=False)
            if push_res.returncode == 0:
                print("Forward track ledger safely committed & pushed via isolated worktree to origin/forward-ledger.")
            else:
                print(f"Note: worktree push to origin/forward-ledger deferred: {push_res.stderr.strip()}")
    except Exception as exc:
        print(f"Note: isolated worktree sync encountered: {exc}")


def step_daily_tracker(timeframe: str = "1d") -> Dict[str, Any]:
    """
    Executes daily tracker with catch-up, hash-chaining, code change detection, and isolated worktree pushing.
    """
    state = initialize_or_load_state()
    
    # Check for code changes against stored file hashes
    check_and_record_code_changes(state)

    panel = load_panel_universe(timeframe=timeframe)
    cost_model = LowTurnoverCostModel(round_trip_bps=state["round_trip_bps"])

    sample_df = next(iter(panel.values()))
    all_timestamps = sample_df.index

    # Find the starting index for unprocessed bars on or after OFFICIAL_START_TS
    official_start_dt = pd.to_datetime(OFFICIAL_START_TS, utc=True)
    last_updated_str = state["last_updated"]

    if last_updated_str is not None:
        last_dt = pd.to_datetime(last_updated_str, utc=True)
        unprocessed_mask = (all_timestamps > last_dt) & (all_timestamps >= official_start_dt)
    else:
        unprocessed_mask = all_timestamps >= official_start_dt

    unprocessed_indices = np.where(unprocessed_mask)[0]

    if len(unprocessed_indices) == 0:
        latest_avail = sample_df.index[-1].isoformat()
        print(f"Forward tracker: up to date. Latest available bar is {latest_avail}. Waiting for next daily close >= {OFFICIAL_START_TS}.")
        save_state(state)
        return state

    print(f"Forward tracker: processing {len(unprocessed_indices)} pending closed bar(s)...")

    for bar_idx in unprocessed_indices:
        current_ts = all_timestamps[bar_idx]
        ts_str = current_ts.to_pydatetime().isoformat()
        is_first_of_month = (current_ts.day == 1)

        # Asset returns on this bar relative to prior bar
        if bar_idx == 0:
            asset_rets = {s: 0.0 for s in UNIVERSE_SYMBOLS}
        else:
            asset_rets = {
                s: float((panel[s]["close"].iloc[bar_idx] / panel[s]["close"].iloc[bar_idx - 1]) - 1.0)
                for s in UNIVERSE_SYMBOLS
            }

        arm_targets = compute_target_weights_for_bar(panel, bar_idx)
        ledger_entry = {"entry_type": "daily_bar", "timestamp": ts_str, "arms": {}}

        for arm_name, arm_data in state["arms"].items():
            prev_weights = arm_data["current_weights"]
            target_weights = arm_targets[arm_name]

            # Rebalancing logic for Arm C: only rebalance on 1st of month, else drift
            if arm_name == "Static_Exposure_45pct" and not is_first_of_month and state["last_updated"] is not None:
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
        append_to_hash_chained_ledger(ledger_entry)

    save_state(state)
    print(f"Forward tracker updated to: {state['last_updated']}.")
    push_ledger_via_isolated_worktree(state["last_updated"])

    return state


if __name__ == "__main__":
    step_daily_tracker()
