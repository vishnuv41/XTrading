"""
walkforward_pooled.py
------------------------
Pooled-data, simplified-filter walk-forward validator.

Why this instead of "more architecture" (a V7-style two-stage
meta-labeling system, which was the other option on the table):

V6's OOS folds on SOL/USDT 1h alone produced 3-13 trades per fold — far
too few to distinguish a real edge from noise, let alone support a
second meta-model trained on an even smaller held-out slice of that.
Two changes attack the sample-size problem directly, without adding
any new model complexity:

  1. POOL training data across symbols (BTC/USDT, ETH/USDT, SOL/USDT by
     default) into one directional model, instead of training a
     separate model per symbol on ~30 trades each. More rows per fold,
     and an "edge" that only shows up on one symbol is much more
     likely to be noise than one that holds up pooled across three.

  2. SIMPLIFY the entry filter stack. V6 stacked six gates
     (probability threshold, a regime-adaptive version of that same
     threshold, an expected-value floor, a hand-tuned score, a
     vol-percentile gate, and a momentum/RSI-band gate) — six
     independent knobs, each one another way to overfit thirty trades.
     This keeps exactly two: a calibrated probability threshold
     (frozen from inner training-only validation, optionally
     auto-selected there too — never on OOS) and a net-of-cost
     expected-value floor. Nothing else.

Everything that made V6 methodologically sound is kept unchanged:
purged expanding walk-forward, an inner chronological calibration
split inside every training fold, Platt calibration, an XGBoost +
CatBoost blend, and no OOS-based tuning of anything.

This is a diagnostic step, not a final trading system: the question it
answers is "does pooling + simplifying reveal a stable edge that
single-symbol V6 couldn't see" — not "ship this."

Usage:
    python walkforward_pooled.py --symbols BTC/USDT ETH/USDT SOL/USDT --timeframe 1h
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, precision_score, recall_score,
)
from sklearn.feature_selection import mutual_info_classif

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pipeline.data_loader import load_ohlcv


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Trade:
    fold: int
    symbol: str
    entry_index: int
    exit_index: int
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    atr: float
    raw_probability: float
    calibrated_probability: float
    expected_value: float
    position_fraction: float
    gross_return: float
    transaction_cost: float
    net_return: float
    exit_reason: str


# ============================================================================
# Data loading (per symbol, restricted to the common overlapping range)
# ============================================================================

def finite_float(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def load_symbol(symbol: str, timeframe: str, exchange: str) -> pd.DataFrame:
    """Load + clean one symbol's OHLCV via the P1 DB bridge (pipeline.data_loader)."""
    df = load_ohlcv(symbol, timeframe, exchange=exchange)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(f"{symbol}: invalid timestamp values found.")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df.dropna(subset=["open", "high", "low", "close"])
            .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))
    return df


def restrict_to_common_range(dfs: dict) -> dict:
    """
    Trim every symbol's DataFrame to the overlapping [start, end] window
    across all symbols — the pooling step is only valid where every
    symbol actually has data (a newer listing like SOL/USDT constrains
    how far back the pool can go).
    """
    start = max(df["timestamp"].iloc[0] for df in dfs.values())
    end = min(df["timestamp"].iloc[-1] for df in dfs.values())
    if start >= end:
        raise ValueError("No overlapping time range across symbols.")

    trimmed = {}
    for symbol, df in dfs.items():
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        trimmed[symbol] = df.loc[mask].reset_index(drop=True)
        if len(trimmed[symbol]) < 500:
            raise ValueError(f"{symbol}: only {len(trimmed[symbol])} rows in the common range — too few.")
    return trimmed


# ============================================================================
# Causal features (single-timeframe only — no MTF, keeps this dependency-free)
# ============================================================================

def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low, open_, volume = x["close"], x["high"], x["low"], x["open"], x["volume"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    for n in [8, 13, 20, 34, 50, 100, 200]:
        x[f"EMA{n}"] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    x["EMA20_dist_ATR"] = (close - x["EMA20"]) / tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().replace(0, np.nan)

    for n in [1, 3, 6, 12, 24, 48, 72, 120, 168]:
        x[f"RET{n}"] = close.pct_change(n)

    for n in [7, 14, 21, 50]:
        x[f"ATR{n}"] = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        x[f"ATR{n}_pct"] = x[f"ATR{n}"] / close.replace(0, np.nan)
    x["ATR_percentile"] = x["ATR14_pct"].rolling(240, min_periods=50).rank(pct=True) * 100.0

    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = (100 - 100/(1+rs)).where(avg_loss != 0, 100.0)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_signal"] = x["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    x["MACD_hist"] = x["MACD"] - x["MACD_signal"]

    bb_mid, bb_std = close.rolling(20, min_periods=20).mean(), close.rolling(20, min_periods=20).std()
    x["BB_width"] = (4 * bb_std) / bb_mid.replace(0, np.nan)
    x["BB_position"] = (close - (bb_mid - 2*bb_std)) / (4 * bb_std).replace(0, np.nan)

    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = x["ATR14"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["ADX14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    candle_range = (high - low).replace(0, np.nan)
    x["BODY_RANGE"] = (close - open_).abs() / candle_range

    vmean, vstd = volume.rolling(20, min_periods=20).mean(), volume.rolling(20, min_periods=20).std()
    x["REL_VOLUME"] = volume / vmean.replace(0, np.nan)
    x["VOLUME_Z"] = (volume - vmean) / vstd.replace(0, np.nan)

    for n in [12, 24, 48, 96]:
        x[f"REALIZED_VOL_{n}"] = np.log(close / close.shift(1)).rolling(n, min_periods=n).std()

    ts = pd.to_datetime(x["timestamp"], utc=True)
    hour, dow = ts.dt.hour.astype(float), ts.dt.dayofweek.astype(float)
    x["HOUR_SIN"], x["HOUR_COS"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    x["DOW_SIN"], x["DOW_COS"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)

    return x


def ensure_numeric(df: pd.DataFrame) -> tuple:
    out = pd.DataFrame(index=df.index)
    dropped = []
    for c in df.columns:
        if str(c).lower() in {"timestamp", "datetime", "label", "symbol"}:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            converted = pd.to_numeric(s, errors="coerce")
            if converted.notna().mean() >= 0.95:
                out[c] = converted
            else:
                dropped.append(str(c))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.loc[:, ~out.columns.duplicated()], dropped


# ============================================================================
# TP-before-SL labels (local implementation — no external labeler dependency)
# ============================================================================

def build_binary_labels(raw: pd.DataFrame, features: pd.DataFrame, side: str,
                         sl_mult: float, risk_reward: float, max_holding: int) -> pd.Series:
    n = len(raw)
    y = np.full(n, np.nan, dtype=float)
    opens, highs, lows = raw["open"].to_numpy(float), raw["high"].to_numpy(float), raw["low"].to_numpy(float)
    atrs = features["ATR14"].to_numpy(float)

    for i in range(n):
        entry_i = i + 1
        if entry_i >= n or i + max_holding >= n:
            continue
        a, entry = atrs[i], opens[entry_i]
        if not (math.isfinite(a) and a > 0 and math.isfinite(entry) and entry > 0):
            continue

        if side == "long":
            sl, tp = entry - sl_mult * a, entry + risk_reward * sl_mult * a
        else:
            sl, tp = entry + sl_mult * a, entry - risk_reward * sl_mult * a

        end_i = min(n - 1, i + max_holding)
        for j in range(entry_i, end_i + 1):
            hi, lo = highs[j], lows[j]
            if side == "long":
                if lo <= sl:
                    y[i] = 0.0; break
                if hi >= tp:
                    y[i] = 1.0; break
            else:
                if hi >= sl:
                    y[i] = 0.0; break
                if lo <= tp:
                    y[i] = 1.0; break
        # unresolved timeout stays NaN -> dropped from training, same as V6
    return pd.Series(y)


# ============================================================================
# Feature selection / transform (unchanged from V6 — training-only, per fold)
# ============================================================================

def select_features_train_only(X_train: pd.DataFrame, y_train: pd.Series, max_features: int, seed: int) -> list:
    valid_columns = [c for c in X_train.columns
                      if pd.to_numeric(X_train[c], errors="coerce").notna().sum() >= 100
                      and pd.to_numeric(X_train[c], errors="coerce").nunique(dropna=True) >= 2]
    if not valid_columns:
        raise RuntimeError("No usable numeric features.")

    X = X_train[valid_columns].copy()
    med = X.median().fillna(0)
    X = X.fillna(med)
    X_for_mi = X.clip(-1e6, 1e6).astype(np.float32)

    try:
        mi = mutual_info_classif(X_for_mi, y_train.astype(int), discrete_features=False, random_state=seed)
        mi_series = pd.Series(mi, index=X.columns)
    except Exception:
        mi_series = pd.Series(0.0, index=X.columns)

    pearson = {}
    yv = y_train.to_numpy(float)
    for c in X.columns:
        xv = X[c].to_numpy(float)
        if np.std(xv) <= 1e-12:
            pearson[c] = 0.0
            continue
        corr = np.corrcoef(xv, yv)[0, 1]
        pearson[c] = abs(float(corr)) if math.isfinite(corr) else 0.0

    score = (mi_series.rank(pct=True) + pd.Series(pearson).rank(pct=True)) / 2.0
    return score.sort_values(ascending=False).head(max_features).index.tolist()


def fit_medians(X_train: pd.DataFrame, columns: list) -> pd.Series:
    return X_train[columns].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)


def transform_matrix(X: pd.DataFrame, columns: list, medians: pd.Series) -> pd.DataFrame:
    out = X.reindex(columns=columns).copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(medians.reindex(columns).fillna(0.0))
    arr = out.to_numpy(dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame(arr, index=out.index, columns=columns)


# ============================================================================
# Models (unchanged from V6 — XGBoost + CatBoost blend)
# ============================================================================

def train_base_models(X_train: pd.DataFrame, y_train: pd.Series, seed: int):
    positive, negative = max(1, int((y_train == 1).sum())), max(1, int((y_train == 0).sum()))
    scale = float(np.clip(np.sqrt(negative / positive), 1.0, 2.0))

    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.025, min_child_weight=8,
        subsample=0.85, colsample_bytree=0.80, gamma=0.10, reg_alpha=0.15, reg_lambda=4.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=max(1, (os.cpu_count() or 4) - 1), scale_pos_weight=scale,
    )
    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.025, loss_function="Logloss", eval_metric="Logloss",
        random_seed=seed, verbose=False, allow_writing_files=False,
        l2_leaf_reg=8.0, random_strength=0.5, border_count=128,
        thread_count=max(1, (os.cpu_count() or 4) - 1),
    )
    xgb.fit(X_train, y_train.astype(int))
    cat.fit(X_train, y_train.astype(int))
    return xgb, cat


def raw_blend_probability(xgb, cat, X: pd.DataFrame, xgb_weight: float, cat_weight: float) -> np.ndarray:
    total = xgb_weight + cat_weight
    wx, wc = xgb_weight / total, cat_weight / total
    px = np.asarray(xgb.predict_proba(X)[:, 1], dtype=float)
    pc = np.asarray(cat.predict_proba(X)[:, 1], dtype=float)
    p = wx * px + wc * pc
    return np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def fit_probability_calibrator(raw_probability: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(raw_probability, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    calibrator.fit(logit, y)
    return calibrator


def apply_probability_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    if calibrator is None:
        return p
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    calibrated = calibrator.predict_proba(logit)[:, 1]
    return np.clip(np.nan_to_num(calibrated, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


# ============================================================================
# SIMPLIFIED entry decision — exactly 2 gates (was 6 in V6)
# ============================================================================

def expected_value_units(p: float, rr: float, transaction_cost_bps: float) -> float:
    """EV in units of initial risk: EV = p*RR - (1-p) - round_trip_cost/risk_unit."""
    p = np.clip(float(p), 0.0, 1.0)
    cost = 2.0 * transaction_cost_bps / 10000.0
    return float(p * rr - (1.0 - p) - cost)


def entry_decision(p_cal: float, threshold: float, rr: float, cost_bps: float, min_ev: float) -> tuple:
    """Gate 1: calibrated probability >= frozen threshold. Gate 2: net EV >= floor. Nothing else."""
    if not math.isfinite(p_cal):
        return False, np.nan
    ev = expected_value_units(p_cal, rr, cost_bps)
    allowed = (p_cal >= threshold) and (ev >= min_ev)
    return allowed, ev


def select_threshold_inner(p_inner: np.ndarray, y_inner: np.ndarray, rr: float, cost_bps: float,
                            min_ev: float, min_trades: int, candidates=None) -> float:
    """
    Choose the probability threshold using ONLY the inner calibration
    slice (never OOS) — maximize mean net EV per trade among candidates
    that clear a minimum trade-count floor, so the choice isn't just
    "the threshold with one lucky trade".
    """
    if candidates is None:
        candidates = np.arange(0.50, 0.86, 0.02)

    best_threshold, best_score = candidates[0], -np.inf
    for t in candidates:
        taken = p_inner >= t
        if taken.sum() < min_trades:
            continue
        evs = np.array([expected_value_units(p, rr, cost_bps) for p in p_inner[taken]])
        realized = np.where(y_inner[taken] == 1, rr, -1.0) - (2.0 * cost_bps / 10000.0)
        mean_realized = realized.mean()
        if evs.mean() < min_ev:
            continue
        if mean_realized > best_score:
            best_score, best_threshold = mean_realized, float(t)

    return best_threshold


# ============================================================================
# Trade engine
# ============================================================================

def execute_trade(raw: pd.DataFrame, symbol: str, signal_index: int, side: str, atr_value: float,
                   sl_mult: float, rr: float, max_holding: int, cost_bps: float,
                   p_raw: float, p_cal: float, ev: float, fold: int) -> Trade:
    entry_index = signal_index + 1
    if entry_index >= len(raw):
        raise ValueError("No next-bar entry available.")
    entry_price = finite_float(raw.iloc[entry_index]["open"])
    atr_value = finite_float(atr_value)
    if not (math.isfinite(entry_price) and entry_price > 0 and math.isfinite(atr_value) and atr_value > 0):
        raise ValueError("Invalid entry price or ATR.")

    if side == "long":
        stop_price, target_price = entry_price - sl_mult * atr_value, entry_price + rr * sl_mult * atr_value
    else:
        stop_price, target_price = entry_price + sl_mult * atr_value, entry_price - rr * sl_mult * atr_value

    last_index = min(len(raw) - 1, entry_index + max_holding)
    exit_index, exit_price, exit_reason = last_index, finite_float(raw.iloc[last_index]["close"], entry_price), "TIMEOUT"

    for j in range(entry_index, last_index + 1):
        hi, lo = finite_float(raw.iloc[j]["high"]), finite_float(raw.iloc[j]["low"])
        if not (math.isfinite(hi) and math.isfinite(lo)):
            continue
        if side == "long":
            if lo <= stop_price:
                exit_index, exit_price, exit_reason = j, stop_price, "SL"; break
            if hi >= target_price:
                exit_index, exit_price, exit_reason = j, target_price, "TP"; break
        else:
            if hi >= stop_price:
                exit_index, exit_price, exit_reason = j, stop_price, "SL"; break
            if lo <= target_price:
                exit_index, exit_price, exit_reason = j, target_price, "TP"; break

    gross_return = (exit_price/entry_price - 1.0) if side == "long" else (entry_price/exit_price - 1.0)
    round_trip_cost = 2.0 * cost_bps / 10000.0

    return Trade(
        fold=fold, symbol=symbol, entry_index=int(entry_index), exit_index=int(exit_index),
        entry_time=str(raw.iloc[entry_index]["timestamp"]), exit_time=str(raw.iloc[exit_index]["timestamp"]),
        side=side, entry_price=float(entry_price), exit_price=float(exit_price),
        stop_price=float(stop_price), target_price=float(target_price), atr=float(atr_value),
        raw_probability=float(p_raw) if math.isfinite(p_raw) else np.nan,
        calibrated_probability=float(p_cal), expected_value=float(ev), position_fraction=1.0,
        gross_return=float(gross_return), transaction_cost=float(round_trip_cost),
        net_return=float(gross_return - round_trip_cost), exit_reason=exit_reason,
    )


def run_trade_engine(raw: pd.DataFrame, symbol: str, features: pd.DataFrame, positions: np.ndarray,
                      p_cal: np.ndarray, p_raw: np.ndarray, threshold: float, args, fold: int) -> tuple:
    trades = []
    diag = {"signals_seen": 0, "accepted": 0, "rejected_probability": 0, "rejected_ev": 0,
            "cooldown": 0, "invalid_atr": 0, "execution_error": 0}
    cooldown_until = -1

    for local_i, raw_index in enumerate(positions):
        raw_index = int(raw_index)
        if raw_index < 0 or raw_index >= len(raw):
            continue
        diag["signals_seen"] += 1

        pc, pr = finite_float(p_cal[local_i]), finite_float(p_raw[local_i])
        if not math.isfinite(pc):
            continue

        allowed, ev = entry_decision(pc, threshold, args.risk_reward, args.transaction_cost_bps, args.min_expected_value)
        if not allowed:
            if pc < threshold:
                diag["rejected_probability"] += 1
            else:
                diag["rejected_ev"] += 1
            continue

        if raw_index <= cooldown_until:
            diag["cooldown"] += 1
            continue

        atr_value = finite_float(features.iloc[local_i].get("ATR14", np.nan))
        if not (math.isfinite(atr_value) and atr_value > 0):
            diag["invalid_atr"] += 1
            continue

        try:
            trade = execute_trade(raw, symbol, raw_index, args.side, atr_value, args.sl_mult,
                                   args.risk_reward, args.max_holding, args.transaction_cost_bps,
                                   pr, pc, ev, fold)
            trades.append(trade)
            diag["accepted"] += 1
            cooldown_until = trade.exit_index + args.cooldown_bars
        except Exception:
            diag["execution_error"] += 1

    return trades, diag


# ============================================================================
# Metrics
# ============================================================================

def trade_metrics(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "total_return": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "avg_trade": 0.0, "transaction_cost": 0.0,
                "tp_first": 0, "sl_first": 0, "timeout": 0}

    returns = np.nan_to_num(np.array([t.net_return for t in trades], dtype=float), nan=0.0)
    equity = np.cumprod(1.0 + returns)
    dd = equity / np.maximum(np.maximum.accumulate(equity), 1e-12) - 1.0
    gains, losses = returns[returns > 0], returns[returns < 0]
    gp, gl = float(gains.sum()) if len(gains) else 0.0, float(-losses.sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    sharpe = (np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(len(returns))
              if len(returns) > 1 and np.std(returns, ddof=1) > 0 else 0.0)

    return {
        "trades": int(len(trades)), "total_return": float(equity[-1] - 1.0),
        "win_rate": float((returns > 0).mean()), "profit_factor": float(pf), "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()), "avg_trade": float(returns.mean()),
        "transaction_cost": float(sum(t.transaction_cost for t in trades)),
        "tp_first": int(sum(t.exit_reason == "TP" for t in trades)),
        "sl_first": int(sum(t.exit_reason == "SL" for t in trades)),
        "timeout": int(sum(t.exit_reason == "TIMEOUT" for t in trades)),
    }


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        return {}
    pred = (p >= 0.5).astype(int)
    return {
        "samples": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


# ============================================================================
# Calendar-based expanding folds (shared timestamp grid across symbols)
# ============================================================================

def make_calendar_folds(reference_timestamps: pd.Series, n_folds: int, min_train_bars: int, test_bars: int) -> list:
    """Same expanding-window logic as V6's make_folds, but returns (train_start_ts, train_end_ts,
    test_start_ts, test_end_ts) instead of row positions, so each symbol can be sliced by its own
    timestamp column — row counts don't need to match exactly across symbols."""
    n_rows = len(reference_timestamps)
    folds = []
    for i in range(n_folds):
        train_end_pos = min_train_bars + i * test_bars
        test_start_pos = train_end_pos
        test_end_pos = min(test_start_pos + test_bars, n_rows)
        if test_start_pos >= n_rows or test_end_pos <= test_start_pos:
            break
        folds.append((
            reference_timestamps.iloc[0],
            reference_timestamps.iloc[train_end_pos - 1],
            reference_timestamps.iloc[test_start_pos],
            reference_timestamps.iloc[test_end_pos - 1],
        ))
    return folds


# ============================================================================
# Per-fold run (pooled across symbols)
# ============================================================================

def run_fold(fold_no: int, symbol_data: dict, train_start_ts, train_end_ts, test_start_ts, test_end_ts, args):
    print(f"\n{'='*78}\nPOOLED FOLD {fold_no}\n{'='*78}")
    print(f"TRAIN: {train_start_ts} -> {train_end_ts}")
    print(f"TEST : {test_start_ts} -> {test_end_ts}")

    purge = max(1, args.max_holding)
    model_frames, cal_frames = [], []

    for symbol, d in symbol_data.items():
        raw, features, numeric, labels = d["raw"], d["features"], d["numeric"], d["labels"]
        ts = raw["timestamp"]

        train_mask = (ts >= train_start_ts) & (ts <= train_end_ts)
        train_idx = np.where(train_mask)[0]
        if len(train_idx) <= purge:
            continue
        train_idx = train_idx[:-purge]  # purge: drop rows whose label horizon could cross into test

        y_train_all = labels.iloc[train_idx].to_numpy(float)
        valid = np.isfinite(y_train_all) & np.isin(y_train_all, [0.0, 1.0])
        train_idx = train_idx[valid]
        if len(train_idx) < 200:
            continue

        cal_size = max(args.min_calibration_bars, int(len(train_idx) * args.calibration_fraction))
        cal_size = min(cal_size, len(train_idx) - 100)
        if cal_size <= 0:
            continue
        model_end = len(train_idx) - cal_size
        model_idx, cal_idx = train_idx[:model_end], train_idx[model_end:]

        if labels.iloc[model_idx].nunique() < 2 or labels.iloc[cal_idx].nunique() < 2:
            continue

        m = numeric.iloc[model_idx].copy(); m["__label__"] = labels.iloc[model_idx].astype(int).to_numpy(); m["__symbol__"] = symbol
        c = numeric.iloc[cal_idx].copy(); c["__label__"] = labels.iloc[cal_idx].astype(int).to_numpy(); c["__symbol__"] = symbol
        model_frames.append(m)
        cal_frames.append(c)

    if not model_frames:
        raise RuntimeError(f"Fold {fold_no}: no symbol had enough training data.")

    pooled_model = pd.concat(model_frames, ignore_index=True)
    pooled_cal = pd.concat(cal_frames, ignore_index=True)
    y_model = pooled_model.pop("__label__"); pooled_model.pop("__symbol__")
    y_cal = pooled_cal.pop("__label__"); pooled_cal.pop("__symbol__")

    print(f"Pooled model-fit rows: {len(pooled_model)} | Pooled inner-calibration rows: {len(pooled_cal)}")
    print("Pooled training labels:", y_model.value_counts().sort_index().to_dict())

    columns = select_features_train_only(pooled_model, y_model, args.max_features, args.random_seed + fold_no)
    medians = fit_medians(pooled_model, columns)
    print(f"Selected features: {len(columns)}")

    X_model = transform_matrix(pooled_model, columns, medians)
    X_cal = transform_matrix(pooled_cal, columns, medians)

    print("Training pooled XGBoost + CatBoost...")
    xgb, cat = train_base_models(X_model, y_model, args.random_seed + fold_no)

    p_cal_raw = raw_blend_probability(xgb, cat, X_cal, args.xgb_weight, args.catboost_weight)
    calibrator = fit_probability_calibrator(p_cal_raw, y_cal.to_numpy(int))
    p_cal_calibrated = apply_probability_calibrator(calibrator, p_cal_raw)
    cal_metrics = classification_metrics(y_cal.to_numpy(int), p_cal_calibrated)

    if args.auto_threshold:
        threshold = select_threshold_inner(p_cal_calibrated, y_cal.to_numpy(int), args.risk_reward,
                                            args.transaction_cost_bps, args.min_expected_value,
                                            min_trades=max(10, len(y_cal)//20))
    else:
        threshold = args.confidence_threshold
    print(f"Frozen threshold for this fold (inner-validation only): {threshold:.2f}")

    # -------------------- OOS: per symbol --------------------
    all_trades, per_symbol_trading, per_symbol_cls, per_symbol_diag, per_symbol_prob_stats = [], {}, {}, {}, {}
    all_test_y, all_test_p = [], []

    for symbol, d in symbol_data.items():
        raw, features, numeric, labels = d["raw"], d["features"], d["numeric"], d["labels"]
        ts = raw["timestamp"]
        test_mask = (ts >= test_start_ts) & (ts <= test_end_ts)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            continue

        X_test = transform_matrix(numeric.iloc[test_idx], columns, medians)
        p_test_raw = raw_blend_probability(xgb, cat, X_test, args.xgb_weight, args.catboost_weight)
        p_test = apply_probability_calibrator(calibrator, p_test_raw)

        # Probability distribution for this fold/symbol's OOS window — this is
        # what actually distinguishes "the model just wasn't confident here"
        # (p_test rarely/never clears threshold — a real market read) from a
        # bug (e.g. p_test is constant, all-NaN, or collapsed to one value).
        finite_p = p_test[np.isfinite(p_test)]
        prob_stats = {
            "n": int(len(p_test)), "n_finite": int(len(finite_p)),
            "min": float(finite_p.min()) if len(finite_p) else None,
            "p25": float(np.percentile(finite_p, 25)) if len(finite_p) else None,
            "median": float(np.median(finite_p)) if len(finite_p) else None,
            "p75": float(np.percentile(finite_p, 75)) if len(finite_p) else None,
            "max": float(finite_p.max()) if len(finite_p) else None,
            "pct_at_or_above_threshold": float((finite_p >= threshold).mean()) if len(finite_p) else None,
        }
        per_symbol_prob_stats[symbol] = prob_stats

        # Independent check on ATR validity for this window (not gated by
        # probability/EV — so if invalid_atr stays 0 below only because
        # rejected_probability already ate every signal, this still tells
        # us whether ATR itself was actually usable here).
        atr_series = features["ATR14"].iloc[test_idx] if "ATR14" in features.columns else pd.Series(dtype=float)
        prob_stats["atr_nan_pct"] = float(atr_series.isna().mean()) if len(atr_series) else None

        y_test = labels.iloc[test_idx].to_numpy(float)
        valid_test = np.isfinite(y_test) & np.isin(y_test, [0.0, 1.0])
        if valid_test.any():
            per_symbol_cls[symbol] = classification_metrics(y_test[valid_test].astype(int), p_test[valid_test])
            all_test_y.extend(y_test[valid_test].astype(int).tolist())
            all_test_p.extend(p_test[valid_test].tolist())

        feature_test = features.iloc[test_idx].reset_index(drop=True)
        trades, diag = run_trade_engine(raw, symbol, feature_test, test_idx, p_test, p_test_raw, threshold, args, fold_no)
        per_symbol_diag[symbol] = diag
        all_trades.extend(trades)
        per_symbol_trading[symbol] = trade_metrics(trades)
        print(f"  {symbol}: {per_symbol_trading[symbol]['trades']} trades, "
              f"return={per_symbol_trading[symbol]['total_return']:+.2%}, "
              f"PF={per_symbol_trading[symbol]['profit_factor']:.2f}")
        if prob_stats["n_finite"] > 0:
            print(f"    probability range [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}], "
                  f"median={prob_stats['median']:.3f}, "
                  f"{prob_stats['pct_at_or_above_threshold']:.1%} of signals >= threshold {threshold:.2f}, "
                  f"ATR14 NaN={prob_stats['atr_nan_pct']:.1%}")
        else:
            print(f"    WARNING: no finite probabilities in this fold's OOS window ({prob_stats['n']} rows total)")
        print(f"    signals_seen={diag['signals_seen']}, accepted={diag['accepted']}, "
              f"rejected_probability={diag['rejected_probability']}, rejected_ev={diag['rejected_ev']}, "
              f"cooldown={diag['cooldown']}, invalid_atr={diag['invalid_atr']}, "
              f"execution_error={diag['execution_error']}")

    combined_cls = classification_metrics(np.array(all_test_y), np.array(all_test_p)) if all_test_y else {}
    combined_trading = trade_metrics(all_trades)

    print(f"\nPOOLED OOS (all symbols combined): {combined_trading['trades']} trades, "
          f"return={combined_trading['total_return']:+.2%}, PF={combined_trading['profit_factor']:.2f}, "
          f"Sharpe={combined_trading['sharpe']:+.2f}")

    return {
        "fold": fold_no, "train_start": str(train_start_ts), "train_end": str(train_end_ts),
        "test_start": str(test_start_ts), "test_end": str(test_end_ts),
        "threshold": threshold, "selected_features": columns,
        "calibration_metrics": cal_metrics, "combined_classification": combined_cls,
        "per_symbol_classification": per_symbol_cls, "per_symbol_trading": per_symbol_trading,
        "per_symbol_diagnostics": per_symbol_diag, "per_symbol_probability_stats": per_symbol_prob_stats,
        "combined_trading": combined_trading, "trades": all_trades,
    }


# ============================================================================
# Main
# ============================================================================

def run_walkforward(args):
    print(f"\n{'='*78}\nPOOLED WALK-FORWARD: {args.symbols} @ {args.timeframe}\n{'='*78}")
    print("2-gate filter (probability threshold + net EV floor) — no regime/score/momentum stacking.\n")

    raw_dfs = {s: load_symbol(s, args.timeframe, args.exchange) for s in args.symbols}
    raw_dfs = restrict_to_common_range(raw_dfs)

    symbol_data = {}
    for symbol, raw in raw_dfs.items():
        features = add_causal_features(raw)
        numeric, dropped = ensure_numeric(features)
        labels = build_binary_labels(raw, features, args.side, args.sl_mult, args.risk_reward, args.max_holding)
        symbol_data[symbol] = {"raw": raw, "features": features, "numeric": numeric, "labels": labels}
        print(f"{symbol}: {len(raw)} rows in common range, {numeric.shape[1]} numeric features, "
              f"labels={labels.value_counts(dropna=False).sort_index().to_dict()}")

    reference_symbol = min(symbol_data, key=lambda s: len(symbol_data[s]["raw"]))
    reference_timestamps = symbol_data[reference_symbol]["raw"]["timestamp"]
    folds = make_calendar_folds(reference_timestamps, args.n_folds, args.min_train_bars, args.test_bars)
    print(f"\n{len(folds)} folds (reference grid: {reference_symbol})")

    fold_results, all_trades = [], []
    for fold_no, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        result = run_fold(fold_no, symbol_data, train_start, train_end, test_start, test_end, args)
        all_trades.extend(result.pop("trades"))
        fold_results.append(result)

    combined = trade_metrics(all_trades)
    per_symbol_combined = {s: trade_metrics([t for t in all_trades if t.symbol == s]) for s in args.symbols}

    print(f"\n{'='*78}\nCOMBINED POOLED WALK-FORWARD OOS RESULT\n{'='*78}")
    print(f"Total trades:   {combined['trades']}")
    print(f"Total return:   {combined['total_return']:+.2%}")
    print(f"Win rate:       {combined['win_rate']:.2%}")
    print(f"Profit factor:  {combined['profit_factor']:.2f}")
    print(f"Sharpe:         {combined['sharpe']:+.2f}")
    print(f"Max drawdown:   {combined['max_drawdown']:+.2%}")
    print("\nPer-symbol breakdown (does the edge hold on each symbol, or only pooled?):")
    for s, m in per_symbol_combined.items():
        print(f"  {s:10s}: {m['trades']:3d} trades, return={m['total_return']:+.2%}, PF={m['profit_factor']:.2f}")

    fold_returns = [f["combined_trading"]["total_return"] for f in fold_results]
    positive_fold_pct = float(np.mean([r > 0 for r in fold_returns])) if fold_returns else 0.0
    status = "EDGE_PROVEN" if (
        combined["trades"] >= args.min_required_trades and combined["total_return"] > 0
        and combined["profit_factor"] > 1.10 and combined["sharpe"] > 0
        and positive_fold_pct >= 0.5 and all(m["trades"] == 0 or m["total_return"] > -0.5 for m in per_symbol_combined.values())
    ) else "EDGE_NOT_PROVEN"

    symbols_safe = "_".join(s.replace("/", "") for s in args.symbols)
    output_dir = Path(args.output_dir or f"models_artifacts/POOLED_{symbols_safe}_{args.timeframe}_walkforward")
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_trades:
        trades_df = pd.DataFrame([asdict(t) for t in all_trades]).sort_values(["fold", "entry_index"]).reset_index(drop=True)
        trades_df.to_csv(output_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()]).to_csv(output_dir / "trades.csv", index=False)

    metrics = {
        "symbols": args.symbols, "timeframe": args.timeframe, "status": status,
        "parameters": vars(args), "folds": fold_results,
        "combined_oos_trading": combined, "per_symbol_combined": per_symbol_combined,
        "fold_return_stats": {"mean": float(np.mean(fold_returns)) if fold_returns else 0.0,
                               "median": float(np.median(fold_returns)) if fold_returns else 0.0,
                               "positive_fold_pct": positive_fold_pct},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nArtifacts: {output_dir}/")
    print(f"\nPOOLED STATUS: {status}")
    return metrics


def build_parser():
    p = argparse.ArgumentParser(description="Pooled multi-symbol, simplified-filter walk-forward validator.")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--side", choices=["long", "short"], default="long")

    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--risk-reward", type=float, default=2.5)
    p.add_argument("--max-holding", type=int, default=48)

    p.add_argument("--confidence-threshold", type=float, default=0.55)
    p.add_argument("--auto-threshold", action="store_true",
                    help="Select the threshold per fold via inner training-only validation instead of the fixed --confidence-threshold.")
    p.add_argument("--min-expected-value", type=float, default=0.0)
    p.add_argument("--xgb-weight", type=float, default=0.65)
    p.add_argument("--catboost-weight", type=float, default=0.35)
    p.add_argument("--cooldown-bars", type=int, default=4)

    p.add_argument("--max-features", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--test-bars", type=int, default=1000)
    p.add_argument("--min-train-bars", type=int, default=3000)
    p.add_argument("--calibration-fraction", type=float, default=0.20)
    p.add_argument("--min-calibration-bars", type=int, default=400)

    p.add_argument("--transaction-cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--min-required-trades", type=int, default=30)
    p.add_argument("--output-dir", default=None)
    return p


def main():
    args = build_parser().parse_args()
    run_walkforward(args)


if __name__ == "__main__":
    main()