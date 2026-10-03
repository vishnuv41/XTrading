"""
ml/baselines.py
------------------
Phase 10/15 requirement: don't trust the ML model's backtest until it
beats simple baselines under IDENTICAL costs/exits. This builds fake
`predictions` DataFrames (pred_label/confidence, same shape ml.predict
returns) for each baseline and feeds them through the SAME
ml.backtest.run_backtest engine the real model is evaluated with — so
comparisons are apples-to-apples, not two different simulators.

Usage:
    from ml.baselines import compare_to_baselines
    from ml.predict import predict

    predictions = predict(df=df, ensemble=model, feature_columns=feature_columns, calibrator=calibrator)
    results = compare_to_baselines(df, predictions, max_holding=20,
                                    confidence_threshold=0.70, transaction_cost_bps=15.0,
                                    periods_per_year=24*365)
    for name, m in results.items():
        print(name, m["metrics"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.backtest import run_backtest


def _predictions_df(index, pred_label: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"pred_label": pred_label, "confidence": confidence}, index=index)


def buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Always long, confidence=1 so it's never filtered by a threshold."""
    n = len(df)
    return _predictions_df(df.index, np.ones(n, dtype=int), np.ones(n))


def majority_class(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Always predict whatever the real model's single most common label is
    (usually HOLD). If the real model can't beat "always do nothing" (or
    "always predict the majority class"), it has no edge at all."""
    label = predictions["pred_label"].mode().iloc[0]
    n = len(df)
    return _predictions_df(df.index, np.full(n, label, dtype=int), np.ones(n))


def ema_crossover(df: pd.DataFrame, fast_col: str = "EMA20", slow_col: str = "EMA50") -> pd.DataFrame:
    """Classic trend baseline: long while fast EMA > slow EMA, short otherwise.
    Requires df already has fast_col/slow_col (indicators.calculate_all_indicators)."""
    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(f"df missing '{fast_col}' / '{slow_col}' — run calculate_all_indicators first.")
    label = np.where(df[fast_col] > df[slow_col], 1, -1)
    return _predictions_df(df.index, label, np.ones(len(df)))


def rsi_mean_reversion(df: pd.DataFrame, rsi_col: str = "RSI14",
                        low: float = 30.0, high: float = 70.0) -> pd.DataFrame:
    """Long when oversold, short when overbought, else flat (label 0)."""
    if rsi_col not in df.columns:
        raise ValueError(f"df missing '{rsi_col}' — run calculate_all_indicators first.")
    rsi = df[rsi_col].values
    label = np.zeros(len(df), dtype=int)
    label[rsi < low] = 1
    label[rsi > high] = -1
    return _predictions_df(df.index, label, np.ones(len(df)))


def random_signal(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Sanity floor: uniform random -1/0/1. The model should beat this by
    a wide margin — if it doesn't, something upstream is broken, not just
    "not profitable yet"."""
    rng = np.random.default_rng(seed)
    label = rng.choice([-1, 0, 1], size=len(df))
    return _predictions_df(df.index, label, np.ones(len(df)))


def compare_to_baselines(
    df: pd.DataFrame,
    model_predictions: pd.DataFrame,
    max_holding: int = 20,
    confidence_threshold: float = 0.70,
    transaction_cost_bps: float = 15.0,
    periods_per_year: int = 8760,
) -> dict:
    """
    Runs the real model's predictions plus every baseline through the
    same run_backtest() with the same cost/exit settings, and returns
    {name: run_backtest() result} for every candidate.

    transaction_cost_bps default of 15.0 (5bps fee + slippage-ish) is
    intentionally >= the 10bps paper-trading fee alone, since
    run_backtest's cost is charged once per side (2x per round trip) —
    check this matches your paper_trading/config.py fee+slippage total
    before trusting the comparison.
    """
    common = dict(
        max_holding=max_holding, confidence_threshold=confidence_threshold,
        transaction_cost_bps=transaction_cost_bps, periods_per_year=periods_per_year,
    )

    candidates = {
        "model": model_predictions,
        "buy_and_hold": buy_and_hold(df),
        "majority_class": majority_class(df, model_predictions),
        "random": random_signal(df),
    }
    # EMA/RSI baselines need indicators already on df — skip gracefully if absent.
    try:
        candidates["ema_crossover"] = ema_crossover(df)
    except ValueError:
        pass
    try:
        candidates["rsi_mean_reversion"] = rsi_mean_reversion(df)
    except ValueError:
        pass

    results = {}
    for name, preds in candidates.items():
        preds = preds.reindex(df.index)
        results[name] = run_backtest(df, preds, **common)
    return results


def print_comparison(results: dict) -> None:
    """Compact table: total_return / sharpe / max_dd / win_rate / profit_factor / num_trades."""
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({
            "strategy": name,
            "total_return": m["total_return"],
            "sharpe": m["sharpe_ratio"],
            "max_dd": m["max_drawdown"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "num_trades": m["num_trades"],
        })
    table = pd.DataFrame(rows).set_index("strategy")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(table)