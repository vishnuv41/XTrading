"""
strategy_lab/phase18/evaluation/bootstrap.py
---------------------------------------------
Paired bootstrap engine for Phase 18 research.
Evaluates statistical distribution of delta Sharpe (Strategy vs Baseline)
using pre-registered 10,000 resamples and seed 42.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def paired_bootstrap_delta_sharpe(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    n_bootstraps: int = 10000,
    seed: int = 42,
    annualization_factor: float = 365.25,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    Computes paired bootstrap confidence interval for Delta Sharpe (Strategy - Baseline).
    """
    # Align returns
    df = pd.DataFrame({"strat": strategy_returns, "base": baseline_returns}).dropna()
    n = len(df)
    if n < 10:
        return {
            "point_delta_sharpe": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value_one_tailed": 1.0,
        }

    strat_arr = df["strat"].to_numpy()
    base_arr = df["base"].to_numpy()

    # Point estimates
    std_s = np.std(strat_arr, ddof=1)
    std_b = np.std(base_arr, ddof=1)
    sharpe_s = (np.mean(strat_arr) / std_s) * np.sqrt(annualization_factor) if std_s > 1e-12 else 0.0
    sharpe_b = (np.mean(base_arr) / std_b) * np.sqrt(annualization_factor) if std_b > 1e-12 else 0.0
    point_delta = sharpe_s - sharpe_b

    # Vectorized bootstrap
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstraps, n))

    s_boot = strat_arr[indices]
    b_boot = base_arr[indices]

    mean_s_boot = np.mean(s_boot, axis=1)
    std_s_boot = np.std(s_boot, axis=1, ddof=1)
    sharpe_s_boot = np.where(std_s_boot > 1e-12, (mean_s_boot / std_s_boot) * np.sqrt(annualization_factor), 0.0)

    mean_b_boot = np.mean(b_boot, axis=1)
    std_b_boot = np.std(b_boot, axis=1, ddof=1)
    sharpe_b_boot = np.where(std_b_boot > 1e-12, (mean_b_boot / std_b_boot) * np.sqrt(annualization_factor), 0.0)

    delta_sharpes = sharpe_s_boot - sharpe_b_boot

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(delta_sharpes, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(delta_sharpes, 100.0 * (1.0 - alpha / 2.0)))
    p_val = float(np.mean(delta_sharpes <= 0.0))

    return {
        "point_delta_sharpe": float(point_delta),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_one_tailed": p_val,
    }
