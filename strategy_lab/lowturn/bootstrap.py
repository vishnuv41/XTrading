"""
strategy_lab/lowturn/bootstrap.py
---------------------------------
Calendar-Week Date-Cluster Block Bootstrap (Rule R6).
Resamples calendar-week blocks with replacement to preserve cross-sectional correlation
and short-horizon temporal dependency.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd


def compute_annualized_sharpe(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Compute annualized Sharpe ratio assuming zero risk-free rate."""
    std = returns.std()
    if std < 1e-8 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def calendar_week_block_bootstrap_sharpe_ci(
    returns_series: pd.Series,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    periods_per_year: int = 365,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes [lower, upper] CI for annualized Sharpe using calendar-week date clusters.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    if len(returns_series) < 14:
        sharpe = compute_annualized_sharpe(returns_series, periods_per_year)
        return sharpe, sharpe, sharpe

    # Assign calendar week clusters (ISO year and week)
    df = pd.DataFrame({"ret": returns_series})
    naive_idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df["year_week"] = naive_idx.to_period("W")
    
    unique_weeks = df["year_week"].unique()
    n_weeks = len(unique_weeks)
    
    # Pre-aggregate returns by week
    week_groups = [df.loc[df["year_week"] == w, "ret"].values for w in unique_weeks]
    
    rng = np.random.default_rng(seed)
    boot_sharpes = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sampled_indices = rng.integers(0, n_weeks, size=n_weeks)
        sampled_rets = np.concatenate([week_groups[idx] for idx in sampled_indices])
        s_std = np.std(sampled_rets)
        if s_std > 1e-8:
            boot_sharpes[i] = np.sqrt(periods_per_year) * np.mean(sampled_rets) / s_std
        else:
            boot_sharpes[i] = 0.0

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_sharpes, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_sharpes, 100 * (1.0 - alpha / 2.0)))
    point_est = compute_annualized_sharpe(returns_series, periods_per_year)

    return point_est, ci_lower, ci_upper
