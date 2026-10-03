from __future__ import annotations
"""
phase14/validation/deconstruction_suite.py
-------------------------------------------
Model Deconstruction Suite (Experiments A-D):
- Experiment A: Long Tail Expected Return E[R_actual | r_hat >= +T]
- Experiment B: Short Tail Expected Return E[R_actual | r_hat <= -T]
- Experiment C: Magnitude Correlation (|r_hat| vs |r_actual|)
- Experiment D: Directional Accuracy (sign(r_hat) == sign(r_actual))
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def evaluate_model_deconstruction(
    r_pred: np.ndarray,
    r_actual: np.ndarray,
    hurdle_thresholds: list[float] = [0.002, 0.004, 0.006, 0.008, 0.010],
    roundtrip_cost: float = 0.0030
) -> dict:
    """
    Deconstructs ML model prediction performance across long/short tails, magnitude, and direction.
    """
    valid_mask = ~np.isnan(r_pred) & ~np.isnan(r_actual)
    r_p = r_pred[valid_mask]
    r_a = r_actual[valid_mask]
    n_samples = len(r_p)

    if n_samples == 0:
        return {"error": "No valid samples for evaluation"}

    # Experiment C: Magnitude Correlation
    mag_corr = float(np.corrcoef(np.abs(r_p), np.abs(r_a))[0, 1]) if n_samples > 1 else 0.0

    # Experiment D: Directional Sign Agreement
    same_sign = (np.sign(r_p) == np.sign(r_a)) & (r_p != 0)
    directional_accuracy = float(np.mean(same_sign)) if n_samples > 0 else 0.0

    # Tail Experiments A & B across Hurdle Thresholds T
    hurdle_results = []
    for T in hurdle_thresholds:
        # Long Tail (Exp A)
        long_mask = r_p >= T
        n_long = int(np.sum(long_mask))
        mean_long_gross = float(np.mean(r_a[long_mask])) if n_long > 0 else 0.0
        mean_long_net = mean_long_gross - roundtrip_cost
        long_win_rate = float(np.mean(r_a[long_mask] > roundtrip_cost)) if n_long > 0 else 0.0

        # Short Tail (Exp B)
        short_mask = r_p <= -T
        n_short = int(np.sum(short_mask))
        # Short return is inverted (-r_actual)
        mean_short_gross = float(np.mean(-r_a[short_mask])) if n_short > 0 else 0.0
        mean_short_net = mean_short_gross - roundtrip_cost
        short_win_rate = float(np.mean(-r_a[short_mask] > roundtrip_cost)) if n_short > 0 else 0.0

        hurdle_results.append({
            "hurdle_T": T,
            "n_long": n_long,
            "long_gross_return_pct": mean_long_gross * 100.0,
            "long_net_return_pct": mean_long_net * 100.0,
            "long_win_rate_pct": long_win_rate * 100.0,
            "n_short": n_short,
            "short_gross_return_pct": mean_short_gross * 100.0,
            "short_net_return_pct": mean_short_net * 100.0,
            "short_win_rate_pct": short_win_rate * 100.0,
        })

    return {
        "n_samples": n_samples,
        "magnitude_correlation": mag_corr,
        "directional_accuracy_pct": directional_accuracy * 100.0,
        "hurdle_experiments": hurdle_results
    }
