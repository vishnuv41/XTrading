"""
strategy_lab/lowturn/exposure_null.py
-------------------------------------
Exposure-Matched Null Control Engine (Rule R5).
Tests whether a strategy's performance exceeds random entries matched for:
- Exact average market exposure (time in market)
- Average trade holding duration
Evaluated under identical 30.0 bps canonical round-trip friction.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .cost_model import LowTurnoverCostModel, CANONICAL_ROUND_TRIP_BPS
from .execution import simulate_portfolio_strategy
from .bootstrap import compute_annualized_sharpe


def evaluate_exposure_matched_null(
    panel: Dict[str, pd.DataFrame],
    strategy_weights_dict: Dict[str, pd.Series],
    strategy_sharpe: float,
    n_simulations: int = 10000,
    round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes empirical p-value and null distribution stats.
    Returns: (p_value, null_mean_sharpe, null_std_sharpe).
    """
    rng = np.random.default_rng(seed)
    symbols = list(panel.keys())
    
    # Calculate target exposure & average trade duration per symbol
    durations = {}
    exposures = {}
    for sym in symbols:
        w = strategy_weights_dict[sym].fillna(0.0)
        exposures[sym] = float((w > 0).mean())
        # Estimate run lengths (durations)
        diffs = (w > 0).astype(int).diff().fillna(0)
        starts = (diffs == 1).sum()
        durations[sym] = max(1, int((w > 0).sum() / max(1, starts)))

    null_sharpes = np.zeros(n_simulations)
    n_samples = len(next(iter(panel.values())))

    for sim_idx in range(n_simulations):
        null_weights_dict = {}
        for sym, df in panel.items():
            exp = exposures[sym]
            avg_dur = durations[sym]
            
            # Markov chain transition probabilities to match exact exposure and duration
            # p(stay in state 1) = 1 - 1/avg_dur
            # p(transition 0 -> 1) = exp * (1 - p11) / (1 - exp)
            p11 = max(0.01, 1.0 - (1.0 / max(1.0, avg_dur)))
            p01 = min(0.99, (exp * (1.0 - p11)) / max(0.01, 1.0 - exp)) if exp < 0.99 else 0.99

            state = int(rng.random() < exp)
            seq = np.zeros(len(df))
            for t in range(len(df)):
                seq[t] = state
                if state == 1:
                    state = int(rng.random() < p11)
                else:
                    state = int(rng.random() < p01)

            null_weights_dict[sym] = pd.Series(seq, index=df.index)

        null_res = simulate_portfolio_strategy(panel, null_weights_dict, round_trip_bps=round_trip_bps)
        null_sharpes[sim_idx] = compute_annualized_sharpe(null_res["net_return"])

    p_val = float(np.mean(null_sharpes >= strategy_sharpe))
    null_mean = float(np.mean(null_sharpes))
    null_std = float(np.std(null_sharpes))

    return p_val, null_mean, null_std
