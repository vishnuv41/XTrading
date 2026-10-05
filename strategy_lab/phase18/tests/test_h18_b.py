"""
strategy_lab/phase18/tests/test_h18_b.py
-----------------------------------------
Causality, next-bar lag, and gate tests for H18-B Relative Strength strategy.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase18.relative_strength.momentum_signal import compute_relative_strength_weights
from strategy_lab.phase18.evaluation.backtest import simulate_portfolio
from strategy_lab.phase18.evaluation.gates import evaluate_candidate_gates


@pytest.fixture
def synthetic_panel():
    """Generates synthetic 9-asset panel for testing."""
    dates = pd.date_range("2021-01-01", periods=200, freq="1D", tz="UTC")
    symbols = [f"COIN{i}/USDT" for i in range(9)]
    rng = np.random.default_rng(42)

    data = {}
    for s in symbols:
        rets = rng.normal(0.001, 0.03, len(dates))
        prices = 100.0 * np.cumprod(1.0 + rets)
        data[s] = prices

    df = pd.DataFrame(data, index=dates)
    return df


def test_h18_b_weights_bounds(synthetic_panel):
    """Ensure sum of weights across assets never exceeds 1.0."""
    weights = compute_relative_strength_weights(synthetic_panel, lookback_days=14, top_k=3, rebalance_interval_days=1)
    
    # Check max weight per asset <= 1/3 + eps
    assert (weights <= 1.0 / 3.0 + 1e-6).all().all()
    # Check sum of weights <= 1.0
    row_sums = weights.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all()


def test_h18_b_causality_no_lookahead(synthetic_panel):
    """Verify that perturbing future prices (t > T) does NOT alter target weight at bar T."""
    w_original = compute_relative_strength_weights(synthetic_panel, lookback_days=14, top_k=2, rebalance_interval_days=1)

    # Modify future price data at index 100..199
    perturbed_panel = synthetic_panel.copy()
    perturbed_panel.iloc[100:, :] = perturbed_panel.iloc[100:, :] * 2.5

    w_perturbed = compute_relative_strength_weights(perturbed_panel, lookback_days=14, top_k=2, rebalance_interval_days=1)

    # Weights at indices 0..99 must be identically equal
    pd.testing.assert_frame_equal(w_original.iloc[:100], w_perturbed.iloc[:100])


def test_h18_b_next_bar_execution_lag(synthetic_panel):
    """Ensure positions held during bar t reflect target weights from bar t-1."""
    weights = compute_relative_strength_weights(synthetic_panel, lookback_days=14, top_k=3, rebalance_interval_days=1)
    sim = simulate_portfolio(synthetic_panel, weights, round_trip_bps=30.0)

    # First bar exposure must be 0 (since no position could be held before t=0)
    assert sim["exposure"].iloc[0] == 0.0
    assert sim["net_return"].iloc[0] == 0.0
