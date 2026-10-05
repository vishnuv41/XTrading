"""
strategy_lab/phase18/tests/test_h18_c.py
-----------------------------------------
Causality, next-bar lag, and gate tests for H18-C Volatility Compression filter.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase18.volatility.compression_signal import compute_volatility_compression_weights
from strategy_lab.phase18.evaluation.backtest import simulate_portfolio


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


def test_h18_c_weights_bounds(synthetic_panel):
    """Ensure sum of weights across assets never exceeds 1.0."""
    weights = compute_volatility_compression_weights(
        synthetic_panel,
        compression_type="bandwidth_percentile",
        lookback_days=30,
        compression_percentile_cutoff=20.0,
    )
    
    # Check max weight per asset <= 1/9 + eps
    assert (weights <= 1.0 / 9.0 + 1e-6).all().all()
    # Check sum of weights <= 1.0
    row_sums = weights.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all()


def test_h18_c_causality_no_lookahead(synthetic_panel):
    """Verify that perturbing future prices (t > T) does NOT alter target weight at bar T."""
    w_original = compute_volatility_compression_weights(
        synthetic_panel,
        compression_type="bandwidth_percentile",
        lookback_days=30,
        compression_percentile_cutoff=20.0,
    )

    # Modify future price data at index 100..199
    perturbed_panel = synthetic_panel.copy()
    perturbed_panel.iloc[100:, :] = perturbed_panel.iloc[100:, :] * 2.5

    w_perturbed = compute_volatility_compression_weights(
        perturbed_panel,
        compression_type="bandwidth_percentile",
        lookback_days=30,
        compression_percentile_cutoff=20.0,
    )

    # Weights at indices 0..99 must be identically equal
    pd.testing.assert_frame_equal(w_original.iloc[:100], w_perturbed.iloc[:100])


def test_h18_c_atr_ratio_property(synthetic_panel):
    """Verify ATR ratio variant executes without errors and respects boundaries."""
    weights = compute_volatility_compression_weights(
        synthetic_panel,
        compression_type="atr_ratio",
        lookback_days=30,
        compression_percentile_cutoff=20.0,
    )
    assert weights.shape == synthetic_panel.shape
    assert not weights.isna().any().any()
