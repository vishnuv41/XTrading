"""
strategy_lab/phase18/tests/test_h18_a.py
-----------------------------------------
Unit, causality, and property tests for Hypothesis H18-A Continuous Funding Overlay.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase18.funding.funding_signal import compute_funding_overlay_weights, load_daily_funding_features
from strategy_lab.phase18.evaluation.backtest import simulate_portfolio


@pytest.fixture
def synthetic_panel():
    """Generates synthetic 9-asset panel for testing."""
    dates = pd.date_range("2021-01-01", periods=200, freq="1D", tz="UTC")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "LTC/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT"]
    rng = np.random.default_rng(42)

    data = {}
    for s in symbols:
        rets = rng.normal(0.001, 0.03, len(dates))
        prices = 100.0 * np.cumprod(1.0 + rets)
        data[s] = prices

    df = pd.DataFrame(data, index=dates)
    return df


def test_h18_a_weights_bounds(synthetic_panel):
    """Ensure sum of weights across assets never exceeds 1.0."""
    weights = compute_funding_overlay_weights(
        synthetic_panel,
        lookback_days=14,
        overlay_weight=0.50,
        overlay_mode="contrarian",
    )
    
    # Check max weight per asset <= 1.0
    assert (weights <= 1.0 + 1e-6).all().all()
    # Check sum of weights <= 1.0
    row_sums = weights.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all()


def test_h18_a_modes_execute(synthetic_panel):
    """Verify both contrarian and momentum overlay modes execute cleanly."""
    w_contrarian = compute_funding_overlay_weights(synthetic_panel, lookback_days=14, overlay_weight=0.25, overlay_mode="contrarian")
    w_momentum = compute_funding_overlay_weights(synthetic_panel, lookback_days=14, overlay_weight=0.25, overlay_mode="momentum")

    assert w_contrarian.shape == synthetic_panel.shape
    assert w_momentum.shape == synthetic_panel.shape
    assert not w_contrarian.isna().any().any()
    assert not w_momentum.isna().any().any()
