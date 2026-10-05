"""
strategy_lab/phase20/tests/test_phase20_portfolio.py
---------------------------------------------------
Causality and property tests for Phase 20 Portfolio Construction.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase20.portfolio_engine import compute_normalized_atr_volatility, compute_continuous_breadth_exposure
from strategy_lab.phase20.portfolio_signals import generate_portfolio_weights


@pytest.fixture
def synthetic_panel():
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


def test_portfolio_weights_bounds(synthetic_panel):
    for m_id in ["Risk_Parity_Sizing", "Breadth_Scaled_Exposure", "Composite_Risk_Portfolio"]:
        w = generate_portfolio_weights(synthetic_panel, mechanism_id=m_id)
        assert w.shape == synthetic_panel.shape
        assert not w.isna().any().any()
        assert (w.sum(axis=1) <= 1.0 + 1e-6).all()
        assert (w >= 0.0).all().all()


def test_portfolio_causality_no_lookahead(synthetic_panel):
    """Ensure future price modifications do not alter weights at bar t <= T."""
    w_orig = generate_portfolio_weights(synthetic_panel, mechanism_id="Composite_Risk_Portfolio")

    perturbed = synthetic_panel.copy()
    perturbed.iloc[100:, :] = perturbed.iloc[100:, :] * 2.5

    w_pert = generate_portfolio_weights(perturbed, mechanism_id="Composite_Risk_Portfolio")

    pd.testing.assert_frame_equal(w_orig.iloc[:100], w_pert.iloc[:100])
