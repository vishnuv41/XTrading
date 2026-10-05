"""
strategy_lab/phase19/tests/test_phase19_regime.py
-------------------------------------------------
Causality and unit tests for Phase 19 Market Regime Classifier.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase19.regime_engine import compute_market_breadth, compute_asset_atr_percentile, compute_trend_quality
from strategy_lab.phase19.regime_signals import generate_regime_weights


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


def test_market_breadth_bounds(synthetic_panel):
    breadth = compute_market_breadth(synthetic_panel, ema_period=50)
    assert len(breadth) == len(synthetic_panel)
    assert (breadth >= 0.0).all() and (breadth <= 1.0).all()


def test_regime_causality_no_lookahead(synthetic_panel):
    """Ensure altering future prices at t > T does not change breadth or signals at t <= T."""
    w_orig = generate_regime_weights(synthetic_panel, rule_id="Regime_Breadth_Filter")

    perturbed = synthetic_panel.copy()
    perturbed.iloc[100:, :] = perturbed.iloc[100:, :] * 3.0

    w_pert = generate_regime_weights(perturbed, rule_id="Regime_Breadth_Filter")

    pd.testing.assert_frame_equal(w_orig.iloc[:100], w_pert.iloc[:100])


def test_all_rules_execute(synthetic_panel):
    for r_id in ["Regime_Breadth_Filter", "Regime_Strong_Breadth_Filter", "Regime_Composite_Filter"]:
        w = generate_regime_weights(synthetic_panel, rule_id=r_id)
        assert w.shape == synthetic_panel.shape
        assert not w.isna().any().any()
        assert (w.sum(axis=1) <= 1.0 + 1e-6).all()
