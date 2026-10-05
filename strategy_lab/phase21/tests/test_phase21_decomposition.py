"""
strategy_lab/phase21/tests/test_phase21_decomposition.py
--------------------------------------------------------
Unit tests for Phase 21 baseline forensic decomposition.
"""

import pytest
import numpy as np
import pandas as pd

from strategy_lab.phase21.decomposition_engine import extract_discrete_trades, run_full_baseline_decomposition


@pytest.fixture
def sample_price_panel():
    dates = pd.date_range("2021-01-01", periods=100, freq="1D", tz="UTC")
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.default_rng(42)

    data = {}
    for s in symbols:
        rets = rng.normal(0.001, 0.02, len(dates))
        prices = 100.0 * np.cumprod(1.0 + rets)
        data[s] = prices

    df = pd.DataFrame(data, index=dates)
    return df


def test_trade_extraction():
    prices = pd.Series([100.0, 102.0, 105.0, 103.0, 99.0, 98.0, 101.0, 104.0])
    weights = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    trades = extract_discrete_trades(prices, weights, round_trip_bps=30.0)
    assert len(trades) >= 1
    for t in trades:
        assert "net_return_pct" in t
        assert "holding_days" in t
        assert t["holding_days"] > 0


def test_full_decomposition_execution(sample_price_panel):
    res = run_full_baseline_decomposition(sample_price_panel, canonical_bps=30.0)

    assert "asset_attribution" in res
    assert "yearly_attribution" in res
    assert "regime_attribution" in res
    assert "trade_expectancy" in res
    assert "friction_sensitivity" in res
    assert "benchmark_comparison" in res
    assert len(res["asset_attribution"]) == 3
