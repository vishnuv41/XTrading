"""
tests/test_lowturn_harness.py
-----------------------------
Unit tests for Phase 17 low-turnover benchmark harness:
- Rule R1 (Causality) test
- Rule R2 (Execution lag t+1) test
- Rule R3 (Cost application) test
- Rule R6 (Calendar-week bootstrap) test
- Strategy grid generation test
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from strategy_lab.lowturn.causality import verify_causality_r1
from strategy_lab.lowturn.execution import simulate_single_asset_strategy, simulate_portfolio_strategy
from strategy_lab.lowturn.cost_model import LowTurnoverCostModel
from strategy_lab.lowturn.bootstrap import calendar_week_block_bootstrap_sharpe_ci, compute_annualized_sharpe
from strategy_lab.lowturn.strategies import build_frozen_strategy_grid, generate_single_ema_weights


@pytest.fixture
def sample_ohlcv():
    n = 300
    dates = pd.date_range("2021-01-01", periods=n, freq="1D", tz=timezone.utc)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, n)
    price = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "open": price * (1.0 + rng.normal(0, 0.002, n)),
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": rng.uniform(100, 1000, n),
        },
        index=dates,
    )


def test_causality_r1(sample_ohlcv):
    grid = build_frozen_strategy_grid("1d")
    for name, fn in grid.items():
        assert verify_causality_r1(fn, sample_ohlcv, test_bar_idx=150), f"Causality R1 failed for {name}"


def test_execution_lag_r2(sample_ohlcv):
    # Strategy flips weight from 0 to 1 at bar index 50
    weights = pd.Series(0.0, index=sample_ohlcv.index)
    weights.iloc[50:] = 1.0

    res = simulate_single_asset_strategy(sample_ohlcv, weights, round_trip_bps=30.0)

    # At bar 50, pos_held should still be 0.0
    assert res["pos_held"].iloc[50] == 0.0
    # At bar 51, pos_held becomes 1.0
    assert res["pos_held"].iloc[51] == 1.0
    # Turnover occurs at bar 51
    assert res["turnover"].iloc[51] == 1.0
    assert res["friction"].iloc[51] == 0.0015  # 15 bps one-way


def test_cost_model():
    cm = LowTurnoverCostModel(round_trip_bps=30.0)
    assert cm.round_trip_bps == 30.0
    assert cm.one_way_pct == 0.0015
    assert cm.apply_turnover_friction(2.0) == 0.0030


def test_calendar_week_bootstrap(sample_ohlcv):
    rets = sample_ohlcv["close"].pct_change().dropna()
    pt, lo, hi = calendar_week_block_bootstrap_sharpe_ci(rets, n_bootstraps=200, seed=1)
    assert lo <= pt <= hi
    assert not np.isnan(lo)
    assert not np.isnan(hi)
