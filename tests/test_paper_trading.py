"""
tests/test_paper_trading.py
------------------------------
Unit tests for paper_trading/. Deliberately does NOT require a live
Postgres or a trained model:
  - Portfolio/execution/exit-manager logic is tested with plain Python
    objects (no I/O at all).
  - db_logger is tested against an in-memory SQLite engine (same
    parameterized-SQL/ON CONFLICT semantics as Postgres for what we use
    here), injected via the `engine=` param every db_logger function
    accepts.
  - PaperTradingEngine.on_bar is tested end-to-end with a STUBBED
    run_realtime_pipeline (patched, same pattern as
    tests/test_realtime_pipeline.py) so it doesn't need a real model or DB.

Run: pytest tests/test_paper_trading.py -v
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from paper_trading import db_logger
from paper_trading.config import PaperTradingConfig
from paper_trading.engine import PaperTradingEngine
from paper_trading.exit_manager import check_exit
from paper_trading.execution import ExecutionSimulator
from paper_trading.models import OpenPosition, PredictionRecord
from paper_trading.portfolio import DuplicatePositionError, VirtualPortfolio

UTC = timezone.utc


def _ts(i=0):
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i)


# ---------------------------------------------------------------------
# VirtualPortfolio
# ---------------------------------------------------------------------

def test_open_position_deducts_cash_and_fee():
    p = VirtualPortfolio(starting_cash=10_000)
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=10.0,
        stop_loss=95.0, take_profit=110.0, risk_pct=0.01, entry_fee=1.0,
    )
    p.open_position(pos)
    assert p.cash == 10_000 - (10.0 * 100.0) - 1.0
    assert p.has_open_position("BTC/USDT")


def test_duplicate_position_raises():
    p = VirtualPortfolio(starting_cash=10_000)
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=1.0,
        stop_loss=95.0, take_profit=110.0, risk_pct=0.01, entry_fee=0.1,
    )
    p.open_position(pos)
    dup = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(1), entry_price=101.0, size=1.0,
        stop_loss=96.0, take_profit=111.0, risk_pct=0.01, entry_fee=0.1,
    )
    with pytest.raises(DuplicatePositionError):
        p.open_position(dup)


def test_insufficient_cash_raises():
    p = VirtualPortfolio(starting_cash=100)
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=10.0,  # notional 1000 > 100 cash
        stop_loss=95.0, take_profit=110.0, risk_pct=0.01, entry_fee=1.0,
    )
    with pytest.raises(ValueError):
        p.open_position(pos)


def test_close_position_realizes_pnl_net_of_fees():
    p = VirtualPortfolio(starting_cash=10_000)
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=10.0,
        stop_loss=95.0, take_profit=110.0, risk_pct=0.01, entry_fee=1.0,
    )
    p.open_position(pos)
    cash_before_close = p.cash

    trade = p.close_position(
        symbol="BTC/USDT", exit_ts=_ts(1), exit_price=110.0, exit_fee=1.1, exit_reason="take_profit",
    )

    gross = (110.0 - 100.0) * 10.0  # 100
    expected_net = gross - 1.0 - 1.1
    assert trade.realized_pnl == pytest.approx(expected_net)
    assert trade.is_win
    assert not p.has_open_position("BTC/USDT")

    # proceeds credited back = entry notional (1000) + gross_pnl (100) - exit_fee (1.1)
    entry_notional = 10.0 * 100.0
    expected_proceeds = entry_notional + gross - 1.1
    assert p.cash == pytest.approx(cash_before_close + expected_proceeds)


def test_short_position_pnl_direction():
    p = VirtualPortfolio(starting_cash=10_000)
    pos = OpenPosition.new(
        exchange="binance", symbol="ETH/USDT", timeframe="1h", side="short",
        entry_ts=_ts(), entry_price=100.0, size=5.0,
        stop_loss=105.0, take_profit=90.0, risk_pct=0.01, entry_fee=0.5,
    )
    p.open_position(pos)
    trade = p.close_position(
        symbol="ETH/USDT", exit_ts=_ts(1), exit_price=90.0, exit_fee=0.45, exit_reason="take_profit",
    )
    # price fell 10, short profits: (100-90)*5 = 50 gross
    assert trade.realized_pnl == pytest.approx(50 - 0.5 - 0.45)
    assert trade.is_win


# ---------------------------------------------------------------------
# exit_manager
# ---------------------------------------------------------------------

def _long_position(stop=95.0, target=110.0, bars_held=0):
    return OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=1.0,
        stop_loss=stop, take_profit=target, risk_pct=0.01, entry_fee=0.1,
    )


def test_exit_stop_loss_triggers():
    pos = _long_position()
    decision = check_exit(pos, bar_high=101, bar_low=94, timeout_bars=48)
    assert decision.should_exit
    assert decision.reason == "stop_loss"
    assert decision.fill_price == 95.0


def test_exit_take_profit_triggers():
    pos = _long_position()
    decision = check_exit(pos, bar_high=111, bar_low=99, timeout_bars=48)
    assert decision.should_exit
    assert decision.reason == "take_profit"


def test_exit_ambiguous_bar_prefers_stop_loss():
    """Bar range touches BOTH SL and TP — SL wins (conservative assumption, see exit_manager docstring)."""
    pos = _long_position()
    decision = check_exit(pos, bar_high=120, bar_low=90, timeout_bars=48)
    assert decision.reason == "stop_loss"


def test_exit_timeout_triggers_when_bars_held_exceeds_limit():
    pos = _long_position(bars_held=5)
    pos.bars_held = 5
    decision = check_exit(pos, bar_high=101, bar_low=99, timeout_bars=5)
    assert decision.should_exit
    assert decision.reason == "timeout"
    assert decision.fill_price is None  # caller supplies bar close


def test_exit_no_trigger_holds():
    pos = _long_position(bars_held=1)
    decision = check_exit(pos, bar_high=101, bar_low=99, timeout_bars=48)
    assert not decision.should_exit


# ---------------------------------------------------------------------
# ExecutionSimulator (fees, slippage, duplicate guard, HOLD handling)
# ---------------------------------------------------------------------

def test_open_from_prediction_hold_does_nothing():
    p = VirtualPortfolio(starting_cash=10_000)
    ex = ExecutionSimulator()
    result = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "HOLD", risk={})
    assert result is None
    assert not p.has_open_position("BTC/USDT")


def test_open_from_prediction_blocked_by_risk_engine():
    """risk dict missing stop_loss/take_profit/position_size => risk engine blocked the trade upstream."""
    p = VirtualPortfolio(starting_cash=10_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": None, "take_profit": None, "position_size": None, "entry_price": 100.0}
    result = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert result is None


def test_open_from_prediction_rejects_nan_stop_loss():
    """NaN != None and NaN is truthy, so a plain `is None`/`not x` guard
    lets a NaN stop_loss/take_profit/position_size through — which then
    silently disables exit_manager's SL/TP checks forever (NaN
    comparisons are always False). Regression test for that bug."""
    import math
    p = VirtualPortfolio(starting_cash=10_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": math.nan, "take_profit": 110.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    result = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert result is None
    assert not p.has_open_position("BTC/USDT")


def test_open_from_prediction_rejects_nan_position_size():
    import math
    p = VirtualPortfolio(starting_cash=10_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 95.0, "take_profit": 110.0, "position_size": math.nan,
            "risk_pct": 0.01, "entry_price": 100.0}
    result = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert result is None
    assert not p.has_open_position("BTC/USDT")


def test_open_from_prediction_applies_slippage_against_trader():
    p = VirtualPortfolio(starting_cash=100_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 95.0, "take_profit": 110.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    pos = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert pos is not None
    # slippage_bps default 5.0 -> long entry fills HIGHER than reference
    assert pos.entry_price > 100.0


def test_open_from_prediction_duplicate_guard_returns_none():
    p = VirtualPortfolio(starting_cash=100_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 95.0, "take_profit": 110.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    first = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    second = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(1), "BUY", risk=risk)
    assert first is not None
    assert second is None
    assert len(p.open_positions) == 1


def test_check_and_close_increments_bars_held_without_exit():
    p = VirtualPortfolio(starting_cash=100_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 90.0, "take_profit": 120.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    pos = p.get_open_position("BTC/USDT")
    assert pos.bars_held == 0
    trade = ex.check_and_close(p, "BTC/USDT", _ts(1), bar_high=101, bar_low=99, bar_close=100.5)
    assert trade is None
    assert pos.bars_held == 1


def test_open_from_prediction_clamps_size_to_available_margin(monkeypatch):
    """
    risk_engine can size a position at ~equity notional even at low
    risk_pct (tight stop). With leverage=1.0 (cash account) that should
    CLAMP the size to what cash actually affords, not reject the trade.
    """
    monkeypatch.setattr("paper_trading.execution.pt_settings",
                         PaperTradingConfig(leverage=1.0), raising=False)
    p = VirtualPortfolio(starting_cash=1_000)
    ex = ExecutionSimulator()
    # position_size implies notional = 20 * 100 = 2000, way over the 1000 cash
    risk = {"stop_loss": 99.0, "take_profit": 102.0, "position_size": 20.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    pos = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert pos is not None
    assert pos.size < 20.0  # clamped down, not rejected
    assert pos.notional_value <= p.starting_cash * 1.01  # roughly affordable at 1x leverage


def test_open_from_prediction_leverage_allows_notional_above_cash(monkeypatch):
    """At leverage > 1, a position's notional can legitimately exceed cash on hand."""
    monkeypatch.setattr("paper_trading.execution.pt_settings",
                         PaperTradingConfig(leverage=5.0), raising=False)
    p = VirtualPortfolio(starting_cash=1_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 95.0, "take_profit": 110.0, "position_size": 20.0,
            "risk_pct": 0.01, "entry_price": 100.0}  # notional ~2000, margin ~400 at 5x
    pos = ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    assert pos is not None
    assert pos.size == pytest.approx(20.0)  # not clamped, fits within margin
    assert pos.notional_value > p.starting_cash  # only possible because of leverage


def test_check_and_close_exits_on_stop_loss():
    p = VirtualPortfolio(starting_cash=100_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 90.0, "take_profit": 120.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    trade = ex.check_and_close(p, "BTC/USDT", _ts(1), bar_high=95, bar_low=88, bar_close=89)
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert not p.has_open_position("BTC/USDT")


def test_signal_flip_closes_opposite_position():
    p = VirtualPortfolio(starting_cash=100_000)
    ex = ExecutionSimulator()
    risk = {"stop_loss": 90.0, "take_profit": 120.0, "position_size": 1.0,
            "risk_pct": 0.01, "entry_price": 100.0}
    ex.open_from_prediction(p, "BTC/USDT", "1h", _ts(), "BUY", risk=risk)
    trade = ex.close_on_signal_flip(p, "BTC/USDT", _ts(1), bar_close=102.0, new_prediction="SELL")
    assert trade is not None
    assert trade.exit_reason == "signal_flip"
    assert not p.has_open_position("BTC/USDT")


# ---------------------------------------------------------------------
# db_logger (SQLite in-memory stand-in for Postgres)
# ---------------------------------------------------------------------

@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE prediction_log (
                id INTEGER PRIMARY KEY, exchange TEXT, symbol TEXT, timeframe TEXT, ts TEXT,
                prediction TEXT, confidence REAL, prob_down REAL, prob_flat REAL, prob_up REAL,
                market_state TEXT, trend TEXT, volatility TEXT, entry_price REAL, stop_loss REAL,
                take_profit REAL, risk_reward_ratio REAL, position_size REAL, risk_pct REAL,
                notional_value REAL, risk_block_reason TEXT, executed INTEGER,
                UNIQUE (exchange, symbol, timeframe, ts)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_log (
                id INTEGER PRIMARY KEY, trade_id TEXT, exchange TEXT, symbol TEXT, timeframe TEXT,
                side TEXT, action TEXT, ts TEXT, price REAL, size REAL, fee REAL,
                stop_loss REAL, take_profit REAL, exit_reason TEXT, realized_pnl REAL,
                realized_pnl_pct REAL, bars_held INTEGER, cash_after REAL, equity_after REAL,
                UNIQUE (trade_id, action)
            )
        """))
    return engine


def test_log_prediction_writes_row(sqlite_engine):
    record = PredictionRecord(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", ts=_ts(), prediction="BUY",
        confidence=0.8, executed=True,
    )
    db_logger.log_prediction(record, engine=sqlite_engine)
    with sqlite_engine.connect() as conn:
        rows = conn.execute(text("SELECT symbol, prediction, executed FROM prediction_log")).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "BTC/USDT"
    assert rows[0][1] == "BUY"


def test_log_prediction_idempotent_on_conflict(sqlite_engine):
    record = PredictionRecord(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", ts=_ts(), prediction="BUY",
    )
    db_logger.log_prediction(record, engine=sqlite_engine)
    db_logger.log_prediction(record, engine=sqlite_engine)  # same (exchange,symbol,timeframe,ts) -> no-op
    with sqlite_engine.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM prediction_log")).fetchall()
    assert rows[0][0] == 1


def test_log_trade_open_and_close_writes_two_rows(sqlite_engine):
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=_ts(), entry_price=100.0, size=1.0,
        stop_loss=95.0, take_profit=110.0, risk_pct=0.01, entry_fee=0.1,
    )
    db_logger.log_trade_open(pos, cash_after=9899.9, equity_after=10000.0, engine=sqlite_engine)

    portfolio = VirtualPortfolio(starting_cash=10_000)
    portfolio.open_position(pos)
    trade = portfolio.close_position(
        symbol="BTC/USDT", exit_ts=_ts(1), exit_price=110.0, exit_fee=0.11, exit_reason="take_profit",
    )
    db_logger.log_trade_close(trade, cash_after=portfolio.cash, equity_after=portfolio.cash, engine=sqlite_engine)

    with sqlite_engine.connect() as conn:
        rows = conn.execute(text("SELECT action, trade_id FROM trade_log ORDER BY id")).fetchall()
    assert [r[0] for r in rows] == ["OPEN", "CLOSE"]
    assert rows[0][1] == rows[1][1] == pos.trade_id


# ---------------------------------------------------------------------
# PaperTradingEngine.on_bar (stubbed pipeline, no DB persistence)
# ---------------------------------------------------------------------

def _synthetic_df(n=260, start_price=100.0):
    rng_dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(start_price, index=range(n))
    return pd.DataFrame({
        "timestamp": rng_dates, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 1000,
    })


def test_engine_on_bar_opens_position_from_buy_signal():
    df = _synthetic_df()
    engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h", model=None, feature_columns=[],
        persist_to_db=False,
    )
    fake_result = {
        "prediction": "BUY", "confidence": 0.9, "entry": 100.0,
        "stop_loss": 95.0, "take_profit": 110.0, "risk_reward_ratio": 2.0,
        "position_size": 1.0, "risk_pct": 0.01, "notional_value": 100.0,
        "risk_block_reason": None, "market_state": "trending", "trend": "up", "volatility": "medium",
    }
    with patch("paper_trading.engine.run_realtime_pipeline", return_value=fake_result):
        summary = engine.on_bar(df)

    assert summary["executed"] is True
    assert summary["prediction"] == "BUY"
    assert engine.portfolio.has_open_position("BTC/USDT")


def test_engine_on_bar_respects_duplicate_guard_across_calls():
    df = _synthetic_df()
    engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h", model=None, feature_columns=[],
        persist_to_db=False,
    )
    fake_result = {
        "prediction": "BUY", "confidence": 0.9, "entry": 100.0,
        "stop_loss": 95.0, "take_profit": 110.0, "risk_reward_ratio": 2.0,
        "position_size": 1.0, "risk_pct": 0.01, "notional_value": 100.0,
        "risk_block_reason": None, "market_state": "trending", "trend": "up", "volatility": "medium",
    }
    with patch("paper_trading.engine.run_realtime_pipeline", return_value=fake_result):
        s1 = engine.on_bar(df)
        s2 = engine.on_bar(pd.concat([df, df.iloc[[-1]]], ignore_index=True))

    assert s1["executed"] is True
    assert s2["executed"] is False  # duplicate guard: still open from s1
    assert len(engine.portfolio.open_positions) == 1


def test_engine_on_bar_hold_does_not_trade():
    df = _synthetic_df()
    engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h", model=None, feature_columns=[],
        persist_to_db=False,
    )
    fake_result = {
        "prediction": "HOLD", "confidence": 0.3, "entry": 100.0,
        "stop_loss": None, "take_profit": None, "risk_reward_ratio": None,
        "position_size": None, "risk_pct": None, "notional_value": None,
        "risk_block_reason": None, "market_state": "ranging", "trend": "flat", "volatility": "low",
    }
    with patch("paper_trading.engine.run_realtime_pipeline", return_value=fake_result):
        summary = engine.on_bar(df)

    assert summary["executed"] is False
    assert len(engine.portfolio.open_positions) == 0
    assert len(engine.portfolio.equity_curve) == 1


def test_run_replay_bounds_window_size():
    """
    Regression test for the O(n^2) replay bug: on_bar must never be
    called with more than max_window_bars rows, even late in a long
    replay, or run_realtime_pipeline's full indicator recompute cost
    grows unboundedly bar-over-bar.
    """
    df = _synthetic_df(n=600)
    engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h", model=None, feature_columns=[],
        persist_to_db=False,
    )
    seen_window_lengths = []

    def fake_pipeline(df, model, feature_columns, calibrator=None, symbol='UNKNOWN',
                       account_equity=None, open_positions=None, daily_pnl_pct=0.0,
                       correlation_matrix=None, **kwargs):
        seen_window_lengths.append(len(df))
        price = float(df.iloc[-1]["close"])
        return {"prediction": "HOLD", "confidence": 0.1, "market_state": "ranging",
                "trend": "flat", "volatility": "low", "entry": price,
                "stop_loss": None, "take_profit": None, "risk_reward_ratio": None,
                "position_size": None, "risk_pct": None, "notional_value": None,
                "risk_block_reason": None}

    with patch("paper_trading.engine.run_realtime_pipeline", side_effect=fake_pipeline):
        engine.run_replay(df, warmup_bars=250, max_window_bars=300)

    assert max(seen_window_lengths) <= 300
    # and it did keep growing up to the cap before plateauing, not stuck at warmup size
    assert max(seen_window_lengths) > 250


# ---------------------------------------------------------------------
# Global Portfolio Accounting Architecture Tests
# ---------------------------------------------------------------------

def _setup_sqlite_db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                ts TIMESTAMP NOT NULL, prediction TEXT NOT NULL, confidence REAL,
                prob_down REAL, prob_flat REAL, prob_up REAL, market_state TEXT,
                trend TEXT, volatility TEXT, entry_price REAL, stop_loss REAL,
                take_profit REAL, risk_reward_ratio REAL, position_size REAL,
                risk_pct REAL, notional_value REAL, risk_block_reason TEXT,
                executed BOOLEAN NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (exchange, symbol, timeframe, ts)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL, side TEXT NOT NULL, action TEXT NOT NULL,
                ts TIMESTAMP NOT NULL, price REAL NOT NULL, size REAL NOT NULL,
                fee REAL NOT NULL, stop_loss REAL, take_profit REAL, exit_reason TEXT,
                realized_pnl REAL, realized_pnl_pct REAL, bars_held INTEGER,
                cash_after REAL NOT NULL, equity_after REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (trade_id, action)
            )
        """))
    return engine


def test_global_portfolio_state_loader():
    sqlite_eng = _setup_sqlite_db()
    # Log BTC closed trade (+87.77 PnL)
    with sqlite_eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO trade_log (trade_id, exchange, symbol, timeframe, side, action, ts, price, size, fee, realized_pnl, cash_after, equity_after)
            VALUES ('t1', 'binance', 'BTC/USDT', '1h', 'long', 'CLOSE', '2026-09-06 23:00:00', 80000, 0.1, 15, 87.77, 10087.77, 10087.77)
        """))
    
    state = db_logger.load_global_portfolio_state(starting_cash=10000.0, engine=sqlite_eng)
    assert state["total_realized_pnl"] == 87.77
    assert state["available_cash"] == 10087.77
    assert state["global_equity"] == 10087.77


def test_daemon_restart_preserves_cash_when_zero_positions():
    sqlite_eng = _setup_sqlite_db()
    # Log past closed trade (+124.26 PnL)
    with sqlite_eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO trade_log (trade_id, exchange, symbol, timeframe, side, action, ts, price, size, fee, realized_pnl, cash_after, equity_after)
            VALUES ('t1', 'binance', 'ETH/USDT', '1h', 'long', 'CLOSE', '2026-09-06 23:00:00', 2500, 1.0, 10, 124.26, 10124.26, 10124.26)
        """))

    # Engine restarts with zero open positions
    engine = PaperTradingEngine(
        symbol="ETH/USDT", timeframe="1h", model=None, feature_columns=[],
        db_engine=sqlite_eng, persist_to_db=True, starting_cash=10000.0,
    )
    # Cash must NOT reset to 10000.0
    assert engine.portfolio.cash == 10124.26


def test_historical_five_trade_reconciliation_exact_equity():
    sqlite_eng = _setup_sqlite_db()
    trades = [
        ('t1', 'BTC/USDT', 'long', 'CLOSE', 87.772559),
        ('t2', 'ETH/USDT', 'long', 'CLOSE', 124.257748),
        ('t3', 'ETH/USDT', 'long', 'CLOSE', -103.669243),
        ('t4', 'ETH/USDT', 'long', 'CLOSE', -104.181561),
        ('t5', 'ETH/USDT', 'long', 'CLOSE', -49.588724),
    ]
    with sqlite_eng.begin() as conn:
        for idx, (tid, sym, side, act, pnl) in enumerate(trades):
            conn.execute(text(f"""
                INSERT INTO trade_log (trade_id, exchange, symbol, timeframe, side, action, ts, price, size, fee, realized_pnl, cash_after, equity_after)
                VALUES ('{tid}', 'binance', '{sym}', '1h', '{side}', '{act}', '2026-09-0{idx+1} 12:00:00', 100, 1, 1, {pnl}, 10000, 10000)
            """))

    state = db_logger.load_global_portfolio_state(starting_cash=10000.0, engine=sqlite_eng)
    expected_equity = 10000.0 - 45.409222
    assert abs(state["global_equity"] - expected_equity) < 1e-5


def test_concurrent_btc_eth_capital_allocation():
    db_eng = create_engine("sqlite:///file:mem_concurrent?mode=memory&cache=shared", connect_args={"check_same_thread": False, "uri": True})
    with db_eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                ts TIMESTAMP NOT NULL, prediction TEXT NOT NULL, confidence REAL,
                prob_down REAL, prob_flat REAL, prob_up REAL, market_state TEXT,
                trend TEXT, volatility TEXT, entry_price REAL, stop_loss REAL,
                take_profit REAL, risk_reward_ratio REAL, position_size REAL,
                risk_pct REAL, notional_value REAL, risk_block_reason TEXT,
                executed BOOLEAN NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (exchange, symbol, timeframe, ts)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL, side TEXT NOT NULL, action TEXT NOT NULL,
                ts TIMESTAMP NOT NULL, price REAL NOT NULL, size REAL NOT NULL,
                fee REAL NOT NULL, stop_loss REAL, take_profit REAL, exit_reason TEXT,
                realized_pnl REAL, realized_pnl_pct REAL, bars_held INTEGER,
                cash_after REAL NOT NULL, equity_after REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (trade_id, action)
            )
        """))

    btc_engine = PaperTradingEngine(
        symbol="BTC/USDT", timeframe="1h", model=None, feature_columns=[],
        db_engine=db_eng, persist_to_db=True, starting_cash=10000.0
    )
    eth_engine = PaperTradingEngine(
        symbol="ETH/USDT", timeframe="1h", model=None, feature_columns=[],
        db_engine=db_eng, persist_to_db=True, starting_cash=10000.0
    )

    errors = []

    def run_btc():
        try:
            for i in range(10):
                state = db_logger.load_global_portfolio_state(starting_cash=10000.0, engine=db_eng)
                btc_engine.portfolio.cash = state["available_cash"]
                time.sleep(0.002)
        except Exception as e:
            errors.append(e)

    def run_eth():
        try:
            for i in range(10):
                state = db_logger.load_global_portfolio_state(starting_cash=10000.0, engine=db_eng)
                eth_engine.portfolio.cash = state["available_cash"]
                time.sleep(0.002)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_btc)
    t2 = threading.Thread(target=run_eth)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0
    final_state = db_logger.load_global_portfolio_state(starting_cash=10000.0, engine=db_eng)
    assert final_state["available_cash"] == 10000.0
    assert final_state["global_equity"] == 10000.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


