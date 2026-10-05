"""
tests/test_realtime_paper_lifecycle.py
----------------------------------------
Comprehensive lifecycle validation suite for the Real-Time Trade Decision Paper Trading Layer.

Verifies:
1. LONG and SHORT simulated paper trade creation from actionable plans.
2. Strict Setup Gate: Rejection of WATCH, NO TRADE, and NEUTRAL setups from creating simulated positions.
3. Duplicate open position prevention for the same symbol.
4. Live Mark-to-Market (MTM) calculation of unrealized/floating P&L.
5. Auto-closure on Take Profit (TP1) touch (status CLOSED, exit_reason TP1_HIT, realized profit).
6. Auto-closure on Stop Loss (SL) touch (status CLOSED, exit_reason SL_HIT, realized loss).
7. Restart recovery and persistence from JSONL ledger.
8. Analytics and accounting reconciliation:
   - Floating P&L is isolated from realized P&L.
   - Closed win rate is only computed over completed trades (N/A when 0 closed).
   - Equity curve and drawdown tracking.
"""

import os
import json
import pytest
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from experiments.dashboard import (
    load_realtime_journal,
    record_journal_entry,
    compute_journal_analytics,
    post_record_paper_trade,
    INITIAL_CAPITAL
)
from fastapi import HTTPException


@pytest.fixture
def temp_journal(monkeypatch, tmp_path):
    """Fixture providing an isolated temporary journal file for tests."""
    temp_file = str(tmp_path / "test_journal.jsonl")
    monkeypatch.setattr("experiments.dashboard.JOURNAL_FILE", temp_file)
    return temp_file


def test_paper_trade_creation_long(temp_journal):
    """Verify creating a LONG paper trade stores all execution parameters in OPEN status."""
    payload = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 60000.0,
        "stop_loss": 58000.0,
        "take_profit_1": 64000.0,
        "rr": 2.0,
        "units": 0.05,
        "regime": "STRONG BULL",
        "setup": "TREND PULLBACK TO EMA"
    }

    res = post_record_paper_trade(payload)
    assert res["status"] == "SUCCESS"
    assert res["id"].startswith("SIM-")

    # Verify journal content
    journal = load_realtime_journal()
    assert len(journal) == 1
    trade = journal[0]
    assert trade["symbol"] == "BTC/USDT"
    assert trade["direction"] == "LONG"
    assert trade["entry_price"] == 60000.0
    assert trade["stop_loss"] == 58000.0
    assert trade["take_profit_1"] == 64000.0
    assert trade["status"] == "OPEN"
    assert trade["pnl_usd"] == 0.0


def test_paper_trade_creation_short(temp_journal):
    """Verify creating a SHORT paper trade stores all execution parameters in OPEN status."""
    payload = {
        "symbol": "ETH/USDT",
        "direction": "SHORT",
        "entry_price": 3000.0,
        "stop_loss": 3100.0,
        "take_profit_1": 2800.0,
        "rr": 2.0,
        "units": 1.0,
        "regime": "STRONG BEAR",
        "setup": "BREAKOUT + MOMENTUM"
    }

    res = post_record_paper_trade(payload)
    assert res["status"] == "SUCCESS"

    journal = load_realtime_journal()
    assert len(journal) == 1
    trade = journal[0]
    assert trade["symbol"] == "ETH/USDT"
    assert trade["direction"] == "SHORT"
    assert trade["status"] == "OPEN"


def test_neutral_and_watch_blocking(temp_journal):
    """Verify non-actionable signals (NEUTRAL / WATCH / NO TRADE) are strictly rejected."""
    for invalid_dir in ["NEUTRAL", "WATCH", "NO_TRADE", ""]:
        with pytest.raises(HTTPException) as exc_info:
            post_record_paper_trade({
                "symbol": "SOL/USDT",
                "direction": invalid_dir,
                "entry_price": 140.0
            })
        assert exc_info.value.status_code == 400
        assert "Cannot record paper trade" in exc_info.value.detail

    # Journal must remain empty
    assert len(load_realtime_journal()) == 0


def test_duplicate_open_position_blocking(temp_journal):
    """Verify opening a duplicate trade on the same asset is blocked while one is already OPEN."""
    payload = {
        "symbol": "SOL/USDT",
        "direction": "LONG",
        "entry_price": 150.0,
        "stop_loss": 145.0,
        "take_profit_1": 160.0,
        "units": 20.0
    }

    # First trade: SUCCESS
    res1 = post_record_paper_trade(payload)
    assert res1["status"] == "SUCCESS"

    # Second trade on same symbol: BLOCKED
    with pytest.raises(HTTPException) as exc_info:
        post_record_paper_trade(payload)
    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.detail


def test_mark_to_market_floating_pnl(temp_journal):
    """Verify MTM updates unrealized P&L without prematurely closing open positions."""
    entry = {
        "id": "SIM-1001",
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 60000.0,
        "stop_loss": 58000.0,
        "take_profit_1": 64000.0,
        "units": 0.1,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    # Mock DB returning current price = 61500 (high=62000, low=59500)
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (62000.0, 59500.0, 61500.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 1
    assert analytics["closed_trades_count"] == 0
    # Expected unrealized: (61500 - 60000) * 0.1 = +150.0 USD
    assert analytics["unrealized_pnl_usd"] == 150.0
    assert analytics["realized_pnl_usd"] == 0.0
    assert analytics["win_rate_display"] == "N/A"


def test_take_profit_hit_closure_long(temp_journal):
    """Verify LONG position auto-closes with realized profit when candle high reaches TP1."""
    entry = {
        "id": "SIM-1002",
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 60000.0,
        "stop_loss": 58000.0,
        "take_profit_1": 64000.0,
        "units": 0.1,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    # Mock DB returning high=64500 (breaches TP 64000)
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (64500.0, 61000.0, 64200.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 0
    assert analytics["closed_trades_count"] == 1
    assert analytics["wins"] == 1
    assert analytics["losses"] == 0
    assert analytics["win_rate_display"] == "100.0%"
    # Realized PnL: (64000 - 60000) * 0.1 = 400.0 USD
    assert analytics["realized_pnl_usd"] == 400.0

    # Verify journal persisted closure state
    journal = load_realtime_journal()
    assert journal[0]["status"] == "CLOSED"
    assert journal[0]["exit_reason"] == "TP1_HIT"
    assert journal[0]["pnl_usd"] == 400.0


def test_stop_loss_hit_closure_long(temp_journal):
    """Verify LONG position auto-closes with realized loss when candle low touches SL."""
    entry = {
        "id": "SIM-1003",
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 60000.0,
        "stop_loss": 58000.0,
        "take_profit_1": 64000.0,
        "units": 0.1,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    # Mock DB returning low=57500 (touches SL 58000)
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (60500.0, 57500.0, 57800.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 0
    assert analytics["closed_trades_count"] == 1
    assert analytics["wins"] == 0
    assert analytics["losses"] == 1
    assert analytics["win_rate_display"] == "0.0%"
    # Realized PnL: (58000 - 60000) * 0.1 = -200.0 USD
    assert analytics["realized_pnl_usd"] == -200.0

    journal = load_realtime_journal()
    assert journal[0]["status"] == "CLOSED"
    assert journal[0]["exit_reason"] == "SL_HIT"
    assert journal[0]["pnl_usd"] == -200.0


def test_restart_recovery_and_journal_persistence(temp_journal):
    """Verify shutting down and reloading from disk maintains full ledger integrity."""
    entry1 = {
        "id": "SIM-1004",
        "symbol": "BNB/USDT",
        "direction": "LONG",
        "entry_price": 800.0,
        "stop_loss": 760.0,
        "take_profit_1": 880.0,
        "units": 2.5,
        "status": "OPEN",
        "ts": "2026-10-05T01:00:00Z"
    }
    entry2 = {
        "id": "SIM-1005",
        "symbol": "SOL/USDT",
        "direction": "SHORT",
        "entry_price": 150.0,
        "stop_loss": 155.0,
        "take_profit_1": 140.0,
        "units": 20.0,
        "status": "CLOSED",
        "exit_price": 140.0,
        "exit_reason": "TP1_HIT",
        "pnl_usd": 200.0,
        "ts": "2026-10-05T01:30:00Z"
    }
    record_journal_entry(entry1)
    record_journal_entry(entry2)

    # Reload from fresh disk read
    recovered = load_realtime_journal()
    assert len(recovered) == 2
    assert recovered[0]["id"] == "SIM-1004"
    assert recovered[0]["status"] == "OPEN"
    assert recovered[1]["id"] == "SIM-1005"
    assert recovered[1]["status"] == "CLOSED"
    assert recovered[1]["pnl_usd"] == 200.0


def test_take_profit_hit_closure_short(temp_journal):
    """Verify SHORT position auto-closes with realized profit when candle low reaches TP1."""
    entry = {
        "id": "SIM-1006",
        "symbol": "ETH/USDT",
        "direction": "SHORT",
        "entry_price": 3000.0,
        "stop_loss": 3100.0,
        "take_profit_1": 2800.0,
        "units": 1.0,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    # Mock DB low=2750 (breaches TP 2800)
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (2950.0, 2750.0, 2780.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 0
    assert analytics["closed_trades_count"] == 1
    assert analytics["wins"] == 1
    # Realized PnL: (3000 - 2800) * 1.0 = +200.0 USD
    assert analytics["realized_pnl_usd"] == 200.0


def test_stop_loss_hit_closure_short(temp_journal):
    """Verify SHORT position auto-closes with realized loss when candle high touches SL."""
    entry = {
        "id": "SIM-1007",
        "symbol": "ETH/USDT",
        "direction": "SHORT",
        "entry_price": 3000.0,
        "stop_loss": 3100.0,
        "take_profit_1": 2800.0,
        "units": 1.0,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    # Mock DB high=3150 (touches SL 3100)
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (3150.0, 2980.0, 3120.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 0
    assert analytics["closed_trades_count"] == 1
    assert analytics["losses"] == 1
    # Realized PnL: (3000 - 3100) * 1.0 = -100.0 USD
    assert analytics["realized_pnl_usd"] == -100.0


def test_analytics_accounting_reconciliation_zero_closed(temp_journal):
    """Verify analytics metrics cleanly handle pure open positions with 0 closed trades."""
    entry = {
        "id": "SIM-1008",
        "symbol": "SOL/USDT",
        "direction": "LONG",
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit_1": 120.0,
        "units": 10.0,
        "status": "OPEN",
        "ts": "2026-10-05T00:00:00Z"
    }
    record_journal_entry(entry)

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (105.0, 98.0, 104.0)

    with patch("experiments.dashboard.get_engine", return_value=mock_engine):
        analytics = compute_journal_analytics()

    assert analytics["open_positions_count"] == 1
    assert analytics["closed_trades_count"] == 0
    assert analytics["win_rate_display"] == "N/A"
    assert analytics["profit_factor_display"] == "N/A"
    assert analytics["payoff_ratio_display"] == "N/A"
    assert analytics["realized_pnl_usd"] == 0.0
    assert analytics["unrealized_pnl_usd"] == 40.0
    assert analytics["current_equity"] == INITIAL_CAPITAL + 40.0
    assert len(analytics["open_positions"]) == 1
    assert analytics["open_positions"][0]["unrealized_pnl"] == 40.0
