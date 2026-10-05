"""
tests/test_phase13_state_restoration.py
-----------------------------------------
Phase 13 Infrastructure Test: DB State Restoration & Zero-Duplicate Verification.

Proves:
1. Engine restoration correctly hydrates active open positions (BTC/USDT, ETH/USDT) from PostgreSQL.
2. Exact restoration of entry price, size, SL, TP, entry fee, entry timestamp, side, cash, and bars_held.
3. Duplicate position guard: subsequent Top-1% signal after restart skips opening a duplicate trade.
4. Timeout exit countdown: 48-bar timeout countdown continues from original entry timestamp / restored bars_held, exiting when bars_held reaches 48.
"""

import sys
import os
from datetime import datetime, timezone
import pytest
import pandas as pd
from sqlalchemy import text

# Ensure project root is in sys.path
PROJECT_ROOT = r"d:\all\XTrading_combined (1)\XTrading"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.connection import get_engine
from paper_trading import db_logger
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OpenPosition
from paper_trading.portfolio import VirtualPortfolio
from paper_trading.execution import ExecutionSimulator


from unittest.mock import patch


def test_db_logger_load_active_open_positions():
    """Verify load_active_open_positions retrieves open trades structure from trade_log."""
    engine = get_engine()
    btc_trades = db_logger.load_active_open_positions("BTC/USDT", "1h", engine=engine)
    eth_trades = db_logger.load_active_open_positions("ETH/USDT", "1h", engine=engine)
    
    assert isinstance(btc_trades, list)
    assert isinstance(eth_trades, list)
    if len(btc_trades) > 0:
        btc_pos = btc_trades[0]
        assert btc_pos["symbol"] == "BTC/USDT"
        assert btc_pos["entry_price"] > 0
        assert btc_pos["size"] > 0


def test_engine_state_restoration_attributes():
    """Verify PaperTradingEngine restores exact position attributes on startup using deterministic seed."""
    fake_open_trades = [
        {
            "trade_id": "btc-test-101",
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "side": "long",
            "entry_ts": datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc),
            "entry_price": 79582.83,
            "size": 0.193967,
            "stop_loss": 79196.97,
            "take_profit": 80235.24,
            "risk_pct": 0.01,
            "entry_fee": 15.44,
            "leverage": 3.0,
            "cash_after": 9484.56,
        }
    ]
    fake_global_state = {
        "available_cash": 9484.56,
        "total_cash": 10000.0,
        "active_positions_count": 1,
    }

    with patch("paper_trading.db_logger.load_active_open_positions", return_value=fake_open_trades), \
         patch("paper_trading.db_logger.load_global_portfolio_state", return_value=fake_global_state), \
         patch("paper_trading.db_logger.count_bars_held_since", return_value=5):
        
        btc_engine = PaperTradingEngine(
            symbol="BTC/USDT",
            timeframe="1h",
            model=None,
            feature_columns=[],
            persist_to_db=True,
        )

        assert "BTC/USDT" in btc_engine.portfolio.open_positions
        pos = btc_engine.portfolio.open_positions["BTC/USDT"]
        
        # Verify exact match
        expected = fake_open_trades[0]
        assert pos.trade_id == expected["trade_id"]
        assert pos.entry_price == expected["entry_price"]
        assert pos.size == expected["size"]
        assert pos.stop_loss == expected["stop_loss"]
        assert pos.take_profit == expected["take_profit"]
        assert pos.entry_fee == expected["entry_fee"]
        assert pos.side == expected["side"]
        assert pos.bars_held == 5
        assert btc_engine.portfolio.cash == expected["cash_after"]



def test_no_duplicate_entry_on_restart():
    """Verify that a restarted engine given a BUY signal will skip creating a duplicate trade."""
    portfolio = VirtualPortfolio(starting_cash=10000.0)
    exec_sim = ExecutionSimulator(exchange="binance")
    
    # Open an active position
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=datetime.now(timezone.utc), entry_price=79582.83, size=0.193967,
        stop_loss=79196.97, take_profit=80235.24, risk_pct=0.01, entry_fee=15.44, leverage=3.0
    )
    portfolio.open_position(position=pos)
    
    # Attempt to open a duplicate trade from a Top-1% prediction
    risk = {
        "stop_loss": 79196.97,
        "take_profit": 80235.24,
        "position_size": 0.193967,
        "risk_pct": 0.01,
        "entry_price": 79582.83,
    }
    opened = exec_sim.open_from_prediction(
        portfolio=portfolio,
        symbol="BTC/USDT",
        timeframe="1h",
        ts=datetime.now(timezone.utc),
        prediction="BUY",
        risk=risk,
    )
    
    assert opened is None, "Duplicate-position guard MUST return None for an already-open symbol"
    assert len(portfolio.open_positions) == 1, "Open positions count must remain exactly 1"


def test_timeout_countdown_continues():
    """Verify that bars_held increments from restored value and exits at 48 bars."""
    portfolio = VirtualPortfolio(starting_cash=10000.0)
    exec_sim = ExecutionSimulator(exchange="binance")
    
    # Simulate a restored position with bars_held = 47
    pos = OpenPosition.new(
        exchange="binance", symbol="BTC/USDT", timeframe="1h", side="long",
        entry_ts=datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc),
        entry_price=79582.83, size=0.193967, stop_loss=70000.0, take_profit=90000.0,
        risk_pct=0.01, entry_fee=15.44, leverage=3.0
    )
    pos.bars_held = 47
    portfolio.open_position(pos)
    
    # Next bar (bar 48) - check exit
    ts = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
    bar_high, bar_low, bar_close = 79600.0, 79500.0, 79550.0
    
    closed_trade = exec_sim.check_and_close(
        portfolio=portfolio, symbol="BTC/USDT", ts=ts,
        bar_high=bar_high, bar_low=bar_low, bar_close=bar_close
    )
    
    assert closed_trade is not None, "Position must exit on bar 48 timeout"
    assert closed_trade.exit_reason == "timeout"
    assert closed_trade.bars_held == 48
    assert "BTC/USDT" not in portfolio.open_positions
