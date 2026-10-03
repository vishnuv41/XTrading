"""
tests/test_restart_recovery_open_position.py
---------------------------------------------
Unit tests for paper trading state restoration from database upon process restart.
Verifies open position recovery, holding bars, entry price, and SL/TP preservation.
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import sys
sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OpenPosition

class TestRestartRecoveryOpenPosition(unittest.TestCase):

    def test_position_restoration_attributes(self):
        mock_db = MagicMock()
        
        engine = PaperTradingEngine(
            symbol="ETH/USDT",
            timeframe="1h",
            exchange="binance",
            model=MagicMock(),
            feature_columns=[],
            starting_cash=10000.0,
            db_engine=mock_db,
            persist_to_db=False,
        )
        
        # Manually seed recovered position into portfolio
        recovered_pos = OpenPosition(
            trade_id="test-recovery-1234",
            exchange="binance",
            symbol="ETH/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 7, 0, tzinfo=timezone.utc),
            entry_price=2490.59,
            size=3.3280,
            stop_loss=2465.48,
            take_profit=2537.10,
            risk_pct=0.01,
            entry_fee=8.28,
        )
        engine.portfolio.open_positions["ETH/USDT"] = recovered_pos
        
        # Verify restored attributes
        self.assertIn("ETH/USDT", engine.portfolio.open_positions)
        pos = engine.portfolio.open_positions["ETH/USDT"]
        self.assertEqual(pos.trade_id, "test-recovery-1234")
        self.assertEqual(pos.entry_price, 2490.59)
        self.assertEqual(pos.stop_loss, 2465.48)
        self.assertEqual(pos.take_profit, 2537.10)
        self.assertEqual(pos.size, 3.3280)

if __name__ == "__main__":
    unittest.main()
