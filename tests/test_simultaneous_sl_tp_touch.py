"""
tests/test_simultaneous_sl_tp_touch.py
---------------------------------------
Unit test verifying barrier priority when a single 1H candle's high/low
touches both Stop Loss and Take Profit levels simultaneously.
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import sys
sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OpenPosition

from paper_trading.exit_manager import check_exit, ExitDecision

class TestSimultaneousSlTpTouch(unittest.TestCase):

    def test_stop_loss_priority_on_extreme_candle(self):
        pos = OpenPosition(
            trade_id="extreme-bar-999",
            exchange="binance",
            symbol="ETH/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc),
            entry_price=2500.0,
            size=1.0,
            stop_loss=2450.0,
            take_profit=2550.0,
            risk_pct=0.01,
            entry_fee=2.5,
        )
        
        # Simulate extreme candle where Low=2400 (< SL 2450) and High=2600 (> TP 2550)
        high = 2600.0
        low = 2400.0
        
        decision = check_exit(pos, bar_high=high, bar_low=low, timeout_bars=48)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "stop_loss")
        self.assertEqual(decision.fill_price, 2450.0)

if __name__ == "__main__":
    unittest.main()
