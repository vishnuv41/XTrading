"""
tests/test_simultaneous_signals.py
------------------------------------
Unit test verifying portfolio behavior under simultaneous signals across multiple assets (BTC/USDT and ETH/USDT).
"""
import unittest
from datetime import datetime, timezone
import sys

sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from paper_trading.portfolio import VirtualPortfolio, DuplicatePositionError
from paper_trading.models import OpenPosition


class TestSimultaneousSignals(unittest.TestCase):

    def setUp(self):
        self.portfolio = VirtualPortfolio(starting_cash=10000.0)

    def test_concurrent_btc_and_eth_positions(self):
        btc_pos = OpenPosition(
            trade_id="btc-trade-001",
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc),
            entry_price=60000.0,
            size=0.1,  # notional: 6000 USD
            stop_loss=59100.0,
            take_profit=61800.0,
            risk_pct=0.01,
            entry_fee=6.0,
        )

        eth_pos = OpenPosition(
            trade_id="eth-trade-001",
            exchange="binance",
            symbol="ETH/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc),
            entry_price=2500.0,
            size=1.0,  # notional: 2500 USD
            stop_loss=2425.0,
            take_profit=2650.0,
            risk_pct=0.01,
            entry_fee=2.5,
        )

        self.portfolio.open_position(btc_pos)
        self.portfolio.open_position(eth_pos)

        self.assertTrue(self.portfolio.has_open_position("BTC/USDT"))
        self.assertTrue(self.portfolio.has_open_position("ETH/USDT"))
        self.assertEqual(len(self.portfolio.open_positions), 2)

        # Mark to market with concurrent prices
        prices = {"BTC/USDT": 60500.0, "ETH/USDT": 2550.0}
        unrealized = self.portfolio.unrealized_pnl(prices)
        # BTC PnL: 0.1 * 500 = +50.0; ETH PnL: 1.0 * 50 = +50.0 => total +100.0
        self.assertAlmostEqual(unrealized, 100.0, places=2)

        total_eq = self.portfolio.equity(prices)
        self.assertAlmostEqual(total_eq, 10000.0 - 6.0 - 2.5 + 100.0, places=2)

    def test_duplicate_position_prevention(self):
        btc_pos1 = OpenPosition(
            trade_id="btc-trade-001",
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc),
            entry_price=60000.0,
            size=0.1,
            stop_loss=59100.0,
            take_profit=61800.0,
            risk_pct=0.01,
            entry_fee=6.0,
        )
        btc_pos2 = OpenPosition(
            trade_id="btc-trade-002",
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            side="long",
            entry_ts=datetime(2026, 9, 7, 13, 0, tzinfo=timezone.utc),
            entry_price=60200.0,
            size=0.1,
            stop_loss=59300.0,
            take_profit=62000.0,
            risk_pct=0.01,
            entry_fee=6.0,
        )

        self.portfolio.open_position(btc_pos1)
        with self.assertRaises(DuplicatePositionError):
            self.portfolio.open_position(btc_pos2)


if __name__ == "__main__":
    unittest.main()
