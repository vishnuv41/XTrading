"""
tests/test_data_integrity_nan_handling.py
---------------------------------------------
Unit test verifying pipeline data integrity when encountering NaNs, nulls, or infinite values in OHLCV stream.
"""
import math
import unittest
from datetime import datetime, timezone
import sys

sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from preprocessing.validator import validate_ohlcv
from preprocessing.cleaner import clean_ohlcv


class TestDataIntegrityNanHandling(unittest.TestCase):

    def test_validate_ohlcv_filters_nan_and_inf(self):
        valid_row = {
            "timestamp": datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0
        }
        nan_close_row = {
            "timestamp": datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": float("nan"), "volume": 10.0
        }
        inf_high_row = {
            "timestamp": datetime(2026, 9, 7, 2, 0, tzinfo=timezone.utc),
            "open": 100.0, "high": float("inf"), "low": 95.0, "close": 102.0, "volume": 10.0
        }
        none_vol_row = {
            "timestamp": datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": None
        }

        input_batch = [valid_row, nan_close_row, inf_high_row, none_vol_row]
        cleaned = validate_ohlcv(input_batch)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["open"], 100.0)
        self.assertEqual(cleaned[0]["close"], 102.0)

    def test_cleaner_handles_invalid_data_without_exception(self):
        ts = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)
        zero_range_zero_vol = {
            "exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1h",
            "ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0
        }
        valid_row = {
            "exchange": "binance", "symbol": "BTC/USDT", "timeframe": "1h",
            "ts": ts, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0
        }
        
        result = clean_ohlcv([zero_range_zero_vol, valid_row])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["close"], 102.0)


if __name__ == "__main__":
    unittest.main()
