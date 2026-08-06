"""
Quick smoke test for indicators/calculate_all_indicators.
Run: python -m tests.test_indicators   (from the person2/ folder)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from indicators import calculate_all_indicators


def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.random(n) * 1.5
    low = close - rng.random(n) * 1.5
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100, 5000, n)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    # guard against any inverted high/low from the random construction above
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


def main():
    df = make_synthetic_ohlcv()
    df = calculate_all_indicators(df)

    print(f"Shape after indicators: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nLast 3 rows:")
    print(df.tail(3).to_string())

    # --- Sanity checks: presence ---
    # trend
    assert "EMA20" in df.columns
    assert "SMA20" in df.columns
    assert "MACD" in df.columns
    assert "ADX14" in df.columns
    assert "supertrend" in df.columns
    assert "supertrend_direction" in df.columns
    assert "ichimoku_tenkan" in df.columns
    assert "ichimoku_senkou_a" in df.columns

    # momentum
    assert "RSI14" in df.columns
    assert "stoch_k" in df.columns
    assert "stoch_d" in df.columns
    assert "cci" in df.columns
    assert "williams_r" in df.columns

    # volatility
    assert "ATR14" in df.columns
    assert "BB_upper" in df.columns
    assert "keltner_mid" in df.columns
    assert "donchian_upper" in df.columns
    assert "donchian_width" in df.columns

    # volume
    assert "VWAP" in df.columns
    assert "OBV" in df.columns
    assert "CMF20" in df.columns
    assert "MFI14" in df.columns

    # --- Sanity checks: bounds ---
    rsi_valid = df["RSI14"].dropna()
    assert rsi_valid.between(0, 100).all(), "RSI out of bounds!"

    assert df["stoch_k"].dropna().between(0, 100).all(), "Stochastic %K out of bounds!"
    assert df["stoch_d"].dropna().between(0, 100).all(), "Stochastic %D out of bounds!"

    assert df["williams_r"].dropna().between(-100, 0).all(), "Williams %R out of bounds!"

    assert df["MFI14"].dropna().between(0, 100).all(), "MFI out of bounds!"

    assert df["CMF20"].dropna().between(-1.01, 1.01).all(), "CMF out of expected bounds!"

    assert set(df["supertrend_direction"].dropna().unique()).issubset({1, -1}), (
        "Supertrend direction invalid!"
    )

    assert (df["donchian_upper"].dropna() >= df["donchian_lower"].dropna()).all(), (
        "Donchian upper below lower!"
    )

    print("\nAll checks passed.")


def test_indicators_smoke():
    """pytest entry point — wraps main() so `pytest tests/` discovers this."""
    main()


if __name__ == "__main__":
    main()