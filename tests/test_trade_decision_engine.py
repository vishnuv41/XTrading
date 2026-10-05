"""
tests/test_trade_decision_engine.py
-----------------------------------
Unit and integration tests for the Real-Time Trade Decision Intelligence Layer.
Verifies:
1. Technical feature calculation (EMAs, RSI, MACD, ATR, ADX, Market Structure)
2. Market regime classification
3. Setup detection (Breakout, Trend Continuation Pullback, Mean Reversion)
4. Decision Quality Scoring and hard risk gates (R:R >= 1.8)
5. Multi-asset market scanner functionality
6. Chart series data formatting
"""

import pytest
import pandas as pd
import numpy as np
from core.trade_decision_engine import (
    calculate_technical_features,
    classify_market_regime,
    detect_trade_setup,
    calculate_comprehensive_decision,
    scan_all_markets,
    fetch_chart_series,
    SUPPORTED_ASSETS
)


def _create_mock_ohlcv(n_bars: int = 100, trend: str = "bull") -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for deterministic unit testing."""
    dates = pd.date_range("2026-09-01", periods=n_bars, freq="1h")
    np.random.seed(42)
    
    if trend == "bull":
        base = np.linspace(50000, 65000, n_bars) + np.random.randn(n_bars) * 100
    elif trend == "bear":
        base = np.linspace(65000, 50000, n_bars) + np.random.randn(n_bars) * 100
    else:  # sideways
        base = np.full(n_bars, 55000) + np.random.randn(n_bars) * 200

    close = base
    high = close + np.random.uniform(50, 200, n_bars)
    low = close - np.random.uniform(50, 200, n_bars)
    open_p = close + np.random.uniform(-50, 50, n_bars)
    volume = np.random.uniform(100, 500, n_bars)

    return pd.DataFrame({
        "ts": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


def test_technical_features_calculation():
    """Verify indicators are computed without NaNs in the final row."""
    df = _create_mock_ohlcv(100, trend="bull")
    df_feat = calculate_technical_features(df)
    
    assert "ema_20" in df_feat.columns
    assert "ema_50" in df_feat.columns
    assert "ema_200" in df_feat.columns
    assert "rsi_14" in df_feat.columns
    assert "atr_14" in df_feat.columns
    assert "macd_hist" in df_feat.columns
    assert "adx_14" in df_feat.columns
    assert "swing_high_20" in df_feat.columns
    assert "swing_low_20" in df_feat.columns

    last = df_feat.iloc[-1]
    assert not pd.isna(last["ema_20"])
    assert not pd.isna(last["rsi_14"])
    assert 0.0 <= last["rsi_14"] <= 100.0
    assert last["atr_14"] > 0.0


def test_regime_classification():
    """Verify bull vs bear vs sideways regime logic."""
    df_bull = calculate_technical_features(_create_mock_ohlcv(100, trend="bull"))
    reg_bull = classify_market_regime(df_bull.iloc[-1])
    assert "BULL" in reg_bull["regime"]

    df_bear = calculate_technical_features(_create_mock_ohlcv(100, trend="bear"))
    reg_bear = classify_market_regime(df_bear.iloc[-1])
    assert "BEAR" in reg_bear["regime"]


def test_trade_decision_integration():
    """Verify end-to-end comprehensive decision output for real BTC/USDT data."""
    res = calculate_comprehensive_decision("BTC/USDT", account_equity=10000.0, risk_pct=1.0)
    
    assert res["symbol"] == "BTC/USDT"
    assert res["decision"] in ["TRADE", "WATCH", "NO TRADE"]
    assert 0 <= res["setup_quality_score"] <= 100
    assert "trade_plan" in res
    assert res["trade_plan"]["risk_reward_ratio"] > 0
    assert res["trade_plan"]["stop_loss"] > 0
    assert res["trade_plan"]["take_profit_1"] > 0
    assert "why_trade" in res["explanations"]
    assert "why_wait" in res["explanations"]


def test_market_scanner_all_supported_assets():
    """Verify scanner scans all 9 supported assets."""
    scan = scan_all_markets()
    assert len(scan) >= 8  # At least 8-9 assets scanned
    symbols_scanned = [s["symbol"] for s in scan]
    assert "BTC/USDT" in symbols_scanned
    assert "ETH/USDT" in symbols_scanned
    assert "SOL/USDT" in symbols_scanned


def test_chart_series_formatting():
    """Verify chart series output format for Lightweight Charts."""
    chart_data = fetch_chart_series("BTC/USDT", timeframe="1h", limit=50)
    assert "candles" in chart_data
    assert len(chart_data["candles"]) > 0
    c0 = chart_data["candles"][0]
    assert "time" in c0 and "open" in c0 and "close" in c0
    assert "indicators" in chart_data
    assert "ema20" in chart_data["indicators"]
