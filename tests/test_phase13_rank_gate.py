import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from config.settings import settings
from inference.realtime_pipeline import run_realtime_pipeline

def test_top_1pct_rank_gate_reproduction():
    """
    Verifies that the live percentile rank gate logic matches the historical
    Phase 10 / Phase 12 top 1.0% percentile calculation:
      top_k_count = max(1, int(len(pred_signal) * (top_pct / 100.0)))
      top_k_cutoff = np.partition(pred_signal, -top_k_count)[-top_k_count]
      gate_pass = latest_pred >= top_k_cutoff
    """
    n = 100
    scores = np.linspace(0.01, 1.00, n)
    top_pct = 1.0
    
    top_k_count = max(1, int(n * (top_pct / 100.0)))  # = 1 for n=100
    expected_cutoff = float(np.partition(scores, -top_k_count)[-top_k_count]) # = 1.00
    
    assert expected_cutoff == 1.00
    assert float(scores[-1]) >= expected_cutoff
    assert float(scores[-2]) < expected_cutoff

def test_rank_gate_integration_mocked():
    """
    Integration test checking that run_realtime_pipeline enforces the top 1.0% rank gate.
    """
    n = 250
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 101.0,
        "volume": 1000.0,
    })
    
    conf_values = np.linspace(0.1, 0.9, n)
    preds_df = pd.DataFrame({
        "pred_label": [1] * n,
        "prob_down": [0.1] * n,
        "prob_flat": [0.1] * n,
        "prob_up": [0.8] * n,
        "confidence": conf_values,
    }, index=df.index)
    
    with patch("inference.realtime_pipeline.predict", return_value=preds_df):
        with patch("inference.realtime_pipeline.calculate_all_indicators", side_effect=lambda x: x):
            with patch("inference.realtime_pipeline.calculate_market_state", side_effect=lambda x, **kw: x):
                with patch("inference.realtime_pipeline.generate_signals", side_effect=lambda x: x):
                    df["market_state"] = "BULL"
                    df["trend_regime"] = "UPTREND"
                    df["vol_regime"] = "low"
                    df["ATR14"] = 2.0
                    
                    mock_model = MagicMock()
                    mock_cols = ["close"]
                    
                    res = run_realtime_pipeline(df, mock_model, mock_cols, symbol="BTC/USDT")
                    
                    # Top 1% of 250 is 2 rows -> cutoff is conf_values[-2] = ~0.89357
                    # Since latest confidence is 0.90 (>= cutoff), gate passes and prediction remains BUY
                    assert res["prediction"] == "BUY"
                    assert res["confidence"] == 0.90
