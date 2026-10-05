import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from ml.train import _make_synthetic_ohlcv
from inference.realtime_pipeline import run_realtime_pipeline


def test_realtime_pipeline_execution():
    df = _make_synthetic_ohlcv(n=300, seed=42)
    dummy_model = None
    dummy_features = ["RET_1"]

    with patch("inference.realtime_pipeline.predict") as mock_predict:
        mock_predict.return_value = pd.DataFrame({
            "pred_label": [1] * 20,
            "confidence": [0.85] * 20,
        })

        result = run_realtime_pipeline(
            df=df,
            model=dummy_model,
            feature_columns=dummy_features,
            calibrator=None,
            symbol="BTC/USDT",
            account_equity=10000,
            open_positions=[],
            daily_pnl_pct=0.0,
        )

        assert result is not None
        assert "prediction" in result
        assert result["prediction"] in ["BUY", "HOLD", "SELL"]
        assert "confidence" in result
        assert "stop_loss" in result
        assert "take_profit" in result