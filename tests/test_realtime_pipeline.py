from unittest.mock import patch
import pandas as pd

from pipeline.data_loader import load_ohlcv
from ml.predict import load_training_artifacts
from inference.realtime_pipeline import run_realtime_pipeline

# Load sample data
df = load_ohlcv("BTC/USDT", "1h")

# Load trained model
artifacts = load_training_artifacts("models_artifacts/BTCUSDT_1h")

# Mock the prediction step
with patch("inference.realtime_pipeline.predict") as mock_predict:
    mock_predict.return_value = pd.DataFrame({
        "pred_label": [1],      # BUY
        "confidence": [0.85],   # Above threshold
    })

    result = run_realtime_pipeline(
        df=df,
        model=artifacts["ensemble"],
        feature_columns=artifacts["feature_columns"],
        calibrator=artifacts["calibrator"],
        symbol="BTC/USDT",
        account_equity=10000,
        open_positions=[],
        daily_pnl_pct=0.0,
    )

print("\n========== RESULT ==========\n")
for k, v in result.items():
    print(f"{k:20}: {v}")