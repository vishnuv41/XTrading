"""
Realtime Inference Pipeline

This module orchestrates the complete prediction pipeline.

Pipeline

Raw OHLCV
      ↓
Indicators
      ↓
Market Regime
      ↓
Trading Strategy
      ↓
Risk Engine
      ↓
Feature Engineering
      ↓
ML Prediction
      ↓
Probability Calibration
      ↓
Final Trading Signal

Person 2
Quantitative Trading & AI
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from indicators import calculate_all_indicators

from regime import calculate_market_state

from strategy.signal import generate_signal

from risk_engine.stoploss import calculate_stop_loss
from risk_engine.takeprofit import calculate_take_profit
from risk_engine.position_size import calculate_position_size

from ml.utils.preprocessing import build_feature_matrix

from ml.predict import predict


logger = logging.getLogger(__name__)


def _latest_row(df: pd.DataFrame):

    if len(df) == 0:
        raise ValueError("Empty dataframe.")

    return df.iloc[-1]


def _extract_market_state(df):

    row = _latest_row(df)

    return {

        "market_state": row["MARKET_STATE"],

        "trend": row["TREND_REGIME"],

        "volatility": row["VOLATILITY_REGIME"]

    }


def _build_response(

    prediction,

    confidence,

    market_state,

    entry,

    stop_loss,

    take_profit,

    explanation

):

    return {

        "prediction": prediction,

        "confidence": float(confidence),

        "market_state": market_state["market_state"],

        "trend": market_state["trend"],

        "volatility": market_state["volatility"],

        "entry": entry,

        "stop_loss": stop_loss,

        "take_profit": take_profit,

        "explanation": explanation

    }
# --------------------------------------------------------
# Main Realtime Pipeline
# --------------------------------------------------------

def run_realtime_pipeline(
    df: pd.DataFrame,
    model,
    feature_columns,
    calibrator=None,
    return_features: bool = False,
):

    logger.info("Starting realtime inference pipeline...")

    # ----------------------------------------------------
    # Step 1 : Indicators
    # ----------------------------------------------------

    logger.info("Calculating indicators...")

    df = calculate_all_indicators(df)

    # ----------------------------------------------------
    # Step 2 : Market Regime
    # ----------------------------------------------------

    logger.info("Detecting market regime...")

    df = calculate_market_state(df)

    market_state = _extract_market_state(df)

    # ----------------------------------------------------
    # Step 3 : Trading Strategy
    # ----------------------------------------------------

    logger.info("Generating trading signal...")

    signal = generate_signal(df)

    latest = df.iloc[-1]

    # ----------------------------------------------------
    # Step 4 : Risk Engine
    # ----------------------------------------------------

    entry_price = float(latest["close"])

    stop_loss = calculate_stop_loss(

        df,

        signal

    )

    take_profit = calculate_take_profit(

        df,

        signal,

        stop_loss

    )

    position_size = calculate_position_size(

        account_balance=10000,

        risk_percent=1,

        entry_price=entry_price,

        stop_loss=stop_loss

    )

    logger.info(

        f"Entry={entry_price:.2f} "

        f"SL={stop_loss:.2f} "

        f"TP={take_profit:.2f}"

    )

    # ----------------------------------------------------
    # Step 5 : Feature Engineering (debug / inspection path only)
    # ----------------------------------------------------
    #
    # `predict()` below builds its own feature matrix internally from raw
    # `df` (via ml.utils.preprocessing.build_feature_matrix). Previously
    # this step built the feature matrix here too and handed the *already
    # -featurized, already-dropna'd* result to predict(), which then ran
    # build_feature_matrix a second time on top of it. Besides being
    # wasteful, that was a correctness risk: dropna() here strips the
    # warm-up rows before the second pass, so rolling indicators
    # (EMA/ATR/etc.) inside predict()'s recompute would restart from a
    # truncated series instead of the true full history, silently
    # skewing values near the start of what's left. Only build features
    # here when explicitly asked for via return_features (a debug/
    # inspection path) — the real prediction path lets predict() do it
    # once, on the untouched `df`.

    if return_features:

        logger.info("Building ML features...")

        feature_df = build_feature_matrix(df)
        X = feature_df[feature_columns].dropna()

        if X.empty:

            raise RuntimeError(

                "Feature dataframe became empty."

            )

        return {

            "features": X

        }

    # --------------------------------------------------------
# Step 6 : Machine Learning Prediction
# --------------------------------------------------------

    logger.info("Running ML prediction...")

    predictions = predict(

        df=df,

        ensemble=model,

        feature_columns=feature_columns,

        calibrator=calibrator

    )

    predictions = predictions.dropna()

    if predictions.empty:

        raise RuntimeError(

            "Prediction dataframe is empty."

        )

    latest_prediction = predictions.iloc[-1]

    pred_label = int(latest_prediction["pred_label"])

    confidence = float(latest_prediction["confidence"])

    # --------------------------------------------------------
    # Convert numerical prediction to signal
    # --------------------------------------------------------

    if pred_label == 1:

        prediction = "BUY"

    elif pred_label == -1:

        prediction = "SELL"

    else:

        prediction = "HOLD"

    logger.info(

        f"Prediction={prediction} "

        f"Confidence={confidence:.3f}"

    )

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    if confidence < 0.40:

        prediction = "HOLD"

        logger.info(

            "Confidence below threshold."

        )

    # --------------------------------------------------------
    # Explanation Engine
    # --------------------------------------------------------

    explanation = []

    if "RSI14" in latest.index:

        if latest["RSI14"] < 30:

            explanation.append(

                "RSI indicates oversold conditions."

            )

        elif latest["RSI14"] > 70:

            explanation.append(

                "RSI indicates overbought conditions."

            )

    if "EMA20" in latest.index and "EMA50" in latest.index:

        if latest["EMA20"] > latest["EMA50"]:

            explanation.append(

                "EMA20 is above EMA50."

            )

        else:

            explanation.append(

                "EMA20 is below EMA50."

            )

    if "ADX14" in latest.index:

        if latest["ADX14"] > 25:

            explanation.append(

                "Strong market trend detected."

            )

        else:

            explanation.append(

                "Weak market trend."

            )

    if "MACD" in latest.index:

        explanation.append(

            "MACD momentum considered."

        )

    if "OBV" in latest.index:

        explanation.append(

            "Volume confirmation applied."

        )

    explanation.append(

        f"Position Size: {position_size:.4f}"

    )

    logger.info("Prediction complete.")
        # --------------------------------------------------------
    # Final Response
    # --------------------------------------------------------

    response = _build_response(

        prediction=prediction,

        confidence=confidence,

        market_state=market_state,

        entry=entry_price,

        stop_loss=stop_loss,

        take_profit=take_profit,

        explanation=explanation

    )

    logger.info("Realtime inference completed successfully.")

    return response


# --------------------------------------------------------
# Local Smoke Test
# --------------------------------------------------------

if __name__ == "__main__":

    import numpy as np

    # -------------------------
    # Create synthetic OHLCV
    # -------------------------

    rng = np.random.default_rng(42)

    n = 300

    close = 100 + rng.normal(0, 1, n).cumsum()

    df = pd.DataFrame({

        "open": close + rng.normal(0, 0.3, n),

        "high": close + rng.uniform(0.1, 1.2, n),

        "low": close - rng.uniform(0.1, 1.2, n),

        "close": close,

        "volume": rng.integers(

            100,

            1000,

            n

        )

    })

    print("=" * 60)

    print("Realtime Pipeline Smoke Test")

    print("=" * 60)

    print()

    print("NOTE")

    print("This file requires:")

    print("• Trained Ensemble Model")

    print("• Feature Columns")

    print("• Strategy Engine")

    print("• Risk Engine")

    print("• Indicators")

    print("• Regime")

    print()

    print("Import Test Passed.")

    print()

    print("Example:")

    print()

    print(

        """

from inference.realtime_pipeline import run_realtime_pipeline

result = run_realtime_pipeline(

    df,

    model,

    feature_columns,

    calibrator

)

print(result)

"""

    )

    print("=" * 60)