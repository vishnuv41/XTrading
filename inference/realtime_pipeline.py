"""
Realtime Inference Pipeline

Pipeline

Raw OHLCV
      ↓
Indicators
      ↓
Market Regime
      ↓
ML Prediction (+ confidence threshold)
      ↓
Risk Engine (stop/target, R:R filter, position size, portfolio gates)
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
from strategy.signal import generate_signals

from risk_engine.stoploss import calculate_stop_loss
from risk_engine.takeprofit import calculate_take_profit
from risk_engine.risk_reward import calculate_risk_reward, meets_minimum_risk_reward
from risk_engine.position_size import calculate_adjusted_position_size
from risk_engine.portfolio_risk import calculate_portfolio_heat, check_new_trade_allowed

from ml.utils.preprocessing import build_feature_matrix
from ml.predict import predict

from config.settings import settings


logger = logging.getLogger(__name__)


def _latest_row(df: pd.DataFrame):
    if len(df) == 0:
        raise ValueError("Empty dataframe.")
    return df.iloc[-1]


def _extract_market_state(df: pd.DataFrame) -> dict:
    row = _latest_row(df)
    return {
        "market_state": row["market_state"],
        "trend": row["trend_regime"],
        "volatility": row["vol_regime"],
    }


def _side_for(prediction: str) -> Optional[str]:
    if prediction == "BUY":
        return "long"
    if prediction == "SELL":
        return "short"
    return None


def _build_response(prediction, confidence, market_state, entry, risk, explanation):
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "market_state": market_state["market_state"],
        "trend": market_state["trend"],
        "volatility": market_state["volatility"],
        "entry": entry,
        "stop_loss": risk.get("stop_loss"),
        "take_profit": risk.get("take_profit"),
        "risk_reward_ratio": risk.get("risk_reward_ratio"),
        "position_size": risk.get("position_size"),
        "risk_pct": risk.get("risk_pct"),
        "notional_value": risk.get("notional_value"),
        "risk_block_reason": risk.get("block_reason"),
        "explanation": explanation,
    }


# --------------------------------------------------------
# Risk Engine wiring
# --------------------------------------------------------

def _apply_risk_engine(
    prediction: str,
    entry_price: float,
    atr: float,
    confidence: float,
    vol_regime: str,
    symbol: str,
    account_equity: float,
    open_positions: list[dict],
    daily_pnl_pct: float,
    correlation_matrix,
) -> tuple[str, dict]:
    """
    Runs the ML-confirmed signal through the full risk engine: stop/target
    placement, minimum R:R, position sizing, and the portfolio-level gates
    (max open trades, daily loss circuit breaker, portfolio heat /
    correlation cap).

    Returns (possibly-downgraded prediction, risk info dict). Any gate
    failure downgrades `prediction` to "HOLD" and sets block_reason.
    """
    risk = {"block_reason": None}
    side = _side_for(prediction)

    if side is None:
        return prediction, risk

    # --- Circuit breaker: daily loss limit -------------------------------
    if daily_pnl_pct <= -settings.risk.max_daily_loss_pct:
        risk["block_reason"] = (
            f"Daily loss limit hit ({daily_pnl_pct:.2%} <= "
            f"-{settings.risk.max_daily_loss_pct:.2%}); no new trades today."
        )
        logger.warning(risk["block_reason"])
        return "HOLD", risk

    # --- Max concurrent open trades ---------------------------------------
    if len(open_positions) >= settings.risk.max_open_trades:
        risk["block_reason"] = (
            f"Max open trades reached ({len(open_positions)}/"
            f"{settings.risk.max_open_trades})."
        )
        logger.warning(risk["block_reason"])
        return "HOLD", risk

    # --- Stop-loss / take-profit -------------------------------------------
    stop_loss = calculate_stop_loss(
        entry_price=entry_price, atr=atr, side=side,
        multiplier=settings.risk.sl_atr_multiplier,
    )
    take_profit = calculate_take_profit(
        entry_price=entry_price, stop_loss=stop_loss, side=side,
        risk_reward_ratio=settings.risk.tp_risk_reward,
    )
    risk["stop_loss"] = stop_loss
    risk["take_profit"] = take_profit
    risk["risk_reward_ratio"] = calculate_risk_reward(entry_price, stop_loss, take_profit)

    # --- Minimum R:R filter --------------------------------------------------
    if not meets_minimum_risk_reward(entry_price, stop_loss, take_profit,
                                      min_ratio=settings.risk.min_risk_reward):
        risk["block_reason"] = (
            f"R:R {risk['risk_reward_ratio']:.2f} below minimum "
            f"{settings.risk.min_risk_reward:.2f}."
        )
        logger.info(risk["block_reason"])
        return "HOLD", risk

    # --- Position sizing (confidence + vol-regime scaled) -------------------
    sizing = calculate_adjusted_position_size(
        account_equity=account_equity, entry_price=entry_price, stop_loss=stop_loss,
        signal_confidence=confidence, vol_regime=vol_regime,
        base_risk_pct=settings.risk.base_risk_pct,
    )
    risk.update(sizing)

    # --- Portfolio exposure gate (heat + correlation) ------------------------
    decision = check_new_trade_allowed(
        open_positions=open_positions, new_symbol=symbol,
        new_risk_pct=sizing["risk_pct"], correlation_matrix=correlation_matrix,
        max_portfolio_heat=settings.risk.max_portfolio_heat,
        max_correlation=settings.risk.correlation_threshold,
        max_correlated_positions=settings.risk.max_correlated_positions,
    )
    risk["portfolio_heat_after_trade"] = decision["heat_after_trade"]

    if not decision["allowed"]:
        risk["block_reason"] = decision["reason"]
        logger.warning(risk["block_reason"])
        return "HOLD", risk

    return prediction, risk


# --------------------------------------------------------
# Main Realtime Pipeline
# --------------------------------------------------------

def run_realtime_pipeline(
    df: pd.DataFrame,
    model,
    feature_columns,
    calibrator=None,
    return_features: bool = False,
    symbol: str = "UNKNOWN",
    account_equity: Optional[float] = None,
    open_positions: Optional[list[dict]] = None,
    daily_pnl_pct: float = 0.0,
    correlation_matrix=None,
):
    """
    New risk-engine-related args:
        symbol: Symbol being traded, used for the portfolio correlation gate.
        account_equity: Live account equity for position sizing. Defaults
            to settings.risk.account_equity if not supplied.
        open_positions: Currently open positions, each a dict with at
            least 'symbol' and 'risk_pct' (see risk_engine/portfolio_risk.py).
            Defaults to [] (no open positions).
        daily_pnl_pct: Today's realized P&L as a fraction of equity
            (negative = loss). Feeds the daily-loss circuit breaker.
        correlation_matrix: Optional pd.DataFrame from
            risk_engine.portfolio_risk.calculate_return_correlation, used
            for the correlated-position cap. If None, only the portfolio
            heat cap is enforced.
    """
    logger.info("Starting realtime inference pipeline...")

    open_positions = open_positions if open_positions is not None else []
    account_equity = account_equity if account_equity is not None else settings.risk.account_equity

    # Step 1: Indicators
    df = calculate_all_indicators(df)

    # Step 2: Market Regime
    df = calculate_market_state(df)
    market_state = _extract_market_state(df)

    # Step 3: Strategy (rule-based signal, kept for the explanation panel —
    # the ML prediction below is what actually drives the trade decision)
    df = generate_signals(df)
    latest = df.iloc[-1]

    # Step 4: Feature Engineering (debug / inspection path only — predict()
    # builds its own feature matrix internally; see note below)
    if return_features:
        logger.info("Building ML features...")
        feature_df = build_feature_matrix(df)
        X = feature_df[feature_columns].dropna()
        if X.empty:
            raise RuntimeError("Feature dataframe became empty.")
        return {"features": X}

    # Step 5: Machine Learning Prediction
    logger.info("Running ML prediction...")
    predictions = predict(df=df, ensemble=model, feature_columns=feature_columns, calibrator=calibrator)
    predictions = predictions.dropna()
    if predictions.empty:
        raise RuntimeError("Prediction dataframe is empty.")

    latest_prediction = predictions.iloc[-1]
    pred_label = int(latest_prediction["pred_label"])
    confidence = float(latest_prediction["confidence"])
    prediction = {1: "BUY", -1: "SELL"}.get(pred_label, "HOLD")

    logger.info(f"Prediction={prediction} Confidence={confidence:.3f}")

    if confidence < settings.ml.confidence_threshold:
        logger.info(f"Confidence {confidence:.3f} below threshold {settings.ml.confidence_threshold:.2f}.")
        prediction = "HOLD"

    # Step 6: Risk Engine
    entry_price = float(latest["close"])
    atr = float(latest["ATR14"])
    vol_regime = market_state["volatility"]

    prediction, risk = _apply_risk_engine(
        prediction=prediction, entry_price=entry_price, atr=atr, confidence=confidence,
        vol_regime=vol_regime, symbol=symbol, account_equity=account_equity,
        open_positions=open_positions, daily_pnl_pct=daily_pnl_pct,
        correlation_matrix=correlation_matrix,
    )

    # Step 7: Explanation Engine
    explanation = []

    if "RSI14" in latest.index:
        if latest["RSI14"] < 30:
            explanation.append("RSI indicates oversold conditions.")
        elif latest["RSI14"] > 70:
            explanation.append("RSI indicates overbought conditions.")

    if "EMA20" in latest.index and "EMA50" in latest.index:
        explanation.append("EMA20 is above EMA50." if latest["EMA20"] > latest["EMA50"] else "EMA20 is below EMA50.")

    if "ADX14" in latest.index:
        explanation.append("Strong market trend detected." if latest["ADX14"] > 25 else "Weak market trend.")

    if "MACD" in latest.index:
        explanation.append("MACD momentum considered.")

    if "OBV" in latest.index:
        explanation.append("Volume confirmation applied.")

    if risk.get("block_reason"):
        explanation.append(f"Risk engine: {risk['block_reason']}")
    elif risk.get("position_size") is not None:
        explanation.append(
            f"Position Size: {risk['position_size']:.4f} "
            f"(risk {risk['risk_pct']:.2%}, R:R {risk['risk_reward_ratio']:.2f})"
        )

    logger.info("Realtime inference completed successfully.")

    return _build_response(
        prediction=prediction, confidence=confidence, market_state=market_state,
        entry=entry_price, risk=risk, explanation=explanation,
    )


# --------------------------------------------------------
# Local Smoke Test
# --------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.uniform(0.1, 1.2, n),
        "low": close - rng.uniform(0.1, 1.2, n),
        "close": close,
        "volume": rng.integers(100, 1000, n),
    })

    print("=" * 60)
    print("Realtime Pipeline Smoke Test")
    print("=" * 60)
    print("\nNOTE\nThis file requires: Trained Ensemble Model, Feature Columns,")
    print("Strategy Engine, Risk Engine, Indicators, Regime.\n")
    print("Import Test Passed.\n")
    print("Example:\n")
    print("""
from inference.realtime_pipeline import run_realtime_pipeline

result = run_realtime_pipeline(
    df, model, feature_columns, calibrator,
    symbol="BTC/USDT", account_equity=10000,
    open_positions=[], daily_pnl_pct=0.0,
)
print(result)
""")
    print("=" * 60) 