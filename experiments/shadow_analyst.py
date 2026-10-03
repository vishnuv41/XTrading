from __future__ import annotations
"""
experiments/shadow_analyst.py
---------------------------------
Mode B: Read-Only Shadow Market Intelligence Layer.
Evaluates dual-direction (LONG vs SHORT) technical market scenarios.

AUTHORITATIVE INVARIANTS:
1. Reads Phase 13 predictions & percentile ranks DIRECTLY from prediction_log table.
   NEVER re-evaluates or re-calculates Phase 13 predictions independently.
2. Fails closed (raises ValueError) if market history or required indicators are missing.
3. Explicitly labels scores as HEURISTIC TECHNICAL SCENARIOS (0-100), not ML probabilities.
4. STRICTLY READ-ONLY / NON-EXECUTING — zero database writes or strategy mutations.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.data_loader import load_ohlcv
from database.connection import get_engine

PHASE13_GATE_THRESHOLD = 0.9900  # Top-1.0% required

def fetch_authoritative_phase13_prediction(symbol: str, timeframe: str = "1h") -> dict:
    """
    Reads Phase 13's authoritative prediction and percentile rank directly from prediction_log.
    """
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, ts, prediction, confidence, executed, created_at
            FROM prediction_log
            WHERE symbol = :sym AND timeframe = :tf
            ORDER BY id DESC
            LIMIT 1;
        """), {"sym": symbol, "tf": timeframe}).fetchone()

    if not row:
        return {
            "prediction": "HOLD",
            "percentile_rank": 0.0,
            "candle_timestamp": None,
            "created_at": None,
            "executed": False,
            "has_record": False
        }

    p_id, p_ts, p_act, p_rank, p_exec, p_created = row
    return {
        "prediction": p_act,
        "percentile_rank": float(p_rank) if p_rank is not None else 0.0,
        "candle_timestamp": p_ts,
        "created_at": p_created,
        "executed": bool(p_exec),
        "has_record": True
    }


def calculate_shadow_analysis(symbol: str, timeframe: str = "1h", account_equity: float = 10000.0) -> dict:
    """
    Computes read-only market scenario analysis for a given symbol.
    Fails closed if history or indicators are missing.
    """
    # 1. Fetch Authoritative Phase 13 Prediction from prediction_log
    phase13_state = fetch_authoritative_phase13_prediction(symbol, timeframe)
    
    # 2. Load latest closed OHLCV history
    df = load_ohlcv(symbol, timeframe, limit=200)
    if len(df) < 60:
        raise ValueError(f"INSUFFICIENT DATA: Need at least 60 closed bars for {symbol} shadow analysis, got {len(df)}.")

    latest_bar = df.iloc[-1]
    close_price = float(latest_bar["close"])
    candle_ts = latest_bar["timestamp"]

    # 3. Calculate Technical Indicators (Strict Validation — Fail Closed on NaNs)
    # TR & ATR (14)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        )
    )
    df["atr_14"] = df["tr"].rolling(14).mean()
    
    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # EMA 20 & EMA 50
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Fail closed check for NaNs in latest row
    last_row = df.iloc[-1]
    if pd.isna(last_row["atr_14"]) or pd.isna(last_row["rsi_14"]) or pd.isna(last_row["ema_20"]) or pd.isna(last_row["ema_50"]):
        raise ValueError(f"INVALID FEATURES: Technical indicators generated NaNs for {symbol} {timeframe}.")

    atr_14 = float(last_row["atr_14"])
    rsi_14 = float(last_row["rsi_14"])
    ema_20 = float(last_row["ema_20"])
    ema_50 = float(last_row["ema_50"])
    atr_pct = (atr_14 / close_price) * 100.0

    # Trend Regime
    if close_price > ema_20 > ema_50:
        regime = "BULLISH TREND"
    elif close_price < ema_20 < ema_50:
        regime = "BEARISH TREND"
    else:
        regime = "RANGING / CONSOLIDATION"

    # 4. Phase 13 Authoritative Status & Explanation
    percentile_rank = phase13_state["percentile_rank"]
    phase13_action = phase13_state["prediction"]
    
    if percentile_rank < PHASE13_GATE_THRESHOLD:
        phase13_reason = (
            f"Phase 13 predicted percentile rank is {percentile_rank*100:.2f}% (cutoff required: 99.00%). "
            f"Prediction does not fall in the Top-1% entry region."
        )
    else:
        phase13_reason = f"Phase 13 predicted percentile rank is {percentile_rank*100:.2f}%, meeting the Top-1% entry gate."

    # 5. Technical Heuristic Scenario Scoring (0-100) & Auditable Factor Breakdown
    # Clearly labeled as HEURISTIC SCENARIO SCORES, NOT ML PROBABILITIES.
    long_ml_pts = round(percentile_rank * 50, 1)
    long_rsi_pts = round((100 - rsi_14) * 0.25, 1)
    long_regime_pts = 25.0 if regime == "BULLISH TREND" else 0.0
    long_score = int(np.clip(long_ml_pts + long_rsi_pts + long_regime_pts, 0, 100))

    short_ml_pts = round((1.0 - percentile_rank) * 50, 1)
    short_rsi_pts = round(rsi_14 * 0.25, 1)
    short_regime_pts = 25.0 if regime == "BEARISH TREND" else 0.0
    short_score = int(np.clip(short_ml_pts + short_rsi_pts + short_regime_pts, 0, 100))

    if long_score > short_score + 10 and long_score >= 55:
        scenario_bias = "LONG-FAVORED SCENARIO"
    elif short_score > long_score + 10 and short_score >= 55:
        scenario_bias = "SHORT-FAVORED SCENARIO"
    else:
        scenario_bias = "NEUTRAL / BALANCED SCENARIO"

    # Setups
    long_sl = close_price - (1.5 * atr_14)
    long_tp1 = close_price + (3.0 * atr_14)
    long_risk = close_price - long_sl
    long_rr = (long_tp1 - close_price) / max(long_risk, 1e-5)

    short_sl = close_price + (1.5 * atr_14)
    short_tp1 = close_price - (3.0 * atr_14)
    short_risk = short_sl - close_price
    short_rr = (close_price - short_tp1) / max(short_risk, 1e-5)

    target_risk_usd = account_equity * 0.01  # 1% risk per trade scenario

    return {
        "mode": "MODE_B_SHADOW_ANALYSIS (READ-ONLY)",
        "disclaimer": "DISCLAIMER: SHADOW MARKET ANALYSIS — READ-ONLY / HEURISTIC SCENARIO ONLY — NOT VALIDATED FOR LIVE TRADING — DOES NOT AFFECT PHASE 13",
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_timestamp": candle_ts.isoformat() if hasattr(candle_ts, 'isoformat') else str(candle_ts),
        "close_price": close_price,
        
        # Authoritative Phase 13 block
        "phase13_authoritative": {
            "prediction": phase13_action,
            "percentile_rank": percentile_rank,
            "percentile_pct": f"{percentile_rank*100:.2f}%",
            "gate_required_pct": "99.00%",
            "reason": phase13_reason,
            "candle_timestamp": str(phase13_state.get("candle_timestamp")),
        },
        
        # Technical Market Context
        "market_context": {
            "regime": regime,
            "rsi_14": round(rsi_14, 2),
            "atr_14": round(atr_14, 4),
            "atr_pct": round(atr_pct, 2),
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
        },
        
        # Heuristic Scenario Analysis
        "scenario_analysis": {
            "scenario_bias": scenario_bias,
            "summary_conclusion": f"Market context is {regime} with RSI {rsi_14:.1f}. Technical heuristic favors {scenario_bias.split()[0]} scenario, but this does NOT alter Phase 13's {phase13_action} call.",
            "long_scenario": {
                "technical_score": long_score,
                "score_label": f"{long_score}/100 (Heuristic)",
                "entry_price": close_price,
                "stop_loss": round(long_sl, 2),
                "take_profit_1": round(long_tp1, 2),
                "risk_reward_ratio": round(long_rr, 2),
                "scenario_risk_usd": round(target_risk_usd, 2),
                "scenario_profit_usd": round(target_risk_usd * long_rr, 2),
                "score_contributors": {
                    "ml_percentile_pts": long_ml_pts,
                    "rsi_pts": long_rsi_pts,
                    "regime_pts": long_regime_pts
                }
            },
            "short_scenario": {
                "technical_score": short_score,
                "score_label": f"{short_score}/100 (Heuristic)",
                "entry_price": close_price,
                "stop_loss": round(short_sl, 2),
                "take_profit_1": round(short_tp1, 2),
                "risk_reward_ratio": round(short_rr, 2),
                "scenario_risk_usd": round(target_risk_usd, 2),
                "scenario_profit_usd": round(target_risk_usd * short_rr, 2),
                "score_contributors": {
                    "ml_percentile_pts": short_ml_pts,
                    "rsi_pts": short_rsi_pts,
                    "regime_pts": short_regime_pts
                }
            }
        }
    }

if __name__ == "__main__":
    print("Testing Hardened Shadow Analyst Mode B...")
    for sym in ["BTC/USDT", "ETH/USDT"]:
        res = calculate_shadow_analysis(sym)
        p13 = res["phase13_authoritative"]
        sc = res["scenario_analysis"]
        print(f"\n==================================================")
        print(f"  {res['symbol']} — MODE B SHADOW SCENARIO ANALYSIS")
        print(f"==================================================")
        print(f"  Close Price      : ${res['close_price']:,.2f}")
        print(f"  Authoritative P13: {p13['prediction']} (Rank: {p13['percentile_pct']} vs {p13['gate_required_pct']} req)")
        print(f"  Phase 13 Reason  : {p13['reason']}")
        print(f"  Market Regime    : {res['market_context']['regime']} (RSI={res['market_context']['rsi_14']})")
        print(f"  Scenario Bias    : {sc['scenario_bias']}")
        print(f"  LONG Scenario    : Score {sc['long_scenario']['score_label']} | SL ${sc['long_scenario']['stop_loss']:,.2f} | TP ${sc['long_scenario']['take_profit_1']:,.2f} | R:R {sc['long_scenario']['risk_reward_ratio']}")
        print(f"  SHORT Scenario   : Score {sc['short_scenario']['score_label']} | SL ${sc['short_scenario']['stop_loss']:,.2f} | TP ${sc['short_scenario']['take_profit_1']:,.2f} | R:R {sc['short_scenario']['risk_reward_ratio']}")
        print(f"  Disclaimer       : WARNING - {res['disclaimer']}")
        print(f"==================================================")
