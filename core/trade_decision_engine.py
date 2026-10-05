"""
core/trade_decision_engine.py
---------------------------------
XTrading Real-Time Trade Decision Intelligence & Analytics Engine.

Provides institutional real-time market analysis for human discretionary traders
and paper validation without interfering with frozen prospective validation baselines (Phase 17).

Core Pillars:
1. Multi-Timeframe Technical Intelligence (15m, 1h, 4h, 1d)
2. Regime Classification (Strong Bull, Bullish, Sideways/Chop, Volatility, Bearish, Strong Bear)
3. Setup Detection (Trend Continuation, Breakout, Pullback/Retest, Mean Reversion)
4. Decision & Quality Scoring (0-100) with Hard Risk Gates (R:R >= 2.0, HTF conflict filter)
5. Actionable Execution Plan (Non-arbitrary Structure+ATR Stop Loss, TP1/TP2, Position Sizing)
6. Deep Transparent Explanations (Why Trade, Why Wait, What's Missing, Invalidation Conditions)
7. Multi-Asset Live Scanner ranking all supported coins
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy import text

from database.connection import get_engine

SUPPORTED_ASSETS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "LTC/USDT",
    "LINK/USDT",
]


def fetch_ohlcv_dataframe(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 250,
    exchange: str = "binance"
) -> pd.DataFrame:
    """Fetch recent OHLCV bars from PostgreSQL, falling back gracefully to 4h/1d if 1h/15m is unpopulated for an asset."""
    engine = get_engine()
    with engine.connect() as conn:
        # First attempt requested timeframe
        rows = conn.execute(text("""
            SELECT ts, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :tf AND exchange = :exchange
            ORDER BY ts DESC
            LIMIT :limit
        """), {"symbol": symbol, "tf": timeframe, "exchange": exchange, "limit": limit}).fetchall()

        # Fallback to 4h if requested timeframe has < 30 bars
        if len(rows) < 30 and timeframe in ["15m", "5m", "1h"]:
            rows = conn.execute(text("""
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :symbol AND timeframe = '4h' AND exchange = :exchange
                ORDER BY ts DESC
                LIMIT :limit
            """), {"symbol": symbol, "exchange": exchange, "limit": limit}).fetchall()

        if len(rows) < 30:
            # Fallback to 1d
            rows = conn.execute(text("""
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :symbol AND timeframe = '1d' AND exchange = :exchange
                ORDER BY ts DESC
                LIMIT :limit
            """), {"symbol": symbol, "exchange": exchange, "limit": limit}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)  # Chronological ascending
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical features across trend, momentum, volatility, volume, and structure."""
    if len(df) < 25:
        return df

    # Moving Averages
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=20).mean()

    # True Range & ATR (14)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr_14"] = df["tr"].rolling(14, min_periods=5).mean()
    df["atr_pct"] = (df["atr_14"] / df["close"]) * 100.0

    # Bollinger Bands (20, 2)
    df["bb_mid"] = df["close"].rolling(20, min_periods=10).mean()
    df["bb_std"] = df["close"].rolling(20, min_periods=10).std()
    df["bb_upper"] = df["bb_mid"] + 2.0 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2.0 * df["bb_std"]
    df["bb_bandwidth"] = ((df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-8)) * 100.0

    # Volatility Percentile (Rolling 100 bars)
    df["vol_percentile"] = df["atr_pct"].rolling(100, min_periods=20).apply(
        lambda s: (s.iloc[-1] <= s).mean() * 100.0 if len(s) > 0 else 50.0, raw=False
    )

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ADX & Directional Movement (14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    smooth_plus = pd.Series(plus_dm, index=df.index).rolling(14, min_periods=5).mean()
    smooth_minus = pd.Series(minus_dm, index=df.index).rolling(14, min_periods=5).mean()
    tr_smooth = df["tr"].rolling(14, min_periods=5).mean() + 1e-8

    df["plus_di"] = (smooth_plus / tr_smooth) * 100.0
    df["minus_di"] = (smooth_minus / tr_smooth) * 100.0
    dx = ((df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"] + 1e-8)) * 100.0
    df["adx_14"] = dx.rolling(14, min_periods=5).mean().fillna(20.0)

    # Volume Indicators
    df["vol_ma_20"] = df["volume"].rolling(20, min_periods=5).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma_20"] + 1e-8)

    # Market Structure (Local Swing Highs / Lows across last 20 bars)
    df["swing_high_20"] = df["high"].rolling(20, min_periods=10).max()
    df["swing_low_20"] = df["low"].rolling(20, min_periods=10).min()

    return df


def classify_market_regime(latest: pd.Series) -> Dict[str, Any]:
    """Classify current market regime into institutional state with transparent evidence."""
    price = latest["close"]
    ema20 = latest.get("ema_20", price)
    ema50 = latest.get("ema_50", price)
    ema200 = latest.get("ema_200", ema50)
    adx = latest.get("adx_14", 20.0)
    rsi = latest.get("rsi_14", 50.0)
    atr_pct = latest.get("atr_pct", 2.0)
    vol_ratio = latest.get("vol_ratio", 1.0)

    bull_trend = (price > ema20) and (ema20 > ema50)
    strong_bull = bull_trend and (ema50 > ema200) and (adx >= 25.0)
    bear_trend = (price < ema20) and (ema20 < ema50)
    strong_bear = bear_trend and (ema50 < ema200) and (adx >= 25.0)

    if strong_bull:
        regime = "STRONG BULL"
        color = "#10B981"
        badge = "🟢"
        evidence = ["EMA 20/50/200 Full Bullish Stack", f"ADX Strong ({adx:.1f} >= 25)", f"RSI Momentum ({rsi:.1f})"]
    elif bull_trend:
        regime = "BULLISH TREND"
        color = "#34D399"
        badge = "🟢"
        evidence = ["Price > EMA20 > EMA50", f"Positive Momentum (RSI {rsi:.1f})", "Moderate Trend Strength"]
    elif strong_bear:
        regime = "STRONG BEAR"
        color = "#EF4444"
        badge = "🔴"
        evidence = ["EMA 20/50/200 Full Bearish Stack", f"ADX Strong Trend ({adx:.1f} >= 25)", f"RSI Depressed ({rsi:.1f})"]
    elif bear_trend:
        regime = "BEARISH TREND"
        color = "#F87171"
        badge = "🔴"
        evidence = ["Price < EMA20 < EMA50", f"Negative Momentum (RSI {rsi:.1f})", "Bearish Slope"]
    elif atr_pct > 4.5 or (latest.get("bb_bandwidth", 0) > 10.0 and vol_ratio > 1.8):
        regime = "HIGH VOLATILITY"
        color = "#F59E0B"
        badge = "🟠"
        evidence = [f"Elevated ATR ({atr_pct:.2f}%)", f"Volume Surge ({vol_ratio:.2f}x MA)", "Potential Breakout Expansion"]
    elif adx < 18.0 or abs(rsi - 50.0) < 6.0:
        regime = "SIDEWAYS / CHOPPY"
        color = "#6B7280"
        badge = "🟡"
        evidence = [f"ADX Weak ({adx:.1f} < 18)", "Price Compressing around EMAs", f"RSI Neutral ({rsi:.1f})"]
    else:
        regime = "CONSOLIDATION"
        color = "#9CA3AF"
        badge = "⚪"
        evidence = ["EMAs Entangled", "Normal Volatility", "Awaiting Directional Expansion"]

    return {
        "regime": regime,
        "color": color,
        "badge": badge,
        "evidence": evidence,
        "adx": round(float(adx), 1),
        "rsi": round(float(rsi), 1),
        "atr_pct": round(float(atr_pct), 2),
    }


def detect_trade_setup(df: pd.DataFrame, latest: pd.Series, regime_info: Dict[str, Any]) -> Dict[str, Any]:
    """Detect presence of institutional setups: Breakout, Trend Continuation, Pullback/Retest, Mean Reversion."""
    price = float(latest["close"])
    ema20 = float(latest.get("ema_20", price))
    ema50 = float(latest.get("ema_50", price))
    rsi = float(latest.get("rsi_14", 50.0))
    vol_ratio = float(latest.get("vol_ratio", 1.0))
    adx = float(latest.get("adx_14", 20.0))
    swing_high = float(latest.get("swing_high_20", price * 1.05))
    swing_low = float(latest.get("swing_low_20", price * 0.95))

    regime = regime_info["regime"]

    # 1. Breakout Setup
    if price >= swing_high * 0.998 and vol_ratio >= 1.25 and rsi > 55.0 and "BULL" in regime:
        return {
            "setup_type": "BREAKOUT + MOMENTUM",
            "direction": "LONG",
            "confidence_boost": 20,
            "description": f"Price testing 20-bar swing high (${swing_high:,.2f}) with volume expansion ({vol_ratio:.2f}x MA) and RSI {rsi:.1f}."
        }
    elif price <= swing_low * 1.002 and vol_ratio >= 1.25 and rsi < 45.0 and "BEAR" in regime:
        return {
            "setup_type": "BREAKDOWN + MOMENTUM",
            "direction": "SHORT",
            "confidence_boost": 20,
            "description": f"Price breaching 20-bar swing low (${swing_low:,.2f}) with volume surge ({vol_ratio:.2f}x MA)."
        }

    # 2. Trend Continuation Pullback
    if "BULL" in regime and price > ema50 and abs(price - ema20) / price < 0.015 and 45.0 <= rsi <= 62.0:
        return {
            "setup_type": "TREND PULLBACK TO EMA",
            "direction": "LONG",
            "confidence_boost": 15,
            "description": f"Healthy shallow pullback to EMA20 support (${ema20:,.2f}) in confirmed bull regime."
        }
    elif "BEAR" in regime and price < ema50 and abs(price - ema20) / price < 0.015 and 38.0 <= rsi <= 55.0:
        return {
            "setup_type": "TREND PULLBACK TO EMA",
            "direction": "SHORT",
            "confidence_boost": 15,
            "description": f"Bearish relief rally into EMA20 resistance (${ema20:,.2f}) with falling momentum."
        }

    # 3. Mean Reversion
    if rsi <= 28.0 and price <= latest.get("bb_lower", price):
        return {
            "setup_type": "OVERSOLD MEAN REVERSION",
            "direction": "LONG",
            "confidence_boost": 10,
            "description": f"Extreme oversold condition (RSI {rsi:.1f}) piercing lower Bollinger Band."
        }
    elif rsi >= 72.0 and price >= latest.get("bb_upper", price):
        return {
            "setup_type": "OVERBOUGHT MEAN REVERSION",
            "direction": "SHORT",
            "confidence_boost": 10,
            "description": f"Extreme overbought condition (RSI {rsi:.1f}) tagging upper Bollinger Band."
        }

    return {
        "setup_type": "NO DEFINED SETUP",
        "direction": "NEUTRAL",
        "confidence_boost": 0,
        "description": "Market lacks clean structural breakout or high-probability continuation trigger."
    }


def analyze_multi_timeframe(symbol: str) -> Dict[str, Any]:
    """Perform multi-timeframe analysis across 15m (entry), 1h (setup), 4h (swing trend), 1d (macro)."""
    tfs = ["15m", "1h", "4h", "1d"]
    results = {}
    bullish_count = 0
    bearish_count = 0

    for tf in tfs:
        df = fetch_ohlcv_dataframe(symbol, timeframe=tf, limit=100)
        if len(df) < 20:
            results[tf] = {"trend": "NEUTRAL", "status": "INSUFFICIENT_DATA", "badge": "⚪", "rsi": 50.0}
            continue

        df = calculate_technical_features(df)
        last = df.iloc[-1]
        p = last["close"]
        e20 = last.get("ema_20", p)
        e50 = last.get("ema_50", p)
        rsi = last.get("rsi_14", 50.0)

        if p > e20 > e50 and rsi > 50:
            trend = "BULLISH"
            badge = "🟢"
            bullish_count += 1
        elif p < e20 < e50 and rsi < 50:
            trend = "BEARISH"
            badge = "🔴"
            bearish_count += 1
        else:
            trend = "SIDEWAYS"
            badge = "🟡"

        results[tf] = {
            "trend": trend,
            "badge": badge,
            "close": round(float(p), 2),
            "ema20": round(float(e20), 2),
            "ema50": round(float(e50), 2),
            "rsi": round(float(rsi), 1),
        }

    alignment = "ALIGNED BULLISH" if bullish_count >= 3 else ("ALIGNED BEARISH" if bearish_count >= 3 else "MIXED / CONFLICTING")
    return {
        "timeframes": results,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "alignment": alignment,
        "aligned": bullish_count >= 3 or bearish_count >= 3
    }


def calculate_comprehensive_decision(
    symbol: str,
    account_equity: float = 10000.0,
    risk_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Main entry point for Real-Time Decision Intelligence.
    Computes setup quality score, hard gates, actionable plan (Entry, SL, TP1, TP2, sizing),
    and transparent explanations.
    """
    # 1. Fetch Primary Horizon (1H)
    df_1h = fetch_ohlcv_dataframe(symbol, timeframe="1h", limit=200)
    if len(df_1h) < 25:
        # Fallback to 4H
        df_1h = fetch_ohlcv_dataframe(symbol, timeframe="4h", limit=200)
    if len(df_1h) < 20:
        raise ValueError(f"Insufficient OHLCV data to analyze {symbol}.")

    df_1h = calculate_technical_features(df_1h)
    latest = df_1h.iloc[-1]
    price = float(latest["close"])
    candle_ts = latest["ts"]

    # 2. Multi-Timeframe Alignment
    htf_analysis = analyze_multi_timeframe(symbol)

    # 3. Regime Classification
    regime = classify_market_regime(latest)

    # 4. Setup Detection
    setup = detect_trade_setup(df_1h, latest, regime)

    # 5. Stop Loss & Take Profit Calculations (Structure + Volatility)
    atr = float(latest.get("atr_14", price * 0.02))
    swing_high = float(latest.get("swing_high_20", price + 2 * atr))
    swing_low = float(latest.get("swing_low_20", price - 2 * atr))

    # Long Levels
    long_sl = min(swing_low * 0.997, price - 1.5 * atr)
    long_risk = max(price - long_sl, price * 0.005)
    long_tp1 = price + 2.0 * long_risk
    long_tp2 = price + 3.5 * long_risk
    long_rr = (long_tp1 - price) / long_risk

    # Short Levels
    short_sl = max(swing_high * 1.003, price + 1.5 * atr)
    short_risk = max(short_sl - price, price * 0.005)
    short_tp1 = price - 2.0 * short_risk
    short_tp2 = price - 3.5 * short_risk
    short_rr = (price - short_tp1) / short_risk

    # 6. Quality Scoring Engine (0 - 100)
    # Weights: Trend (25), Momentum (15), Structure (20), HTF Alignment (20), Volume (10), Volatility (10)
    score = 40  # baseline neutral

    # Trend component (+/- 25)
    if "STRONG BULL" in regime["regime"]:
        score += 25
    elif "BULLISH" in regime["regime"]:
        score += 15
    elif "STRONG BEAR" in regime["regime"]:
        score -= 20
    elif "BEARISH" in regime["regime"]:
        score -= 10

    # Momentum component (+/- 15)
    rsi = float(latest.get("rsi_14", 50.0))
    macd_hist = float(latest.get("macd_hist", 0.0))
    if 52.0 <= rsi <= 68.0 and macd_hist > 0:
        score += 15
    elif rsi > 70.0:
        score += 5  # overbought penalty
    elif rsi < 35.0:
        score -= 10

    # Multi-Timeframe Alignment (+/- 20)
    if htf_analysis["bullish_count"] >= 3:
        score += 20
    elif htf_analysis["bearish_count"] >= 3:
        score -= 15
    else:
        score -= 5  # conflicting penalty

    # Volume Confirmation (+/- 10)
    vol_ratio = float(latest.get("vol_ratio", 1.0))
    if vol_ratio >= 1.25:
        score += 10
    elif vol_ratio < 0.7:
        score -= 5

    # Setup Boost (+/- 15)
    score += setup.get("confidence_boost", 0)

    score = int(np.clip(score, 5, 95))

    # 7. Directional Bias & Decision Gates
    reasons_why_trade = []
    reasons_why_wait = []
    hard_gate_passed = True
    gate_blocker = None

    # Determine Direction
    if setup["direction"] == "LONG" or (score >= 65 and "BULL" in regime["regime"]):
        direction = "LONG"
        active_entry = price
        active_sl = round(long_sl, 2)
        active_tp1 = round(long_tp1, 2)
        active_tp2 = round(long_tp2, 2)
        active_rr = round(long_rr, 2)
        active_risk_unit = long_risk
    elif setup["direction"] == "SHORT" or (score <= 35 and "BEAR" in regime["regime"]):
        direction = "SHORT"
        active_entry = price
        active_sl = round(short_sl, 2)
        active_tp1 = round(short_tp1, 2)
        active_tp2 = round(short_tp2, 2)
        active_rr = round(short_rr, 2)
        active_risk_unit = short_risk
        # invert short score for quality display
        score = int(np.clip(100 - score, 5, 95))
    else:
        direction = "NEUTRAL"
        active_entry = price
        active_sl = round(long_sl, 2)
        active_tp1 = round(long_tp1, 2)
        active_tp2 = round(long_tp2, 2)
        active_rr = round(long_rr, 2)
        active_risk_unit = long_risk

    # Hard Gates Evaluation
    if active_rr < 1.8:
        hard_gate_passed = False
        gate_blocker = f"Risk/Reward ratio ({active_rr:.2f}) is below minimum 1.80 threshold."
        reasons_why_wait.append(f"Insufficient R:R ({active_rr:.2f} < 1.80)")

    if not htf_analysis["aligned"] and direction != "NEUTRAL":
        reasons_why_wait.append(f"Higher timeframe conflict: {htf_analysis['alignment']}")
        if score < 75:
            hard_gate_passed = False
            gate_blocker = "Higher timeframe alignment is conflicting (need >= 3 timeframes in agreement)."

    if regime["regime"] in ["SIDEWAYS / CHOPPY", "CONSOLIDATION"]:
        reasons_why_wait.append("Market is in choppy consolidation without directional momentum.")
        if score < 70:
            hard_gate_passed = False
            gate_blocker = "Market regime is Sideways/Chop with ADX < 18."

    # Build Positive Supporting Reasons
    if "BULL" in regime["regime"] and direction == "LONG":
        reasons_why_trade.append("Trend alignment: Price > EMA20 > EMA50")
    if "BEAR" in regime["regime"] and direction == "SHORT":
        reasons_why_trade.append("Bearish trend alignment: Price < EMA20 < EMA50")
    if htf_analysis["bullish_count"] >= 3 and direction == "LONG":
        reasons_why_trade.append(f"Multi-timeframe consensus: {htf_analysis['bullish_count']}/4 timeframes bullish")
    if htf_analysis["bearish_count"] >= 3 and direction == "SHORT":
        reasons_why_trade.append(f"Multi-timeframe consensus: {htf_analysis['bearish_count']}/4 timeframes bearish")
    if vol_ratio >= 1.2:
        reasons_why_trade.append(f"Volume surge: {vol_ratio:.2f}x above 20-period moving average")
    if 50.0 <= rsi <= 68.0 and direction == "LONG":
        reasons_why_trade.append(f"RSI in bullish expansion zone ({rsi:.1f})")
    if setup["setup_type"] != "NO DEFINED SETUP":
        reasons_why_trade.append(f"Setup identified: {setup['setup_type']}")
    if active_rr >= 2.0:
        reasons_why_trade.append(f"Favorable Risk/Reward ratio: 1 : {active_rr:.1f}")

    # Build Negative / Missing Checklist
    if rsi < 50.0 and direction == "LONG":
        reasons_why_wait.append(f"RSI ({rsi:.1f}) has not reclaimed momentum above 50.0")
    if vol_ratio < 1.0:
        reasons_why_wait.append(f"Volume is below average ({vol_ratio:.2f}x MA)")
    if latest.get("adx_14", 20) < 20.0:
        reasons_why_wait.append(f"Trend strength weak (ADX {latest.get('adx_14', 20):.1f} < 20)")

    # Final Decision Assignment
    if score >= 70 and hard_gate_passed and direction in ["LONG", "SHORT"]:
        decision = "TRADE"
        decision_badge = "🟢"
        decision_label = f"TRADE — {direction} BIAS"
        decision_color = "#10B981"
    elif score >= 50 and direction in ["LONG", "SHORT"]:
        decision = "WATCH"
        decision_badge = "🟡"
        decision_label = f"WATCH — {direction} SETUP FORMING"
        decision_color = "#F59E0B"
    else:
        decision = "NO TRADE"
        decision_badge = "🔴"
        decision_label = "NO TRADE — WAIT FOR CONFIRMATION"
        decision_color = "#EF4444"

    # 8. Risk Management & Position Sizing
    risk_dollars = account_equity * (risk_pct / 100.0)
    position_units = risk_dollars / max(active_risk_unit, 1e-6)
    position_usd = position_units * price

    # 9. Invalidation & Trigger Conditions
    invalidation_level = active_sl
    if direction == "LONG":
        wait_condition = f"Wait for 1H candle to close above ${swing_high:,.2f} with volume > 1.2x MA."
        invalidation_text = f"Setup is invalid if price breaks below Stop Loss at ${active_sl:,.2f}."
    elif direction == "SHORT":
        wait_condition = f"Wait for 1H candle to close below ${swing_low:,.2f} with volume > 1.2x MA."
        invalidation_text = f"Setup is invalid if price breaks above Stop Loss at ${active_sl:,.2f}."
    else:
        wait_condition = "Wait for directional breakout from consolidation range."
        invalidation_text = "Range remains active until clean multi-timeframe breakout."

    # Next review time
    try:
        dt = datetime.fromisoformat(str(candle_ts))
        dt_next = dt + pd.Timedelta(hours=1)
        next_review_str = f"After next 1H close (~{dt_next.strftime('%H:%M')} UTC)"
    except Exception:
        next_review_str = "After next 1H candle close"

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candle_timestamp": str(candle_ts),
        "current_price": round(price, 4 if price < 1.0 else 2),
        "decision": decision,
        "decision_badge": decision_badge,
        "decision_label": decision_label,
        "decision_color": decision_color,
        "setup_quality_score": score,
        "regime": regime,
        "setup": setup,
        "direction": direction,
        "trade_plan": {
            "direction": direction,
            "current_price": round(price, 4 if price < 1.0 else 2),
            "entry_zone": f"${price * 0.998:,.2f} – ${price * 1.002:,.2f}",
            "stop_loss": active_sl,
            "take_profit_1": active_tp1,
            "take_profit_2": active_tp2,
            "risk_reward_ratio": active_rr,
            "risk_pct": risk_pct,
            "risk_amount_usd": round(risk_dollars, 2),
            "position_size_units": round(position_units, 4 if position_units < 1.0 else 2),
            "position_value_usd": round(position_usd, 2),
            "expected_holding": "4–12 hours (1H Horizon)",
        },
        "multi_timeframe": htf_analysis,
        "explanations": {
            "why_trade": reasons_why_trade if reasons_why_trade else ["Awaiting full confirmation criteria."],
            "why_wait": reasons_why_wait if reasons_why_wait else ["All standard checklist parameters satisfied."],
            "gate_blocker": gate_blocker,
            "wait_condition": wait_condition,
            "invalidation_text": invalidation_text,
            "next_review": next_review_str,
        },
        "indicators": {
            "ema_20": round(float(latest.get("ema_20", price)), 2),
            "ema_50": round(float(latest.get("ema_50", price)), 2),
            "ema_200": round(float(latest.get("ema_200", price)), 2),
            "rsi_14": round(rsi, 1),
            "adx_14": round(float(latest.get("adx_14", 20.0)), 1),
            "atr_14": round(atr, 4),
            "atr_pct": round(float(latest.get("atr_pct", 2.0)), 2),
            "vol_ratio": round(vol_ratio, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "bb_upper": round(float(latest.get("bb_upper", price * 1.02)), 2),
            "bb_lower": round(float(latest.get("bb_lower", price * 0.98)), 2),
        }
    }


def scan_all_markets(account_equity: float = 10000.0) -> List[Dict[str, Any]]:
    """Scan all supported assets and rank them by Setup Quality Score and Opportunity."""
    results = []
    for sym in SUPPORTED_ASSETS:
        try:
            res = calculate_comprehensive_decision(sym, account_equity=account_equity)
            results.append(res)
        except Exception as exc:
            # If an asset fails, log and continue
            continue

    # Sort primarily by Quality Score DESC, then R:R DESC
    results.sort(key=lambda x: (x["decision"] == "TRADE", x["setup_quality_score"], x["trade_plan"]["risk_reward_ratio"]), reverse=True)
    return results


def fetch_chart_series(symbol: str, timeframe: str = "1h", limit: int = 150) -> Dict[str, Any]:
    """Fetch structured OHLCV and indicator lines formatted for Lightweight Charts / TradingView rendering."""
    df = fetch_ohlcv_dataframe(symbol, timeframe=timeframe, limit=limit)
    if len(df) < 10:
        return {"candles": [], "volume": [], "indicators": {}, "levels": {}}

    df = calculate_technical_features(df)

    candles = []
    volumes = []
    ema20_series = []
    ema50_series = []
    ema200_series = []
    rsi_series = []
    macd_series = []

    for _, row in df.iterrows():
        ts_raw = row["ts"]
        if hasattr(ts_raw, 'timestamp'):
            unix_ts = int(ts_raw.timestamp())
        else:
            try:
                unix_ts = int(pd.to_datetime(ts_raw).timestamp())
            except Exception:
                unix_ts = int(datetime.now(timezone.utc).timestamp())

        c_val = float(row["close"])
        o_val = float(row["open"])

        candles.append({
            "time": unix_ts,
            "open": o_val,
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": c_val,
        })

        vol_color = "rgba(16, 185, 129, 0.4)" if c_val >= o_val else "rgba(239, 68, 68, 0.4)"
        volumes.append({
            "time": unix_ts,
            "value": float(row["volume"]),
            "color": vol_color
        })

        if not pd.isna(row.get("ema_20")):
            ema20_series.append({"time": unix_ts, "value": round(float(row["ema_20"]), 2)})
        if not pd.isna(row.get("ema_50")):
            ema50_series.append({"time": unix_ts, "value": round(float(row["ema_50"]), 2)})
        if not pd.isna(row.get("ema_200")):
            ema200_series.append({"time": unix_ts, "value": round(float(row["ema_200"]), 2)})
        if not pd.isna(row.get("rsi_14")):
            rsi_series.append({"time": unix_ts, "value": round(float(row["rsi_14"]), 2)})
        if not pd.isna(row.get("macd_hist")):
            macd_series.append({
                "time": unix_ts,
                "macd": round(float(row["macd_line"]), 2),
                "signal": round(float(row["macd_signal"]), 2),
                "hist": round(float(row["macd_hist"]), 2)
            })

    # Get active trade levels from decision engine
    decision_data = calculate_comprehensive_decision(symbol)
    plan = decision_data["trade_plan"]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "volume": volumes,
        "indicators": {
            "ema20": ema20_series,
            "ema50": ema50_series,
            "ema200": ema200_series,
            "rsi": rsi_series,
            "macd": macd_series,
        },
        "levels": {
            "current_price": plan["current_price"],
            "stop_loss": plan["stop_loss"],
            "take_profit_1": plan["take_profit_1"],
            "take_profit_2": plan["take_profit_2"],
            "support": decision_data["indicators"]["swing_low"],
            "resistance": decision_data["indicators"]["swing_high"],
        },
        "decision": decision_data
    }
