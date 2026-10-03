from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_lab.base_strategy import BaseStrategy

class MLRegimeFilteredStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "1h", params: dict = None):
        super().__init__("Strategy D — ML Regime Filtered (Trend/Reversion)", symbol, timeframe, params)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        self.tp_atr_mult = self.params.get("tp_atr_mult", 3.0)

    def on_bar(self, df_window: pd.DataFrame) -> dict:
        if len(df_window) < 100:
            return {"signal": "HOLD", "reason": "Insufficient history for ML Regime Filter"}

        df = df_window.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["ema_20"] = c.ewm(span=20, adjust=False).mean()
        df["ema_50"] = c.ewm(span=50, adjust=False).mean()

        df["tr"] = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        df["atr"] = df["tr"].rolling(14).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        curr = df.iloc[-1]
        curr_close = float(curr["close"])
        curr_atr = float(curr["atr"])
        rsi_val = float(curr["rsi"])

        if pd.isna(curr_atr) or curr_atr <= 0:
            return {"signal": "HOLD", "reason": "Invalid ATR"}

        ema_20 = float(curr["ema_20"])
        ema_50 = float(curr["ema_50"])
        
        if curr_close > ema_20 > ema_50:
            regime = "BULLISH_TREND"
        elif curr_close < ema_20 < ema_50:
            regime = "BEARISH_TREND"
        else:
            regime = "RANGING"

        if regime == "BULLISH_TREND" and rsi_val < 45:
            sl = curr_close - (self.sl_atr_mult * curr_atr)
            tp = curr_close + (self.tp_atr_mult * curr_atr)
            return {
                "signal": "LONG",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Bullish Regime Pullback (RSI={rsi_val:.1f})"
            }

        if regime == "BEARISH_TREND" and rsi_val > 55:
            sl = curr_close + (self.sl_atr_mult * curr_atr)
            tp = curr_close - (self.tp_atr_mult * curr_atr)
            return {
                "signal": "SHORT",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Bearish Regime Rally (RSI={rsi_val:.1f})"
            }

        if regime == "RANGING":
            if rsi_val < 32:
                sl = curr_close - (self.sl_atr_mult * curr_atr)
                tp = curr_close + (self.tp_atr_mult * curr_atr)
                return {
                    "signal": "LONG",
                    "entry_price": curr_close,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": f"Ranging Regime Oversold (RSI={rsi_val:.1f})"
                }
            elif rsi_val > 68:
                sl = curr_close + (self.sl_atr_mult * curr_atr)
                tp = curr_close - (self.tp_atr_mult * curr_atr)
                return {
                    "signal": "SHORT",
                    "entry_price": curr_close,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": f"Ranging Regime Overbought (RSI={rsi_val:.1f})"
                }

        return {"signal": "HOLD", "reason": "No regime-filtered setup"}
