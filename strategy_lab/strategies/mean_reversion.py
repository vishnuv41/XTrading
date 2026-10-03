from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_lab.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "1h", params: dict = None):
        super().__init__("Strategy C — Mean Reversion (RSI+Bollinger)", symbol, timeframe, params)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 30.0)
        self.rsi_overbought = self.params.get("rsi_overbought", 70.0)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.5)

    def on_bar(self, df_window: pd.DataFrame) -> dict:
        if len(df_window) < 50:
            return {"signal": "HOLD", "reason": "Insufficient history for Bollinger/RSI"}

        df = df_window.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["ma_20"] = c.rolling(20).mean()
        df["std_20"] = c.rolling(20).std()
        df["bb_upper"] = df["ma_20"] + (2.0 * df["std_20"])
        df["bb_lower"] = df["ma_20"] - (2.0 * df["std_20"])

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["tr"] = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        df["atr"] = df["tr"].rolling(14).mean()

        curr = df.iloc[-1]
        curr_close = float(curr["close"])
        curr_atr = float(curr["atr"])
        rsi_val = float(curr["rsi"])

        if pd.isna(curr_atr) or curr_atr <= 0:
            return {"signal": "HOLD", "reason": "Invalid ATR"}

        if curr_close <= curr["bb_lower"] and rsi_val <= self.rsi_oversold:
            sl = curr_close - (self.sl_atr_mult * curr_atr)
            tp = float(curr["ma_20"])
            if tp > curr_close:
                return {
                    "signal": "LONG",
                    "entry_price": curr_close,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": f"Oversold BB Touch + RSI={rsi_val:.1f}"
                }

        if curr_close >= curr["bb_upper"] and rsi_val >= self.rsi_overbought:
            sl = curr_close + (self.sl_atr_mult * curr_atr)
            tp = float(curr["ma_20"])
            if tp < curr_close:
                return {
                    "signal": "SHORT",
                    "entry_price": curr_close,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": f"Overbought BB Touch + RSI={rsi_val:.1f}"
                }

        return {"signal": "HOLD", "reason": "No mean reversion extreme"}
