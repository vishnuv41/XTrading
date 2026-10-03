from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_lab.base_strategy import BaseStrategy

class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "1h", params: dict = None):
        super().__init__("Strategy B — Momentum Breakout (Donchian+Vol)", symbol, timeframe, params)
        self.channel_period = self.params.get("channel_period", 20)
        self.vol_mult = self.params.get("vol_mult", 1.25)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        self.tp_atr_mult = self.params.get("tp_atr_mult", 3.5)

    def on_bar(self, df_window: pd.DataFrame) -> dict:
        if len(df_window) < self.channel_period + 5:
            return {"signal": "HOLD", "reason": "Insufficient history for Donchian Channel"}

        df = df_window.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        df["upper_channel"] = h.shift(1).rolling(self.channel_period).max()
        df["lower_channel"] = l.shift(1).rolling(self.channel_period).min()
        df["vol_ma"] = v.shift(1).rolling(20).mean()

        df["tr"] = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        df["atr"] = df["tr"].rolling(14).mean()

        curr = df.iloc[-1]
        curr_close = float(curr["close"])
        curr_atr = float(curr["atr"])

        if pd.isna(curr_atr) or curr_atr <= 0:
            return {"signal": "HOLD", "reason": "Invalid ATR"}

        vol_ratio = (curr["volume"] / curr["vol_ma"]) if curr["vol_ma"] > 0 else 1.0

        if curr_close > curr["upper_channel"] and vol_ratio >= self.vol_mult:
            sl = curr_close - (self.sl_atr_mult * curr_atr)
            tp = curr_close + (self.tp_atr_mult * curr_atr)
            return {
                "signal": "LONG",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Donchian Upper Breakout + VolRatio={vol_ratio:.2f}"
            }

        if curr_close < curr["lower_channel"] and vol_ratio >= self.vol_mult:
            sl = curr_close + (self.sl_atr_mult * curr_atr)
            tp = curr_close - (self.tp_atr_mult * curr_atr)
            return {
                "signal": "SHORT",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Donchian Lower Breakout + VolRatio={vol_ratio:.2f}"
            }

        return {"signal": "HOLD", "reason": "No channel breakout"}
