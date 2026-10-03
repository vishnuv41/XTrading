from __future__ import annotations
import numpy as np
import pandas as pd
from strategy_lab.base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = "1h", params: dict = None):
        super().__init__("Strategy A — Trend Following (EMA+ADX)", symbol, timeframe, params)
        self.fast_ema = self.params.get("fast_ema", 20)
        self.slow_ema = self.params.get("slow_ema", 50)
        self.trend_ema = self.params.get("trend_ema", 200)
        self.adx_threshold = self.params.get("adx_threshold", 22.0)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 2.0)
        self.tp_atr_mult = self.params.get("tp_atr_mult", 4.0)

    def on_bar(self, df_window: pd.DataFrame) -> dict:
        if len(df_window) < self.trend_ema:
            return {"signal": "HOLD", "reason": "Insufficient history for EMA200"}

        df = df_window.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["ema_fast"] = c.ewm(span=self.fast_ema, adjust=False).mean()
        df["ema_slow"] = c.ewm(span=self.slow_ema, adjust=False).mean()
        df["ema_trend"] = c.ewm(span=self.trend_ema, adjust=False).mean()

        df["tr"] = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        df["atr"] = df["tr"].rolling(14).mean()

        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_smooth = df["tr"].rolling(14).sum()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-8))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-8))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
        adx = dx.rolling(14).mean().iloc[-1]

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        curr_close = float(curr["close"])
        curr_atr = float(curr["atr"])

        if pd.isna(curr_atr) or curr_atr <= 0:
            return {"signal": "HOLD", "reason": "Invalid ATR"}

        fast_cross_above = (prev["ema_fast"] <= prev["ema_slow"]) and (curr["ema_fast"] > curr["ema_slow"])
        if fast_cross_above and curr_close > curr["ema_trend"] and adx >= self.adx_threshold:
            sl = curr_close - (self.sl_atr_mult * curr_atr)
            tp = curr_close + (self.tp_atr_mult * curr_atr)
            return {
                "signal": "LONG",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Bullish EMA Cross + ADX={adx:.1f}"
            }

        fast_cross_below = (prev["ema_fast"] >= prev["ema_slow"]) and (curr["ema_fast"] < curr["ema_slow"])
        if fast_cross_below and curr_close < curr["ema_trend"] and adx >= self.adx_threshold:
            sl = curr_close + (self.sl_atr_mult * curr_atr)
            tp = curr_close - (self.tp_atr_mult * curr_atr)
            return {
                "signal": "SHORT",
                "entry_price": curr_close,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": f"Bearish EMA Cross + ADX={adx:.1f}"
            }

        return {"signal": "HOLD", "reason": "No trend crossover setup"}
