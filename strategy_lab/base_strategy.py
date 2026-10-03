from __future__ import annotations
"""
strategy_lab/base_strategy.py
------------------------------
Abstract base class for all Strategy Lab candidates.
Purely offline, non-mutating strategy interface.
"""
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str, symbol: str, timeframe: str, params: dict = None):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}

    @abstractmethod
    def on_bar(self, df_window: pd.DataFrame) -> dict:
        """
        Processes an OHLCV window ending at current bar.
        Returns:
            {
                "signal": "LONG" | "SHORT" | "HOLD",
                "entry_price": float,
                "stop_loss": float,
                "take_profit": float,
                "reason": str
            }
        """
        pass
