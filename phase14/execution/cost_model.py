from __future__ import annotations
"""
phase14/execution/cost_model.py
-------------------------------
Explicit Cost & Friction Engine for Phase 14.
Models:
- Exchange Taker / Maker Fee
- Slippage
- Bid-Ask Spread
- Funding Rate Accumulation (Perpetuals)
"""

from config.costs import (
    TAKER_FEE_BPS,
    MAKER_FEE_BPS,
    SLIPPAGE_BPS,
    BID_ASK_SPREAD_BPS,
    FUNDING_RATE_8H_PCT,
)

class CostModel:
    def __init__(
        self,
        taker_fee_pct: float = TAKER_FEE_BPS / 10_000.0,
        maker_fee_pct: float = MAKER_FEE_BPS / 10_000.0,
        slippage_pct: float = SLIPPAGE_BPS / 10_000.0,
        bid_ask_spread_pct: float = BID_ASK_SPREAD_BPS / 10_000.0,
        funding_rate_8h_pct: float = FUNDING_RATE_8H_PCT
    ):
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.slippage_pct = slippage_pct
        self.bid_ask_spread_pct = bid_ask_spread_pct
        self.funding_rate_8h_pct = funding_rate_8h_pct

    def calculate_entry_execution(self, price: float, side: str) -> tuple[float, float]:
        """Returns (executed_entry_price, entry_fee_usd_per_unit)"""
        half_spread = (price * self.bid_ask_spread_pct) / 2.0
        slip = price * self.slippage_pct
        
        if side == "LONG":
            exec_price = price + half_spread + slip
        else:
            exec_price = price - half_spread - slip
            
        fee_usd = exec_price * self.taker_fee_pct
        return exec_price, fee_usd

    def calculate_exit_execution(self, price: float, side: str) -> tuple[float, float]:
        """Returns (executed_exit_price, exit_fee_usd_per_unit)"""
        half_spread = (price * self.bid_ask_spread_pct) / 2.0
        slip = price * self.slippage_pct
        
        if side == "LONG":
            exec_price = price - half_spread - slip
        else:
            exec_price = price + half_spread + slip
            
        fee_usd = exec_price * self.taker_fee_pct
        return exec_price, fee_usd

    def calculate_funding_cost(self, notional: float, bars_held: int, timeframe_hours: int = 1) -> float:
        """Calculates accumulated funding fee over holding period."""
        hours_held = bars_held * timeframe_hours
        funding_periods = hours_held / 8.0
        return notional * self.funding_rate_8h_pct * funding_periods
