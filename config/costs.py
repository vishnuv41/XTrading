"""
config/costs.py
---------------
Canonical transaction friction and execution cost configuration.
All research and backtesting harnesses must import cost parameters from here
to maintain consistency across the entire codebase.

Canonical Round-Trip Friction: 30.0 bps (0.0030)
- One-Way Entry Friction: 15.0 bps (0.0015)
- One-Way Exit Friction:  15.0 bps (0.0015)
"""

# Canonical Basis Points
CANONICAL_ROUND_TRIP_BPS: float = 30.0
CANONICAL_ONE_WAY_BPS: float = 15.0

# Decimals
CANONICAL_ROUND_TRIP_PCT: float = CANONICAL_ROUND_TRIP_BPS / 10_000.0  # 0.0030
CANONICAL_ONE_WAY_PCT: float = CANONICAL_ONE_WAY_BPS / 10_000.0        # 0.0015

# Granular Breakdown (Binance Retail Baseline)
TAKER_FEE_BPS: float = 5.0
MAKER_FEE_BPS: float = 2.0
SLIPPAGE_BPS: float = 7.5
BID_ASK_SPREAD_BPS: float = 2.5

# Annualized / Periodic Funding Cost
FUNDING_RATE_8H_BPS: float = 1.0
FUNDING_RATE_8H_PCT: float = FUNDING_RATE_8H_BPS / 10_000.0  # 0.0001
