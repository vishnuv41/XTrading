"""
regime package

Single entry point: calculate_market_state(df) runs trend regime
detection, volatility regime detection, and combines them into one
categorical + one-hot encoded market state feature set.

Requires df to already have ADX computed (indicators/trend/adx.py) and
a 'close' column before calling.
"""

from .trend_regime import calculate_trend_regime, calculate_hurst_exponent
from .volatility_regime import calculate_volatility_regime, calculate_realized_volatility
from .market_state import calculate_market_state, market_state_summary, ALL_MARKET_STATES

__all__ = [
    "calculate_trend_regime", "calculate_hurst_exponent",
    "calculate_volatility_regime", "calculate_realized_volatility",
    "calculate_market_state", "market_state_summary", "ALL_MARKET_STATES",
]