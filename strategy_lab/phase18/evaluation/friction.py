"""
strategy_lab/phase18/evaluation/friction.py
---------------------------------------------
Canonical and sensitivity friction modeling for Phase 18 research.
"""

from typing import List

CANONICAL_ROUND_TRIP_BPS = 30.0
SENSITIVITY_SCENARIOS_BPS: List[float] = [15.0, 30.0, 45.0]


class Phase18CostModel:
    """
    Standard friction model enforcing exact basis point conversion for one-way trades.
    """
    def __init__(self, round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS):
        self.round_trip_bps = float(round_trip_bps)
        self.one_way_bps = self.round_trip_bps / 2.0
        self.one_way_pct = self.one_way_bps / 10000.0

    def compute_friction(self, turnover: float) -> float:
        """Calculate friction drag from turnover fraction."""
        return float(turnover * self.one_way_pct)
