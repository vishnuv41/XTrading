"""
strategy_lab/lowturn/cost_model.py
----------------------------------
Phase 17 Cost & Friction Module (Rule R3).
Canonical round-trip friction is 30.0 bps (15.0 bps per entry/exit leg).
Supports cost sensitivity evaluations across [0, 10, 20, 30, 40, 50] bps.
"""

from typing import List
from config.costs import (
    CANONICAL_ROUND_TRIP_BPS,
    CANONICAL_ONE_WAY_BPS,
    CANONICAL_ROUND_TRIP_PCT,
    CANONICAL_ONE_WAY_PCT,
)

SENSITIVITY_GRID_BPS: List[float] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


class LowTurnoverCostModel:
    def __init__(self, round_trip_bps: float = CANONICAL_ROUND_TRIP_BPS):
        self.round_trip_bps = round_trip_bps
        self.one_way_bps = round_trip_bps / 2.0
        self.round_trip_pct = self.round_trip_bps / 10_000.0
        self.one_way_pct = self.one_way_bps / 10_000.0

    def apply_turnover_friction(self, turnover: float) -> float:
        """
        Applies friction proportional to turnover.
        turnover = sum(|weight_{t} - weight_{t-1}|)
        Friction penalty = turnover * one_way_pct
        """
        return turnover * self.one_way_pct
