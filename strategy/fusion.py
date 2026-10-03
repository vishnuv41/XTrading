"""
strategy/fusion.py
----------------------
Signal fusion layer (Part D of the architecture review). Makes the ML
prediction a decision-QUALITY layer rather than the sole source of
truth: the ML model's BUY/SELL + calibrated confidence is checked
against the independent rule-based strategies in strategy/registry.py
and the current regime, and downgraded to HOLD when they actively
disagree or the regime itself is uncertain ('mixed_*').

Deliberately interpretable rules, not a learned/averaged blend — the
whole point is that a human (or a future ML meta-model, later) can read
`reasons` and see exactly why a trade was upgraded, passed through, or
vetoed. This does NOT replace inference.realtime_pipeline's existing
ML-confidence-threshold + risk_engine gating; call it in addition, as
an extra advisory field, or as an extra veto before the risk engine —
see the __main__ example for both usages. Wiring it into the live path
is a deliberate follow-up decision, not made here (see review notes:
"do not modify the live execution path unless absolutely necessary").

Fusion rule (mirrors the worked examples in the architecture review):
    1. regime is 'mixed_*'              -> NO_TRADE ("regime uncertain")
    2. ML signal is HOLD                 -> HOLD (nothing to fuse)
    3. any rule strategy fires the
       OPPOSITE direction to ML          -> HOLD ("conflicting signals")
    4. >=2 rule strategies AGREE with ML -> quality = HIGH
       1 rule strategy agrees            -> quality = MEDIUM
       0 rule strategies fire (all HOLD) -> quality = LOW (ML-only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .registry import latest_signals


@dataclass
class FusedSignal:
    signal: str            # 'BUY' | 'SELL' | 'HOLD'
    quality: str            # 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
    ml_signal: str
    ml_confidence: float
    market_state: Optional[str]
    agreeing_strategies: list = field(default_factory=list)
    conflicting_strategies: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "signal": self.signal, "quality": self.quality,
            "ml_signal": self.ml_signal, "ml_confidence": self.ml_confidence,
            "market_state": self.market_state,
            "agreeing_strategies": self.agreeing_strategies,
            "conflicting_strategies": self.conflicting_strategies,
            "reasons": self.reasons,
        }


def fuse_signals(
    ml_signal: str,
    ml_confidence: float,
    market_state: Optional[str],
    rule_signals: dict[str, dict],
    ml_confidence_threshold: float = 0.5,
) -> FusedSignal:
    """
    Pure fusion logic — no dataframe/indicator dependency, so it's easy
    to unit test and easy to call from either a live pipeline or a
    vectorized backtest that's already computed everything per-row.

    Args:
        ml_signal: 'BUY' | 'SELL' | 'HOLD' from ml.predict.
        ml_confidence: calibrated 0-1 confidence for ml_signal.
        market_state: e.g. 'trending_low_vol', 'mixed_high_vol', or None
            if unavailable (treated as unknown/neutral, not blocked).
        rule_signals: output of strategy.registry.latest_signals(df) —
            {strategy_name: {"signal": ..., "reason": [...], ...}}.
        ml_confidence_threshold: below this, treat ML as not confident
            enough to fuse (mirrors settings.ml.confidence_threshold —
            pass that in so this stays a pure function, not a settings-
            reader).

    Returns:
        FusedSignal.
    """
    reasons = []

    if market_state and market_state.startswith("mixed"):
        return FusedSignal(
            signal="HOLD", quality="NONE", ml_signal=ml_signal, ml_confidence=ml_confidence,
            market_state=market_state, reasons=["Regime uncertain (mixed) — no trade."],
        )

    if ml_signal == "HOLD":
        return FusedSignal(
            signal="HOLD", quality="NONE", ml_signal=ml_signal, ml_confidence=ml_confidence,
            market_state=market_state, reasons=["ML signal is HOLD."],
        )

    if ml_confidence < ml_confidence_threshold:
        return FusedSignal(
            signal="HOLD", quality="NONE", ml_signal=ml_signal, ml_confidence=ml_confidence,
            market_state=market_state,
            reasons=[f"ML confidence {ml_confidence:.2f} below threshold {ml_confidence_threshold:.2f}."],
        )

    agreeing, conflicting = [], []
    for name, sig in rule_signals.items():
        s = sig.get("signal")
        if s == ml_signal:
            agreeing.append(name)
        elif s in ("BUY", "SELL"):  # fired, but the opposite direction
            conflicting.append(name)

    if conflicting:
        reasons.append(f"Conflicting signal(s) from: {', '.join(conflicting)}.")
        return FusedSignal(
            signal="HOLD", quality="NONE", ml_signal=ml_signal, ml_confidence=ml_confidence,
            market_state=market_state, agreeing_strategies=agreeing,
            conflicting_strategies=conflicting, reasons=reasons,
        )

    if len(agreeing) >= 2:
        quality = "HIGH"
        reasons.append(f"Confirmed by {len(agreeing)} independent strategies: {', '.join(agreeing)}.")
    elif len(agreeing) == 1:
        quality = "MEDIUM"
        reasons.append(f"Confirmed by 1 independent strategy: {agreeing[0]}.")
    else:
        quality = "LOW"
        reasons.append("ML-only signal — no rule-based strategy confirmation (none fired).")

    if market_state:
        reasons.append(f"Regime: {market_state}.")

    return FusedSignal(
        signal=ml_signal, quality=quality, ml_signal=ml_signal, ml_confidence=ml_confidence,
        market_state=market_state, agreeing_strategies=agreeing,
        conflicting_strategies=conflicting, reasons=reasons,
    )


def fuse_from_dataframe(
    df: pd.DataFrame,
    ml_signal: str,
    ml_confidence: float,
    ml_confidence_threshold: float = 0.5,
    market_state_col: str = "market_state",
) -> FusedSignal:
    """Convenience wrapper: runs strategy.registry.latest_signals(df) for you."""
    rule_signals = latest_signals(df)
    market_state = df[market_state_col].iloc[-1] if market_state_col in df.columns else None
    return fuse_signals(ml_signal, ml_confidence, market_state, rule_signals, ml_confidence_threshold)


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state

    n = 400
    rng = np.random.default_rng(21)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.random(n) * 1.5,
        "low": close - rng.random(n) * 1.5,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })
    df = calculate_all_indicators(df)
    df = calculate_market_state(df)

    fused = fuse_from_dataframe(df, ml_signal="BUY", ml_confidence=0.78, ml_confidence_threshold=0.70)
    print(fused.as_dict())

