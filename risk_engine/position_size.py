"""
Position Size

Decides how much capital to risk on a given trade. This is where the
strategy's edge (or lack of one) actually gets translated into capital
at risk — a good entry signal with bad position sizing still loses
money over time, and a mediocre signal with disciplined sizing can
survive long enough to be improved.

Three approaches, meant to be combined rather than used in isolation:

1. Fixed-fractional risk sizing (the baseline): risk a fixed % of
   account equity per trade, position size derived from that % and the
   stop-loss distance. This is the simplest and safest default and is
   what most retail risk management guides recommend as a floor.

2. Kelly Criterion (edge-based sizing): given a strategy's historical
   win rate and average win/loss ratio, the Kelly formula computes the
   theoretically optimal fraction of capital to risk to maximize long-
   run geometric growth. Full Kelly is aggressive and assumes the
   win-rate/win-loss inputs are exactly correct (they never are, since
   they're estimated from a finite backtest) — so this module defaults
   to FRACTIONAL Kelly (typically 1/4 to 1/2 of full Kelly), which is
   the standard practical adjustment for parameter uncertainty.

3. Confidence/volatility scaling: the position size from either method
   above is further scaled by the strategy's signal_confidence (from
   strategy/signals.py) and inversely by current volatility regime, so
   low-confidence signals or high-volatility regimes automatically get
   smaller size rather than needing a separate manual override.
"""

import pandas as pd


def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                              fraction: float = 0.5) -> float:
    """
    Calculate the (fractional) Kelly bet size as a fraction of capital.

    Kelly formula: f* = W - (1 - W) / R
      where W = win rate, R = avg_win / avg_loss (win/loss ratio)

    Args:
        win_rate: Historical win rate, 0-1 (from backtest.py).
        avg_win: Average winning trade size (in R-multiples or currency,
            must be consistent units with avg_loss).
        avg_loss: Average losing trade size (positive number representing
            the magnitude of the loss).
        fraction: Fraction of full Kelly to actually use (default 0.5,
            i.e. "half Kelly" — the standard hedge against estimation
            error in win_rate/avg_win/avg_loss, which are only ever
            estimates from a finite, non-stationary sample).

    Returns:
        Recommended fraction of capital to risk (0 if the edge is
        negative — Kelly correctly says "don't bet" rather than a
        meaningless negative number).
    """
    if not (0 < win_rate < 1):
        raise ValueError("win_rate must be between 0 and 1 (exclusive)")
    if avg_win <= 0 or avg_loss <= 0:
        raise ValueError("avg_win and avg_loss must both be positive")
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be between 0 (exclusive) and 1 (inclusive)")

    win_loss_ratio = avg_win / avg_loss
    full_kelly = win_rate - (1 - win_rate) / win_loss_ratio

    return max(full_kelly * fraction, 0.0)


def calculate_fixed_fractional_size(account_equity: float, risk_pct: float,
                                     entry_price: float, stop_loss: float) -> float:
    """
    Calculate position size (in units of the asset) using fixed-
    fractional risk: risk a fixed % of equity, sized so that if the
    stop is hit, the loss equals exactly that %.

    Args:
        account_equity: Current account equity (in quote currency, e.g. USD).
        risk_pct: Fraction of equity to risk on this trade, 0-1
            (e.g. 0.01 for 1% risk per trade — a common conservative
            default; many blown accounts trace back to risking too much
            per trade, not to a bad win rate).
        entry_price: Trade entry price.
        stop_loss: Trade stop-loss price.

    Returns:
        Position size in units of the asset (e.g. number of BTC, not
        USD notional). Multiply by entry_price for notional value.
    """
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if not (0 < risk_pct <= 1):
        raise ValueError("risk_pct must be between 0 (exclusive) and 1 (inclusive)")

    risk_amount = account_equity * risk_pct
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        raise ValueError("stop_distance is zero (entry_price == stop_loss)")

    return risk_amount / stop_distance


def calculate_adjusted_position_size(account_equity: float, entry_price: float,
                                      stop_loss: float, signal_confidence: float,
                                      vol_regime: str, base_risk_pct: float = 0.01,
                                      kelly_fraction: float | None = None) -> dict:
    """
    The recommended entry point: combines fixed-fractional risk sizing
    with confidence and volatility-regime scaling into one final size.

    Args:
        account_equity: Current account equity.
        entry_price: Trade entry price.
        stop_loss: Trade stop-loss price.
        signal_confidence: 0-1 score from strategy/signals.py. Scales
            position size linearly — a 0.6-confidence signal gets 60%
            of the base size, not the full size, so conviction is
            reflected in capital at risk rather than being all-or-nothing.
        vol_regime: 'low', 'medium', or 'high' (from regime/volatility_regime.py).
            High-vol regimes scale size down (more risk per unit of
            price movement), low-vol regimes scale up slightly.
        base_risk_pct: The starting fixed-fractional risk % before
            scaling (default 0.01 = 1% of equity).
        kelly_fraction: Optional. If provided (e.g. from
            calculate_kelly_fraction), used as a CAP on the final
            risk_pct rather than blindly overriding base_risk_pct — so a
            confident, high-Kelly-edge strategy can size up, but never
            beyond what fixed-fractional risk discipline would allow
            on its own if Kelly's estimate turns out to be too optimistic.

    Returns:
        dict with keys:
          'risk_pct'        - final risk % actually used
          'position_size'   - size in asset units
          'notional_value'  - position_size * entry_price
          'risk_amount'     - account_equity * risk_pct (capital at risk)
          'blocked_reason'  - present (non-None) only when account_equity
                               was <= 0; all size/amount fields are 0.0
                               in that case rather than raising.
    """
    if vol_regime not in ("low", "medium", "high"):
        raise ValueError("vol_regime must be 'low', 'medium', or 'high'")
    if not (0 <= signal_confidence <= 1):
        raise ValueError("signal_confidence must be between 0 and 1")

    if account_equity <= 0:
        # Graceful degradation, not a crash: an equity-depleted account is
        # an expected (if unhappy) state for a leveraged system in a loss
        # streak, and every caller of this function should get a safe
        # "don't trade" result rather than having to catch a ValueError
        # from several layers down. inference.realtime_pipeline already
        # gates on this earlier for the same reason — this is the second
        # line of defense for any other caller.
        return {
            "risk_pct": 0.0,
            "position_size": 0.0,
            "notional_value": 0.0,
            "risk_amount": 0.0,
            "blocked_reason": f"account_equity non-positive ({account_equity:.2f})",
        }

    vol_scalar = {"low": 1.2, "medium": 1.0, "high": 0.6}[vol_regime]

    risk_pct = base_risk_pct * signal_confidence * vol_scalar

    if kelly_fraction is not None:
        risk_pct = min(risk_pct, kelly_fraction)

    # Hard ceiling regardless of how the inputs combine — no single
    # trade should ever risk more than 2x the base risk_pct, protecting
    # against a bug or edge case in the scaling logic compounding into
    # an oversized bet.
    risk_pct = min(risk_pct, base_risk_pct * 2)

    position_size = calculate_fixed_fractional_size(
        account_equity, max(risk_pct, 0.0001), entry_price, stop_loss
    )

    return {
        "risk_pct": risk_pct,
        "position_size": position_size,
        "notional_value": position_size * entry_price,
        "risk_amount": account_equity * risk_pct,
    }


if __name__ == "__main__":
    kelly = calculate_kelly_fraction(win_rate=0.55, avg_win=2.0, avg_loss=1.0, fraction=0.5)
    print(f"Half-Kelly fraction (55% win rate, 2:1 win/loss): {kelly:.4f}")

    size = calculate_fixed_fractional_size(
        account_equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=96
    )
    print(f"Fixed-fractional size (1% risk, $10k equity, $4 stop distance): {size:.2f} units")

    result = calculate_adjusted_position_size(
        account_equity=10_000, entry_price=100, stop_loss=96,
        signal_confidence=0.8, vol_regime="high", base_risk_pct=0.01,
        kelly_fraction=kelly,
    )
    print("\nAdjusted position size (0.8 confidence, high-vol regime):")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    result_low_vol = calculate_adjusted_position_size(
        account_equity=10_000, entry_price=100, stop_loss=96,
        signal_confidence=0.8, vol_regime="low", base_risk_pct=0.01,
        kelly_fraction=kelly,
    )
    print("\nSame signal, low-vol regime instead:")
    for k, v in result_low_vol.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")