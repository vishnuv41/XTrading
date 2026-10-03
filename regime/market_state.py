"""
Market State

Combines trend_regime + volatility_regime into a single categorical
"market state" label — this is the feature the strategy engine and ML
model should actually condition on, rather than reading trend/vol
regimes separately. Six states:

    trending_low_vol      trending_medium_vol      trending_high_vol
    ranging_low_vol        ranging_medium_vol        ranging_high_vol

(plus 'mixed_*' variants when trend_regime returned 'mixed' rather than
a clean trending/ranging call — see trend_regime.py)

Why this matters in practice:
  - trending_low_vol   -> cleanest environment for trend-following entries,
                          tight stops are viable (low noise)
  - trending_high_vol  -> trend is real but noisy; widen stops (ATR-based
                          sizing already helps) and expect more whipsaw
  - ranging_low_vol     -> classic squeeze; mean-reversion (Bollinger/RSI
                          fades) works, or wait for a breakout
  - ranging_high_vol    -> choppiest, most dangerous state; many systems
                          are better off reducing size or standing aside
                          entirely here

This module also exposes a numeric encoding, since most ML libraries
(XGBoost, LightGBM) want the state either one-hot encoded or as an
ordinal/categorical dtype rather than a raw string column.
"""

import pandas as pd

from .trend_regime import calculate_trend_regime
from .volatility_regime import calculate_volatility_regime


# All possible labels, defined once so downstream one-hot encoding is
# consistent across training runs even if a given dataset doesn't happen
# to contain every state.
ALL_MARKET_STATES = [
    "trending_low_vol", "trending_medium_vol", "trending_high_vol",
    "ranging_low_vol", "ranging_medium_vol", "ranging_high_vol",
    "mixed_low_vol", "mixed_medium_vol", "mixed_high_vol",
]


def calculate_market_state(df: pd.DataFrame,
                            adx_col: str = "ADX14",
                            close_col: str = "close",
                            recompute_regimes: bool = True,
                            **regime_kwargs) -> pd.DataFrame:
    """
    Add a combined market_state column (and one-hot columns) to the
    dataframe.

    Args:
        df: DataFrame. Must already have the ADX column computed
            (indicators/trend/adx.py) since trend_regime depends on it.
        adx_col: Name of the ADX column.
        close_col: Close price column.
        recompute_regimes: If True (default), calls calculate_trend_regime
            and calculate_volatility_regime internally. Set False if
            'trend_regime' and 'vol_regime' columns already exist on df
            (e.g. computed earlier in a pipeline) to avoid redundant work.
        **regime_kwargs: Passed through to calculate_trend_regime /
            calculate_volatility_regime (e.g. adx_threshold, window).
            Kwargs are dispatched by name so callers can tune both at
            once without needing to call them separately.

    Returns:
        df with new columns:
          'trend_regime', 'vol_regime'  (if recompute_regimes=True)
          'market_state'                - combined categorical label
          one-hot columns 'state_<label>' for every label in
              ALL_MARKET_STATES (0/1 int columns, consistent set of
              columns regardless of what's present in this particular df)
    """
    if recompute_regimes:
        trend_kwargs = {k: v for k, v in regime_kwargs.items()
                         if k in ("adx_threshold", "hurst_window",
                                   "hurst_trending_threshold", "hurst_ranging_threshold",
                                   "latest_only")}
        vol_kwargs = {k: v for k, v in regime_kwargs.items()
                      if k in ("window", "lookback_for_percentile",
                                "low_pct", "high_pct", "use_garch")}

        df = calculate_trend_regime(df, adx_col=adx_col, close_col=close_col, **trend_kwargs)
        df = calculate_volatility_regime(df, close_col=close_col, **vol_kwargs)
    else:
        for required in ("trend_regime", "vol_regime"):
            if required not in df.columns:
                raise ValueError(
                    f"recompute_regimes=False but '{required}' column missing. "
                    f"Either compute it first or leave recompute_regimes=True."
                )

    def combine(trend, vol):
        if pd.isna(trend) or pd.isna(vol):
            return pd.NA
        return f"{trend}_{vol}_vol"

    df["market_state"] = [
        combine(t, v) for t, v in zip(df["trend_regime"], df["vol_regime"])
    ]

    # Consistent one-hot columns regardless of which states actually
    # appear in this dataframe, so feature matrices line up across
    # different training runs / assets / date ranges.
    for state in ALL_MARKET_STATES:
        df[f"state_{state}"] = (df["market_state"] == state).astype(int)

    return df


def market_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: return counts and percentage of time spent in
    each market state — useful sanity check when first wiring this up,
    and later as a quick diagnostic on why a strategy is over/under-
    trading (e.g. if 80% of history is 'ranging_high_vol', a pure
    trend-following strategy should expect to sit out most of the time).
    """
    if "market_state" not in df.columns:
        raise ValueError("Run calculate_market_state() first")

    counts = df["market_state"].value_counts(dropna=True)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators.trend.adx import calculate_adx

    n = 600
    rng = np.random.default_rng(3)
    vol = np.concatenate([np.full(300, 0.5), np.full(300, 2.0)])
    close = 100 + np.cumsum(rng.normal(0, 1, n) * vol / 10)
    dummy = pd.DataFrame({
        "close": close,
        "high": close + rng.random(n),
        "low": close - rng.random(n),
    })

    dummy = calculate_adx(dummy)
    dummy = calculate_market_state(dummy)

    print(dummy[["close", "trend_regime", "vol_regime", "market_state"]].tail(10))
    print()
    print(market_state_summary(dummy))