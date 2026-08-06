"""
Trend Regime Detection

Answers one question: is the market trending or ranging right now?
This matters because trend-following strategies (EMA crossovers, MACD,
Supertrend) systematically lose money in choppy/ranging conditions, and
mean-reversion strategies (RSI extremes, Bollinger fades) lose money in
strong trends. Feeding the strategy/ML layer a regime label lets it
switch behavior instead of applying one rule set everywhere.

Two independent signals are combined:

1. ADX (Average Directional Index) - already computed in
   indicators/trend/adx.py. High ADX = strong directional move underway,
   regardless of direction.

2. Hurst Exponent - a statistical measure of a time series' tendency to
   trend or mean-revert, computed via rescaled range (R/S) analysis:
     H > 0.5  -> trending / persistent (momentum continues)
     H = 0.5  -> random walk (no memory)
     H < 0.5  -> mean-reverting / anti-persistent (moves tend to reverse)
   Hurst is slower-moving and more statistically grounded than ADX, so
   using both catches short bursts (ADX) and structural regime (Hurst).
"""

import pandas as pd
import numpy as np


def calculate_hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent of a price series using rescaled range
    (R/S) analysis on log returns, over a single window.

    This is deliberately a simplified/fast estimator (log-log regression
    of R/S statistic vs lag) suitable for rolling computation over many
    windows — not a research-grade multi-method estimate.

    Args:
        series: A window of price data (e.g. df['close'].iloc[i-100:i]).
        min_lag: Smallest lag to test (default 2).
        max_lag: Largest lag to test (default 20). Must be < len(series)/2.

    Returns:
        Estimated Hurst exponent (float). Returns np.nan if the window is
        too short or the series is degenerate (zero variance).
    """
    series = series.dropna()
    if len(series) < max_lag * 2:
        return np.nan

    log_returns = np.diff(np.log(series.values))
    if np.std(log_returns) == 0 or len(log_returns) < max_lag:
        return np.nan

    lags = range(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = log_returns[lag:] - log_returns[:-lag]
        std = np.std(diff)
        tau.append(std if std > 0 else np.nan)

    tau = np.array(tau)
    valid = ~np.isnan(tau) & (tau > 0)
    if valid.sum() < 3:
        return np.nan

    log_lags = np.log(list(lags))[valid]
    log_tau = np.log(tau[valid])

    # Slope of log(tau) vs log(lag) ~ Hurst exponent (via variance scaling)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    hurst = slope * 2  # R/S-style scaling correction
    return float(np.clip(hurst, 0.0, 1.0))


def calculate_trend_regime(df: pd.DataFrame, adx_col: str = "ADX14",
                            close_col: str = "close",
                            adx_threshold: float = 25.0,
                            hurst_window: int = 100,
                            hurst_trending_threshold: float = 0.55,
                            hurst_ranging_threshold: float = 0.45) -> pd.DataFrame:
    """
    Add trend-regime columns to the dataframe.

    Requires ADX to already be computed (see indicators/trend/adx.py) —
    this module intentionally does not recompute it, to keep a single
    source of truth for indicator values.

    Args:
        df: DataFrame that already has an ADX column (default 'ADX14').
        adx_col: Name of the ADX column to use.
        close_col: Close price column, used for Hurst exponent.
        adx_threshold: ADX above this = "trending" by the ADX signal
            (default 25, the standard convention).
        hurst_window: Rolling window size for Hurst exponent (default 100
            candles — needs to be reasonably large for a stable estimate).
        hurst_trending_threshold: Hurst above this = trending signal.
        hurst_ranging_threshold: Hurst below this = ranging signal.

    Returns:
        df with new columns:
          'hurst'              - rolling Hurst exponent estimate
          'adx_trending'       - bool, ADX-based trending flag
          'hurst_trending'     - bool, Hurst-based trending flag
          'trend_regime'       - categorical: 'trending', 'ranging', 'mixed'
                                  ('mixed' = ADX and Hurst disagree, treat
                                  with caution / lower confidence downstream)
    """
    if adx_col not in df.columns:
        raise ValueError(f"'{adx_col}' not found — run calculate_adx() first")
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in dataframe")

    df["hurst"] = df[close_col].rolling(window=hurst_window).apply(
        lambda w: calculate_hurst_exponent(pd.Series(w)), raw=False
    )

    df["adx_trending"] = df[adx_col] > adx_threshold
    df["hurst_trending"] = df["hurst"] > hurst_trending_threshold
    hurst_ranging = df["hurst"] < hurst_ranging_threshold

    def classify(row_adx_trend, row_hurst_trend, row_hurst_range):
        if row_adx_trend and row_hurst_trend:
            return "trending"
        if (not row_adx_trend) and row_hurst_range:
            return "ranging"
        return "mixed"

    df["trend_regime"] = [
        classify(a, h, r) for a, h, r in zip(df["adx_trending"], df["hurst_trending"], hurst_ranging)
    ]
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators.trend.adx import calculate_adx

    n = 300
    rng = np.random.default_rng(1)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "high": close + rng.random(n),
        "low": close - rng.random(n),
    })
    dummy = calculate_adx(dummy)
    dummy = calculate_trend_regime(dummy)
    print(dummy[["close", "ADX14", "hurst", "trend_regime"]].tail(10))
    print(dummy["trend_regime"].value_counts())