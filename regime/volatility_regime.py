"""
Volatility Regime Detection

Volatility clusters — calm periods and turbulent periods each tend to
persist (this is the core empirical fact GARCH models are built on).
Knowing which regime you're in matters because:
  - Position sizing should shrink in high-vol regimes (same % move = more
    risk) and can expand in low-vol regimes.
  - Stop-loss/take-profit distances (ATR-based) drift naturally with vol,
    but the *regime label itself* is a useful categorical ML feature that
    raw ATR doesn't fully capture (ATR is a point-in-time value; regime
    captures persistence).
  - Squeeze regimes (low vol) often precede breakouts — useful entry
    timing signal when combined with Bollinger/Keltner squeeze.

Two approaches are provided:

1. GARCH(1,1) via the `arch` package — the statistically proper way to
   model volatility clustering and forecast forward volatility. Falls
   back automatically if `arch` isn't installed (it's an extra
   dependency, not in base pandas/numpy).

2. Rolling realized volatility + percentile bucketing — simpler, no
   extra dependency, and often good enough for regime *labeling* (as
   opposed to forecasting). Used as the default and as the fallback.
"""

import pandas as pd
import numpy as np


def calculate_realized_volatility(df: pd.DataFrame, close_col: str = "close",
                                   window: int = 20) -> pd.Series:
    """
    Rolling realized volatility: standard deviation of log returns,
    annualization-agnostic (kept in per-candle units so it composes
    cleanly with other per-candle features).
    """
    log_returns = np.log(df[close_col] / df[close_col].shift(1))
    return log_returns.rolling(window=window, min_periods=window).std()


def calculate_volatility_regime(df: pd.DataFrame, close_col: str = "close",
                                 window: int = 20,
                                 lookback_for_percentile: int = 500,
                                 low_pct: float = 0.33,
                                 high_pct: float = 0.67,
                                 use_garch: bool = False) -> pd.DataFrame:
    """
    Add volatility-regime columns to the dataframe.

    Args:
        df: DataFrame with a close price column.
        window: Rolling window for realized volatility (default 20).
        lookback_for_percentile: How far back to look when ranking current
            volatility into low/medium/high (default 500 candles). A
            rolling percentile is used rather than a fixed threshold so
            the regime definition adapts as the asset's baseline
            volatility drifts over months (crypto vol regimes shift a lot
            year to year).
        low_pct / high_pct: Percentile cutoffs for low/high buckets
            (default bottom/top third).
        use_garch: If True, attempt to use a rolling GARCH(1,1) forecast
            instead of realized vol. Requires the `arch` package; silently
            falls back to realized volatility if unavailable or if fitting
            fails on a given window (GARCH can fail to converge on short
            or degenerate windows).

    Returns:
        df with new columns:
          'realized_vol'     - rolling realized volatility
          'vol_percentile'   - current vol's rolling percentile rank (0-1)
          'vol_regime'       - categorical: 'low', 'medium', 'high'
          'garch_vol'        - forecasted conditional vol (only if
                                use_garch=True and fitting succeeded;
                                otherwise this column is omitted)
    """
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in dataframe")

    realized_vol = calculate_realized_volatility(df, close_col=close_col, window=window)
    df["realized_vol"] = realized_vol

    vol_series = realized_vol
    if use_garch:
        garch_vol = _try_garch_volatility(df[close_col])
        if garch_vol is not None:
            df["garch_vol"] = garch_vol
            vol_series = garch_vol
        # else: silently keep realized_vol as the basis (fallback)

    df["vol_percentile"] = vol_series.rolling(
        window=lookback_for_percentile, min_periods=window
    ).rank(pct=True)

    def bucket(p):
        if pd.isna(p):
            return np.nan
        if p <= low_pct:
            return "low"
        if p >= high_pct:
            return "high"
        return "medium"

    df["vol_regime"] = df["vol_percentile"].apply(bucket)
    return df


def _try_garch_volatility(close: pd.Series):
    """
    Attempt to fit a GARCH(1,1) model and return in-sample conditional
    volatility. Returns None if the `arch` package is unavailable or
    fitting fails, so callers can fall back gracefully rather than
    crashing the whole pipeline over an optional dependency.

    Note: this fits GARCH once over the full series for conditional
    volatility extraction. For genuine forward-looking forecasts inside
    a walk-forward backtest, refit on each expanding/rolling window
    instead of calling this on the full history.
    """
    try:
        import importlib
        arch_mod = importlib.import_module("arch")
        arch_model = arch_mod.arch_model
    except (ImportError, ModuleNotFoundError):
        return None

    returns = 100 * np.log(close / close.shift(1)).dropna()
    if len(returns) < 100:
        return None

    try:
        model = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
        result = model.fit(disp="off")
        cond_vol = result.conditional_volatility / 100  # undo the *100 scaling
        cond_vol = cond_vol.reindex(close.index)  # align back, leaves NaN for first row
        return cond_vol
    except Exception:
        return None


if __name__ == "__main__":
    n = 600
    rng = np.random.default_rng(2)
    # Simulate a volatility regime shift partway through
    vol = np.concatenate([np.full(300, 0.5), np.full(300, 2.0)])
    close = 100 + np.cumsum(rng.normal(0, 1, n) * vol / 10)
    dummy = pd.DataFrame({"close": close})

    dummy = calculate_volatility_regime(dummy)
    print(dummy[["close", "realized_vol", "vol_percentile", "vol_regime"]].tail(10))
    print(dummy["vol_regime"].value_counts())