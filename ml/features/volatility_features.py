"""
Volatility features
----------------------
Realized-volatility estimators (close-to-close, Parkinson, Garman-Klass),
vol-of-vol, and normalized ATR/Bollinger-width features (auto-picked up
if those indicator columns are already present).
"""

import numpy as np
import pandas as pd


def calculate_volatility_features(
    df: pd.DataFrame,
    windows: list = None,
    price_col: str = "close",
    annualize_factor: float = None,
) -> pd.DataFrame:
    """
    Append volatility features.

    Parameters
    ----------
    windows : rolling windows for realized vol (default [10, 20, 50]).
    annualize_factor : if given (e.g. sqrt(365*24) for hourly bars),
        multiplies realized vol columns to annualize them. Left as None
        (raw per-bar vol) by default since bar frequency varies by asset.

    Adds columns:
        REALIZED_VOL_{w}       : rolling std of log returns
        PARKINSON_VOL_{w}      : high-low range estimator (more efficient than close-to-close)
        GARMAN_KLASS_VOL_{w}   : OHLC-based estimator, captures intrabar drift
        VOL_OF_VOL_{w}         : rolling std of REALIZED_VOL_{w} itself
        ATR_NORM                : ATR14 / close, if ATR14 column present
        BB_WIDTH_ZSCORE_{w}     : rolling z-score of BB_width, if present
    """
    df = df.copy()
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if windows is None:
        windows = [10, 20, 50]

    log_ret = np.log(df[price_col] / df[price_col].shift(1))
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    for w in windows:
        rv = log_ret.rolling(w).std()
        if annualize_factor:
            rv = rv * annualize_factor
        df[f"REALIZED_VOL_{w}"] = rv

        df[f"PARKINSON_VOL_{w}"] = np.sqrt(
            (1 / (4 * w * np.log(2))) * (log_hl ** 2).rolling(w).sum()
        )

        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        df[f"GARMAN_KLASS_VOL_{w}"] = np.sqrt(gk.rolling(w).mean().clip(lower=0))

        df[f"VOL_OF_VOL_{w}"] = df[f"REALIZED_VOL_{w}"].rolling(w).std()

    if "ATR14" in df.columns:
        df["ATR_NORM"] = df["ATR14"] / df[price_col]

    if "BB_width" in df.columns:
        w = windows[0]
        roll_mean = df["BB_width"].rolling(w).mean()
        roll_std = df["BB_width"].rolling(w).std()
        df[f"BB_WIDTH_ZSCORE_{w}"] = (df["BB_width"] - roll_mean) / roll_std

    return df