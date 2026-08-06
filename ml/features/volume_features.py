"""
Volume features
------------------
Volume-derived ML features: relative volume, volume trend, dollar
volume, and VWAP/OBV-derived signals (picked up automatically if those
indicator columns are already present).
"""

import numpy as np
import pandas as pd


def calculate_volume_features(
    df: pd.DataFrame,
    zscore_window: int = 20,
    roc_periods: list = None,
) -> pd.DataFrame:
    """
    Append volume-derived features.

    Adds columns:
        VOLUME_ZSCORE_{w}      : rolling z-score of volume
        VOLUME_ROC_{n}         : n-bar volume rate of change (%)
        REL_VOLUME_{w}         : volume / rolling mean volume
        DOLLAR_VOLUME          : close * volume
        VOLUME_TREND_{w}       : slope of volume over w bars (simple linreg)
        OBV_SLOPE_{w}          : slope of OBV over w bars (if OBV column present)
        VWAP_DEVIATION         : % distance of close from VWAP (if VWAP column present)
    """
    df = df.copy()
    required = {"close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if roc_periods is None:
        roc_periods = [1, 5, 10, 20]

    vol = df["volume"]

    roll_mean = vol.rolling(zscore_window).mean()
    roll_std = vol.rolling(zscore_window).std()
    df[f"VOLUME_ZSCORE_{zscore_window}"] = (vol - roll_mean) / roll_std
    df[f"REL_VOLUME_{zscore_window}"] = vol / roll_mean

    for n in roc_periods:
        df[f"VOLUME_ROC_{n}"] = vol.pct_change(n)

    df["DOLLAR_VOLUME"] = df["close"] * vol

    def _slope(x: np.ndarray) -> float:
        idx = np.arange(len(x))
        if np.isnan(x).any():
            return np.nan
        return np.polyfit(idx, x, 1)[0]

    df[f"VOLUME_TREND_{zscore_window}"] = vol.rolling(zscore_window).apply(_slope, raw=True)

    if "OBV" in df.columns:
        df[f"OBV_SLOPE_{zscore_window}"] = df["OBV"].rolling(zscore_window).apply(_slope, raw=True)

    if "VWAP" in df.columns:
        df["VWAP_DEVIATION"] = (df["close"] - df["VWAP"]) / df["VWAP"] * 100

    return df