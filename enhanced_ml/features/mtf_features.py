"""
enhanced_ml/features/mtf_features.py
----------------------------------------
FreqAI concept borrowed: multi-timeframe feature expansion — bring
higher-timeframe trend/volatility context (e.g. 4h, 1d) onto a
lower-timeframe (e.g. 1h) row, so the model can see "what does the
bigger picture look like right now" alongside the base-timeframe
features.

ANTI-LEAKAGE — read before changing anything here:
OHLCV timestamps (per pipeline/data_loader.py) are bar OPEN times, the
standard ccxt/Binance convention. A higher-timeframe bar's OHLC values
(and anything computed from them) are only actually KNOWN once that
bar CLOSES — i.e. at open_timestamp + timeframe_duration, not at
open_timestamp itself. A 4h bar opening at 04:00 covers 04:00-08:00;
its close/high/low/indicators are not real until 08:00. Merging HTF
features onto LTF rows by raw open-timestamp (HTF.timestamp <=
LTF.timestamp) would let the 1h bars at 05:00, 06:00, 07:00 see a 4h
candle that hasn't finished forming yet — a genuine lookahead leak.

Fix: every HTF row's timestamp is shifted forward by that timeframe's
duration BEFORE the merge, so it represents "earliest moment this
bar's data is actually available", then merge_asof(direction='backward')
against the base timeframe's timestamps. A 1h bar at 07:00 gets the 4h
bar that opened at 00:00 (available from 04:00) — the LAST FULLY
CLOSED 4h bar as of 07:00 — never the 04:00-08:00 bar still in progress.

Only a small, curated, already-relative (not absolute-price) column
set is brought in per higher timeframe, matching the earlier finding
that relative/normalized features hold up better under regime shift
than absolute price levels (ADX14, RSI14, ATR_NORM, PRICE_VS_EMA20,
PRICE_VS_EMA50) — deliberately not "hundreds of features", per the
project's incremental-feature-addition rule. Extend
_CURATED_COLUMNS if/when you want more, one at a time, measured.
"""

from functools import partial
from typing import Callable, List

import pandas as pd

from ml.utils.preprocessing import build_feature_matrix
from pipeline.data_loader import load_ohlcv

_CURATED_COLUMNS = ["ADX14", "RSI14", "ATR_NORM", "PRICE_VS_EMA20", "PRICE_VS_EMA50"]

_TIMEFRAME_TO_TIMEDELTA = {
    "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1), "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4), "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12), "1d": pd.Timedelta(days=1),
}


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe not in _TIMEFRAME_TO_TIMEDELTA:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; add it to _TIMEFRAME_TO_TIMEDELTA in "
            f"mtf_features.py (must be exact, this drives the anti-leakage shift)."
        )
    return _TIMEFRAME_TO_TIMEDELTA[timeframe]


def _add_one_higher_timeframe(
    df: pd.DataFrame, symbol: str, higher_timeframe: str, exchange: str, has_volume: bool,
) -> pd.DataFrame:
    htf_raw = load_ohlcv(symbol, higher_timeframe, exchange=exchange)
    htf_features = build_feature_matrix(htf_raw, has_volume=has_volume)

    htf_duration = _timeframe_to_timedelta(higher_timeframe)
    htf_slice = htf_features[["timestamp"] + _CURATED_COLUMNS].copy()
    htf_slice["available_at"] = htf_slice["timestamp"] + htf_duration
    htf_slice = htf_slice.drop(columns=["timestamp"]).sort_values("available_at")

    prefix = higher_timeframe.upper() + "_"
    htf_slice = htf_slice.rename(columns={c: prefix + c for c in _CURATED_COLUMNS})

    original_index = df.index
    left = df[["timestamp"]].copy().sort_values("timestamp")
    merged = pd.merge_asof(
        left, htf_slice, left_on="timestamp", right_on="available_at", direction="backward",
    )
    merged = merged.set_index(left.index).reindex(original_index)
    merged = merged.drop(columns=["timestamp", "available_at"])

    return pd.concat([df, merged], axis=1)


def make_mtf_feature_fn(
    symbol: str, higher_timeframes: List[str], exchange: str, has_volume: bool = True,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Build a function compatible with FeatureEngine.register() — i.e.
    fn(df) -> df, same row count/index — that adds curated features
    from each higher timeframe in `higher_timeframes`.

    Usage:
        engine = FeatureEngine()
        engine.register(make_mtf_feature_fn("BTC/USDT", ["4h", "1d"], exchange="binance"))
        features_df = engine.build(raw_ohlcv_df)

    `df` passed in must already have a 'timestamp' column (true for
    ml.utils.preprocessing.build_feature_matrix's output, which is what
    FeatureEngine.build() produces before this runs).
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError(
                "mtf feature fn requires a 'timestamp' column in the incoming df — "
                "register this after the base build_feature_matrix step, not before."
            )
        result = df
        for htf in higher_timeframes:
            result = _add_one_higher_timeframe(result, symbol, htf, exchange, has_volume)
        return result

    _fn.__name__ = f"mtf_features({','.join(higher_timeframes)})"
    return _fn