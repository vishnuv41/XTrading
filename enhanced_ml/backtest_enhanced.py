"""
enhanced_ml/backtest_enhanced.py

V4 TP-BEFORE-SL strategy backtester.

Main improvements over V3
--------------------------
1. Correctly resolves the persisted training final-test window from the
   `timestamp` column instead of the RangeIndex.

2. Uses direct XGBoost + CatBoost probability blending:
       XGB 70%
       CAT 30%

3. Replaces the V3 "everything must pass" AND gate with a weighted
   multi-timeframe score.

4. Model probability receives the largest weight.

5. 1H is the primary entry timeframe.
   4H and 1D are soft directional confirmation.

6. ATR-based SL/TP.

7. Cooldown between trades.

8. Prevents multiple simultaneous positions.

9. Includes transaction costs.

10. Saves:
       trades.csv
       equity.csv
       metrics.json

11. Does NOT tune parameters using the final-test window.

Example
-------
python -m enhanced_ml.backtest_enhanced ^
    --symbol SOL/USDT ^
    --timeframe 1h ^
    --higher-timeframes 4h 1d ^
    --side long ^
    --sl-mult 1.5 ^
    --risk-reward 2.5 ^
    --max-holding 48 ^
    --holdout-frac 0.30 ^
    --confidence-threshold 0.65 ^
    --transaction-cost-bps 10 ^
    --ensemble-mode xgb_cat ^
    --xgb-weight 0.70 ^
    --catboost-weight 0.30 ^
    --min-score 10 ^
    --cooldown-bars 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.data_loader import load_ohlcv
from enhanced_ml.train_enhanced import build_mtf_feature_fn
from ml.predict import load_training_artifacts


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> str | None:
    """
    Find a column case-insensitively.

    Also supports a few common naming variations.
    """
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # Loose normalized comparison
    compact = {
        "".join(ch for ch in str(c).lower() if ch.isalnum()): c
        for c in df.columns
    }

    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())

        if key in compact:
            return compact[key]

    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}. "
            f"Available columns include: {list(df.columns)[:50]}"
        )

    return None


def _get_series(
    df: pd.DataFrame,
    candidates: list[str],
    default: float = np.nan,
) -> pd.Series:
    col = _find_column(df, candidates)

    if col is None:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns and guarantee a UTC timestamp column.
    """
    result = df.copy()

    timestamp_col = _find_column(
        result,
        ["timestamp", "datetime", "date", "time"],
        required=True,
    )

    if timestamp_col != "timestamp":
        result["timestamp"] = result[timestamp_col]

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError("OHLCV contains invalid timestamps.")

    required = ["open", "high", "low", "close"]

    for col in required:
        actual = _find_column(result, [col], required=True)

        if actual != col:
            result[col] = result[actual]

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        )

    if "volume" in result.columns:
        result["volume"] = pd.to_numeric(
            result["volume"],
            errors="coerce",
        )

    result = (
        result
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------------------
# Correct final-test resolver
# ---------------------------------------------------------------------

def _resolve_final_test_window(
    df: pd.DataFrame,
    holdout_frac: float,
    metadata: dict | None = None,
) -> tuple[int, int]:
    """
    Resolve the exact raw-OHLCV final-test window.

    IMPORTANT
    ---------
    load_ohlcv() currently returns a DataFrame with a RangeIndex.

    Training metadata stores final-test boundaries as timestamps.

    Therefore metadata timestamps MUST be resolved against the
    `timestamp` column, never against df.index.

    Returns
    -------
    (start_idx, end_idx)
        Inclusive positional indices.
    """

    if not (0.0 < holdout_frac < 1.0):
        raise ValueError(
            "holdout_frac must be between 0 and 1."
        )

    if df.empty:
        raise ValueError(
            "Cannot resolve final-test window from empty DataFrame."
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "Raw OHLCV DataFrame must contain a timestamp column."
        )

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Raw OHLCV timestamp column contains invalid timestamps."
        )

    timestamps = timestamps.reset_index(drop=True)

    if not timestamps.is_monotonic_increasing:
        order = np.argsort(
            timestamps.to_numpy()
        )

        sorted_positions = np.asarray(
            order,
            dtype=int,
        )

        sorted_timestamps = (
            timestamps.iloc[sorted_positions]
            .reset_index(drop=True)
        )
    else:
        sorted_positions = np.arange(
            len(df),
            dtype=int,
        )

        sorted_timestamps = timestamps

    # -------------------------------------------------------------
    # Metadata-first resolution
    # -------------------------------------------------------------

    if metadata:
        start_value = metadata.get(
            "final_test_start"
        )

        end_value = metadata.get(
            "final_test_end"
        )

        if start_value is None:
            for key in (
                "final_test_start_time",
                "test_start",
                "test_start_time",
            ):
                if metadata.get(key) is not None:
                    start_value = metadata[key]
                    break

        if start_value is not None:
            try:
                start_ts = pd.Timestamp(
                    start_value
                )

                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize("UTC")
                else:
                    start_ts = start_ts.tz_convert("UTC")

                start_sorted = int(
                    sorted_timestamps.searchsorted(
                        start_ts,
                        side="left",
                    )
                )

                if start_sorted >= len(
                    sorted_timestamps
                ):
                    raise ValueError(
                        "final_test_start is after available OHLCV."
                    )

                start_idx = int(
                    sorted_positions[start_sorted]
                )

                if end_value is not None:
                    end_ts = pd.Timestamp(
                        end_value
                    )

                    if end_ts.tzinfo is None:
                        end_ts = end_ts.tz_localize("UTC")
                    else:
                        end_ts = end_ts.tz_convert("UTC")

                    end_sorted = int(
                        sorted_timestamps.searchsorted(
                            end_ts,
                            side="right",
                        ) - 1
                    )

                    if end_sorted < 0:
                        raise ValueError(
                            "final_test_end is before available OHLCV."
                        )

                    end_idx = int(
                        sorted_positions[end_sorted]
                    )
                else:
                    end_idx = int(
                        sorted_positions[-1]
                    )

                if start_idx > end_idx:
                    raise ValueError(
                        f"Invalid final-test window: "
                        f"{start_idx} > {end_idx}"
                    )

                return start_idx, end_idx

            except Exception as exc:
                print(
                    "WARNING: Could not resolve persisted "
                    f"final-test metadata; falling back. Reason: {exc}"
                )

    # -------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------

    start_idx = max(
        0,
        int(len(df) * (1.0 - holdout_frac)),
    )

    end_idx = len(df) - 1

    return start_idx, end_idx


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def _load_metadata(
    artifact_dir: Path,
) -> dict | None:

    metadata_path = artifact_dir / "metadata.json"

    if not metadata_path.exists():
        print(
            f"WARNING: metadata.json not found at "
            f"{metadata_path}"
        )
        return None

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception as exc:
        print(
            f"WARNING: Could not read metadata.json: {exc}"
        )

        return None


# ---------------------------------------------------------------------
# Technical calculations
# ---------------------------------------------------------------------

def _calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        period,
        min_periods=period,
    ).mean()


def _calculate_regime(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate simple trend/regime information for a timeframe.

    This intentionally does not use future candles.
    """

    df = raw.copy()

    close = df["close"]

    df["ema20_regime"] = (
        close.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    df["ema50_regime"] = (
        close.ewm(
            span=50,
            adjust=False,
        ).mean()
    )

    df["ema100_regime"] = (
        close.ewm(
            span=100,
            adjust=False,
        ).mean()
    )

    df["ema20_slope"] = (
        df["ema20_regime"]
        .pct_change(3)
    )

    df["ema50_slope"] = (
        df["ema50_regime"]
        .pct_change(3)
    )

    # ADX
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = (
        df["high"]
        - df["high"].shift(1)
    )

    down_move = (
        df["low"].shift(1)
        - df["low"]
    )

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr = tr.rolling(
        14,
        min_periods=14,
    ).mean()

    plus_di = (
        100.0
        * plus_dm.rolling(
            14,
            min_periods=14,
        ).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100.0
        * minus_dm.rolling(
            14,
            min_periods=14,
        ).mean()
        / atr.replace(0, np.nan)
    )

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    df["adx_regime"] = dx.rolling(
        14,
        min_periods=14,
    ).mean()

    df["bull_trend"] = (
        (df["ema20_regime"] > df["ema50_regime"])
        & (
            df["ema20_slope"] > 0
        )
    )

    return df


# ---------------------------------------------------------------------
# Higher timeframe alignment
# ---------------------------------------------------------------------

def _build_htf_regime(
    raw: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:

    regime = _calculate_regime(raw)

    regime = regime[
        [
            "timestamp",
            "ema20_regime",
            "ema50_regime",
            "ema100_regime",
            "ema20_slope",
            "ema50_slope",
            "adx_regime",
            "bull_trend",
        ]
    ].copy()

    duration = _timeframe_to_timedelta(
        timeframe
    )

    # IMPORTANT:
    # A candle's features only become available after that candle closes.
    regime["available_at"] = (
        regime["timestamp"] + duration
    )

    regime = regime.sort_values(
        "available_at"
    )

    regime = regime.rename(
        columns={
            "ema20_regime":
                f"{timeframe.upper()}_EMA20",
            "ema50_regime":
                f"{timeframe.upper()}_EMA50",
            "ema100_regime":
                f"{timeframe.upper()}_EMA100",
            "ema20_slope":
                f"{timeframe.upper()}_SLOPE20",
            "ema50_slope":
                f"{timeframe.upper()}_SLOPE50",
            "adx_regime":
                f"{timeframe.upper()}_ADX",
            "bull_trend":
                f"{timeframe.upper()}_BULL",
        }
    )

    return regime


def _timeframe_to_timedelta(
    timeframe: str,
) -> pd.Timedelta:

    tf = timeframe.strip().lower()

    if tf.endswith("m"):
        return pd.Timedelta(
            minutes=int(tf[:-1])
        )

    if tf.endswith("h"):
        return pd.Timedelta(
            hours=int(tf[:-1])
        )

    if tf.endswith("d"):
        return pd.Timedelta(
            days=int(tf[:-1])
        )

    if tf.endswith("w"):
        return pd.Timedelta(
            weeks=int(tf[:-1])
        )

    raise ValueError(
        f"Unsupported timeframe: {timeframe}"
    )


def _merge_htf_regime(
    base: pd.DataFrame,
    htf_regime: pd.DataFrame,
) -> pd.DataFrame:

    left = base[
        ["timestamp"]
    ].copy()

    left["_position"] = np.arange(
        len(left)
    )

    left = left.sort_values(
        "timestamp"
    )

    right = htf_regime.copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )

    merged = (
        merged
        .sort_values("_position")
        .drop(
            columns=[
                "_position",
                "timestamp",
                "available_at",
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    result = base.reset_index(
        drop=True
    ).copy()

    for col in merged.columns:
        result[col] = merged[col].values

    return result


# ---------------------------------------------------------------------
# Model probability
# ---------------------------------------------------------------------

def _get_model_probabilities(
    artifacts: dict,
    X: pd.DataFrame,
    mode: str,
    xgb_weight: float,
    catboost_weight: float,
) -> np.ndarray:

    ensemble = artifacts["ensemble"]

    if mode == "stacking":
        return ensemble.predict_proba(X)

    base = ensemble.fitted_base_models_

    if (
        "xgboost" not in base
        or "catboost" not in base
    ):
        raise RuntimeError(
            "Artifacts do not contain both "
            "xgboost and catboost models."
        )

    xgb = base[
        "xgboost"
    ].predict_proba(X)

    cat = base[
        "catboost"
    ].predict_proba(X)

    total = (
        xgb_weight
        + catboost_weight
    )

    if total <= 0:
        raise ValueError(
            "XGB/CatBoost weights must have positive sum."
        )

    xgb_weight /= total
    catboost_weight /= total

    return (
        xgb_weight * xgb
        + catboost_weight * cat
    )


# ---------------------------------------------------------------------
# V4 scoring
# ---------------------------------------------------------------------

def _calculate_v4_score(
    row: pd.Series,
    probability: float,
    confidence_threshold: float,
    min_adx: float,
    rsi_min: float,
    rsi_max: float,
    max_distance_atr: float,
    atr_percentile_min: float,
    atr_percentile_max: float,
) -> tuple[int, dict[str, float]]:
    """
    Weighted score.

    Maximum theoretical score = 16.

    Model:
        +4 if probability >= threshold
        +1 additional if probability >= 0.75

    1H:
        +2 trend
        +1 slope

    4H:
        +2 trend
        +1 slope

    1D:
        +1 trend

    Other:
        +1 ADX
        +1 RSI
        +1 extension
        +1 ATR regime
    """

    score = 0

    components: dict[str, float] = {}

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    if probability >= confidence_threshold:
        score += 4
        components["MODEL"] = 4
    else:
        components["MODEL"] = 0

    if probability >= 0.75:
        score += 1
        components["MODEL_BONUS"] = 1
    else:
        components["MODEL_BONUS"] = 0

    # -------------------------------------------------------------
    # 1H trend
    # -------------------------------------------------------------

    h1_ema20 = _safe_float(
        row.get("1H_EMA20", np.nan),
        np.nan,
    )

    h1_ema50 = _safe_float(
        row.get("1H_EMA50", np.nan),
        np.nan,
    )

    h1_slope = _safe_float(
        row.get("1H_SLOPE20", np.nan),
        np.nan,
    )

    if (
        math.isfinite(h1_ema20)
        and math.isfinite(h1_ema50)
        and h1_ema20 > h1_ema50
    ):
        score += 2
        components["1H_TREND"] = 2
    else:
        components["1H_TREND"] = 0

    if (
        math.isfinite(h1_slope)
        and h1_slope > 0
    ):
        score += 1
        components["1H_SLOPE"] = 1
    else:
        components["1H_SLOPE"] = 0

    # -------------------------------------------------------------
    # 4H trend
    # -------------------------------------------------------------

    h4_ema20 = _safe_float(
        row.get("4H_EMA20", np.nan),
        np.nan,
    )

    h4_ema50 = _safe_float(
        row.get("4H_EMA50", np.nan),
        np.nan,
    )

    h4_slope = _safe_float(
        row.get("4H_SLOPE20", np.nan),
        np.nan,
    )

    if (
        math.isfinite(h4_ema20)
        and math.isfinite(h4_ema50)
        and h4_ema20 > h4_ema50
    ):
        score += 2
        components["4H_TREND"] = 2
    else:
        components["4H_TREND"] = 0

    if (
        math.isfinite(h4_slope)
        and h4_slope > 0
    ):
        score += 1
        components["4H_SLOPE"] = 1
    else:
        components["4H_SLOPE"] = 0

    # -------------------------------------------------------------
    # 1D trend
    # -------------------------------------------------------------

    d1_ema20 = _safe_float(
        row.get("1D_EMA20", np.nan),
        np.nan,
    )

    d1_ema50 = _safe_float(
        row.get("1D_EMA50", np.nan),
        np.nan,
    )

    if (
        math.isfinite(d1_ema20)
        and math.isfinite(d1_ema50)
        and d1_ema20 > d1_ema50
    ):
        score += 1
        components["1D_TREND"] = 1
    else:
        components["1D_TREND"] = 0

    # -------------------------------------------------------------
    # ADX
    # -------------------------------------------------------------

    adx = _safe_float(
        row.get("ADX14", np.nan),
        np.nan,
    )

    if not math.isfinite(adx):
        adx = _safe_float(
            row.get("ADX", np.nan),
            np.nan,
        )

    if (
        math.isfinite(adx)
        and adx >= min_adx
    ):
        score += 1
        components["ADX"] = 1
    else:
        components["ADX"] = 0

    # -------------------------------------------------------------
    # RSI
    # -------------------------------------------------------------

    rsi = _safe_float(
        row.get("RSI14", np.nan),
        np.nan,
    )

    if not math.isfinite(rsi):
        rsi = _safe_float(
            row.get("RSI", np.nan),
            np.nan,
        )

    if (
        math.isfinite(rsi)
        and rsi_min <= rsi <= rsi_max
    ):
        score += 1
        components["RSI"] = 1
    else:
        components["RSI"] = 0

    # -------------------------------------------------------------
    # EMA extension
    # -------------------------------------------------------------

    close = _safe_float(
        row.get("close", np.nan),
        np.nan,
    )

    atr = _safe_float(
        row.get("ATR14", np.nan),
        np.nan,
    )

    if not math.isfinite(atr):
        atr = _safe_float(
            row.get("atr14", np.nan),
            np.nan,
        )

    if (
        math.isfinite(close)
        and math.isfinite(h1_ema20)
        and math.isfinite(atr)
        and atr > 0
    ):
        distance_atr = (
            close - h1_ema20
        ) / atr

        if (
            distance_atr <= max_distance_atr
            and distance_atr >= -1.0
        ):
            score += 1
            components["EXTENSION"] = 1
        else:
            components["EXTENSION"] = 0

    else:
        components["EXTENSION"] = 0

    # -------------------------------------------------------------
    # ATR percentile
    # -------------------------------------------------------------

    atr_pct = _safe_float(
        row.get("ATR_PERCENTILE", np.nan),
        np.nan,
    )

    if (
        math.isfinite(atr_pct)
        and atr_percentile_min
        <= atr_pct
        <= atr_percentile_max
    ):
        score += 1
        components["ATR_REGIME"] = 1
    else:
        components["ATR_REGIME"] = 0

    components["TOTAL"] = score

    return score, components


# ---------------------------------------------------------------------
# ATR percentile
# ---------------------------------------------------------------------

def _add_atr_percentile(
    df: pd.DataFrame,
    lookback: int = 500,
) -> pd.DataFrame:

    result = df.copy()

    atr = _get_series(
        result,
        ["ATR14", "atr14", "ATR"],
    )

    def percentile_rank(
        values: pd.Series,
    ) -> float:

        current = values.iloc[-1]

        if not np.isfinite(current):
            return np.nan

        clean = values.dropna()

        if len(clean) < 20:
            return np.nan

        return (
            100.0
            * (clean <= current).mean()
        )

    result["ATR_PERCENTILE"] = (
        atr
        .rolling(
            lookback,
            min_periods=50,
        )
        .apply(
            percentile_rank,
            raw=False,
        )
    )

    return result


# ---------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------

def _simulate_long_trade(
    df: pd.DataFrame,
    entry_idx: int,
    sl_price: float,
    tp_price: float,
    max_holding: int,
) -> tuple[int, str, float]:
    """
    Simulate a long trade.

    Returns
    -------
    exit_idx
    exit_reason
    exit_price
    """

    last_idx = min(
        len(df) - 1,
        entry_idx + max_holding,
    )

    for i in range(
        entry_idx + 1,
        last_idx + 1,
    ):
        high = float(
            df.iloc[i]["high"]
        )

        low = float(
            df.iloc[i]["low"]
        )

        # Conservative ambiguity handling:
        # if both TP and SL are touched in one candle,
        # assume SL first.
        if (
            low <= sl_price
            and high >= tp_price
        ):
            return (
                i,
                "sl",
                sl_price,
            )

        if low <= sl_price:
            return (
                i,
                "sl",
                sl_price,
            )

        if high >= tp_price:
            return (
                i,
                "tp",
                tp_price,
            )

    exit_idx = last_idx

    return (
        exit_idx,
        "timeout",
        float(
            df.iloc[exit_idx]["close"]
        ),
    )


# ---------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------

def run_backtest(
    symbol: str,
    timeframe: str,
    higher_timeframes: list[str],
    side: str,
    sl_mult: float,
    risk_reward: float,
    max_holding: int,
    holdout_frac: float,
    confidence_threshold: float,
    transaction_cost_bps: float,
    ensemble_mode: str,
    xgb_weight: float,
    catboost_weight: float,
    min_score: int,
    min_adx: float,
    rsi_min: float,
    rsi_max: float,
    max_distance_atr: float,
    atr_percentile_min: float,
    atr_percentile_max: float,
    cooldown_bars: int,
) -> dict:

    if side.lower() != "long":
        raise NotImplementedError(
            "V4 currently implements long-side trading."
        )

    print(
        f"Loading {symbol} {timeframe}..."
    )

    raw = load_ohlcv(
        symbol,
        timeframe,
        exchange="binance",
    )

    raw = _normalize_ohlcv(raw)

    print(
        f"OHLCV rows: {len(raw)}"
    )

    artifact_dir = Path(
        "models_artifacts"
    ) / (
        symbol.replace("/", "")
        + "_"
        + timeframe
        + "_v2"
    )

    print(
        f"Loading V2 artifacts: "
        f"{artifact_dir}"
    )

    artifacts = load_training_artifacts(
        str(artifact_dir)
    )

    metadata = _load_metadata(
        artifact_dir
    )

    # -------------------------------------------------------------
    # Build model features
    # -------------------------------------------------------------

    print(
        "Rebuilding V2 MTF predictions..."
    )

    feature_fn = build_mtf_feature_fn(
        symbol,
        higher_timeframes,
        exchange="binance",
    )

    features = feature_fn(
        raw.copy(),
        has_volume=True,
    )

    feature_columns = artifacts[
        "feature_columns"
    ]

    missing_features = [
        c
        for c in feature_columns
        if c not in features.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Artifact feature mismatch. "
            f"Missing {len(missing_features)} features: "
            f"{missing_features[:20]}"
        )

    # -------------------------------------------------------------
    # Model-valid rows
    # -------------------------------------------------------------

    X_all = features[
        feature_columns
    ].copy()

    valid_model = (
        X_all.notna().all(axis=1)
    )

    print(
        f"Model-valid feature rows: "
        f"{int(valid_model.sum())}"
    )

    # -------------------------------------------------------------
    # Calculate indicators required by V4
    # -------------------------------------------------------------

    features["ATR14"] = _get_series(
        features,
        ["ATR14", "atr14", "ATR"],
    )

    features["RSI14"] = _get_series(
        features,
        ["RSI14", "rsi14", "RSI"],
    )

    features["ADX14"] = _get_series(
        features,
        ["ADX14", "adx14", "ADX"],
    )

    features["EMA20"] = _get_series(
        features,
        ["EMA20", "ema20"],
    )

    features = _add_atr_percentile(
        features
    )

    # -------------------------------------------------------------
    # Higher timeframe regimes
    # -------------------------------------------------------------

    print(
        "Loading higher timeframe regime data..."
    )

    for htf in higher_timeframes:

        htf_raw = load_ohlcv(
            symbol,
            htf,
            exchange="binance",
        )

        htf_raw = _normalize_ohlcv(
            htf_raw
        )

        regime = _build_htf_regime(
            htf_raw,
            htf,
        )

        features = _merge_htf_regime(
            features,
            regime,
        )

    # -------------------------------------------------------------
    # Final-test window
    # -------------------------------------------------------------

    start_idx, end_idx = (
        _resolve_final_test_window(
            raw,
            holdout_frac,
            metadata,
        )
    )

    start_ts = raw.iloc[
        start_idx
    ]["timestamp"]

    end_ts = raw.iloc[
        end_idx
    ]["timestamp"]

    print()

    if metadata:
        print(
            "Training metadata final-test start: "
            f"{metadata.get('final_test_start')}"
        )

        print(
            "Training metadata final-test end:   "
            f"{metadata.get('final_test_end')}"
        )

        print(
            "Training metadata test samples:     "
            f"{metadata.get('final_test_n_samples')}"
        )

    print(
        "Resolved raw OHLCV test start:       "
        f"{start_ts}"
    )

    print(
        "Resolved raw OHLCV test end:         "
        f"{end_ts}"
    )

    print(
        "Resolved raw OHLCV test rows:        "
        f"{end_idx - start_idx + 1}"
    )

    print(
        "Resolved raw OHLCV test indices:     "
        f"{start_idx} -> {end_idx}"
    )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    model_positions = np.flatnonzero(
        valid_model.to_numpy()
    )

    model_X = X_all.loc[
        valid_model
    ].copy()

    probabilities = _get_model_probabilities(
        artifacts,
        model_X,
        ensemble_mode,
        xgb_weight,
        catboost_weight,
    )

    if probabilities.shape[1] < 2:
        raise RuntimeError(
            "Model probability output does not contain "
            "binary class probabilities."
        )

    p1 = probabilities[:, 1]

    # Map predictions back to raw dataframe position
    probability_by_raw_idx = np.full(
        len(raw),
        np.nan,
        dtype=float,
    )

    probability_by_raw_idx[
        model_positions
    ] = p1

    features["MODEL_P1"] = (
        probability_by_raw_idx
    )

    # -------------------------------------------------------------
    # Backtest
    # -------------------------------------------------------------

    test_mask = np.zeros(
        len(raw),
        dtype=bool,
    )

    test_mask[
        start_idx:end_idx + 1
    ] = True

    print()
    print("=" * 78)
    print(
        "V4 WEIGHTED TP-BEFORE-SL BACKTEST"
    )
    print("=" * 78)

    print(
        f"Symbol:               {symbol}"
    )

    print(
        f"Timeframe:            {timeframe}"
    )

    print(
        f"Side:                 {side}"
    )

    print(
        f"Final-test start:     {start_ts}"
    )

    print(
        f"Final-test end:       {end_ts}"
    )

    print(
        f"Final-test rows:      "
        f"{end_idx - start_idx + 1}"
    )

    print(
        f"SL:                   "
        f"{sl_mult:.3f} x ATR14"
    )

    print(
        f"TP:                   "
        f"{sl_mult * risk_reward:.3f} x ATR14"
    )

    print(
        f"Max holding:          "
        f"{max_holding} bars"
    )

    print(
        f"Cost:                 "
        f"{transaction_cost_bps:.2f} bps/side"
    )

    print(
        f"Model confidence:     "
        f"{confidence_threshold:.3f}"
    )

    print(
        f"Minimum V4 score:     "
        f"{min_score}"
    )

    print(
        f"XGB/CAT:              "
        f"{xgb_weight:.2f}/{catboost_weight:.2f}"
    )

    print(
        f"Min ADX:              "
        f"{min_adx:.2f}"
    )

    print(
        f"RSI range:            "
        f"{rsi_min:.1f} - {rsi_max:.1f}"
    )

    print(
        f"Max EMA20 distance:   "
        f"{max_distance_atr:.2f} ATR"
    )

    print(
        f"ATR percentile:       "
        f"{atr_percentile_min:.0f} - "
        f"{atr_percentile_max:.0f}"
    )

    print(
        f"Cooldown:             "
        f"{cooldown_bars} bars"
    )

    print("=" * 78)

    # -------------------------------------------------------------
    # Score/filter statistics
    # -------------------------------------------------------------

    rejection_counts = {
        "model_probability": 0,
        "score": 0,
        "ADX": 0,
        "RSI": 0,
        "ATR_REGIME": 0,
        "BELOW_EMA": 0,
        "OVEREXTENDED": 0,
        "1H_TREND": 0,
        "4H_TREND": 0,
        "1D_TREND": 0,
    }

    accepted_entries = []

    # -------------------------------------------------------------
    # Position state
    # -------------------------------------------------------------

    in_position = False

    entry_idx = None
    entry_price = None
    entry_atr = None
    entry_score = None
    entry_probability = None

    next_allowed_entry = start_idx

    trades = []

    # -------------------------------------------------------------
    # Equity
    # -------------------------------------------------------------

    equity = 1.0

    equity_records = []

    # -------------------------------------------------------------
    # Transaction cost
    # -------------------------------------------------------------

    cost_rate = (
        transaction_cost_bps
        / 10000.0
    )

    # -------------------------------------------------------------
    # Process candles
    # -------------------------------------------------------------

    for i in range(
        start_idx,
        end_idx + 1,
    ):

        row = features.iloc[i]

        close = _safe_float(
            row.get("close", np.nan),
            np.nan,
        )

        high = _safe_float(
            row.get("high", np.nan),
            np.nan,
        )

        low = _safe_float(
            row.get("low", np.nan),
            np.nan,
        )

        atr = _safe_float(
            row.get("ATR14", np.nan),
            np.nan,
        )

        probability = _safe_float(
            row.get("MODEL_P1", np.nan),
            np.nan,
        )

        timestamp = raw.iloc[
            i
        ]["timestamp"]

        # ---------------------------------------------------------
        # Existing position
        #
        # We simulate exits immediately when a trade is open.
        # ---------------------------------------------------------

        if in_position:

            bars_held = (
                i - entry_idx
            )

            sl_price = (
                entry_price
                - sl_mult * entry_atr
            )

            tp_price = (
                entry_price
                + sl_mult
                * risk_reward
                * entry_atr
            )

            exit_reason = None
            exit_price = None

            # Conservative intrabar rule:
            # if both touched, SL wins.
            if (
                low <= sl_price
                and high >= tp_price
            ):
                exit_reason = "sl"
                exit_price = sl_price

            elif low <= sl_price:
                exit_reason = "sl"
                exit_price = sl_price

            elif high >= tp_price:
                exit_reason = "tp"
                exit_price = tp_price

            elif bars_held >= max_holding:
                exit_reason = "timeout"
                exit_price = close

            if exit_reason is not None:

                gross_return = (
                    exit_price
                    / entry_price
                    - 1.0
                )

                # Entry + exit costs
                net_return = (
                    gross_return
                    - 2.0 * cost_rate
                )

                equity *= (
                    1.0 + net_return
                )

                trades.append(
                    {
                        "entry_index":
                            entry_idx,
                        "entry_timestamp":
                            raw.iloc[
                                entry_idx
                            ]["timestamp"],
                        "entry_price":
                            entry_price,
                        "exit_index":
                            i,
                        "exit_timestamp":
                            timestamp,
                        "exit_price":
                            exit_price,
                        "bars_held":
                            bars_held,
                        "exit_reason":
                            exit_reason,
                        "entry_probability":
                            entry_probability,
                        "entry_score":
                            entry_score,
                        "atr":
                            entry_atr,
                        "sl_price":
                            sl_price,
                        "tp_price":
                            tp_price,
                        "gross_return":
                            gross_return,
                        "transaction_cost":
                            2.0 * cost_rate,
                        "net_return":
                            net_return,
                        "equity":
                            equity,
                    }
                )

                in_position = False

                entry_idx = None
                entry_price = None
                entry_atr = None
                entry_score = None
                entry_probability = None

                next_allowed_entry = (
                    i + cooldown_bars
                )

        # ---------------------------------------------------------
        # Equity record
        # ---------------------------------------------------------

        equity_records.append(
            {
                "timestamp":
                    timestamp,
                "index":
                    i,
                "equity":
                    equity,
            }
        )

        # ---------------------------------------------------------
        # Don't enter while already positioned
        # ---------------------------------------------------------

        if in_position:
            continue

        if i < next_allowed_entry:
            continue

        # ---------------------------------------------------------
        # Model probability
        # ---------------------------------------------------------

        if not math.isfinite(
            probability
        ):
            rejection_counts[
                "model_probability"
            ] += 1

            continue

        # ---------------------------------------------------------
        # Calculate score
        # ---------------------------------------------------------

        score, components = (
            _calculate_v4_score(
                row=row,
                probability=probability,
                confidence_threshold=
                    confidence_threshold,
                min_adx=min_adx,
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                max_distance_atr=
                    max_distance_atr,
                atr_percentile_min=
                    atr_percentile_min,
                atr_percentile_max=
                    atr_percentile_max,
            )
        )

        # Individual diagnostic counters
        if components["MODEL"] == 0:
            rejection_counts[
                "model_probability"
            ] += 1

        if components["ADX"] == 0:
            rejection_counts[
                "ADX"
            ] += 1

        if components["RSI"] == 0:
            rejection_counts[
                "RSI"
            ] += 1

        if components["ATR_REGIME"] == 0:
            rejection_counts[
                "ATR_REGIME"
            ] += 1

        if components["1H_TREND"] == 0:
            rejection_counts[
                "1H_TREND"
            ] += 1

        if components["4H_TREND"] == 0:
            rejection_counts[
                "4H_TREND"
            ] += 1

        if components["1D_TREND"] == 0:
            rejection_counts[
                "1D_TREND"
            ] += 1

        if score < min_score:
            rejection_counts[
                "score"
            ] += 1

            continue

        # ---------------------------------------------------------
        # ATR validity
        # ---------------------------------------------------------

        if (
            not math.isfinite(atr)
            or atr <= 0
            or not math.isfinite(close)
        ):
            rejection_counts[
                "ATR_REGIME"
            ] += 1

            continue

        # ---------------------------------------------------------
        # Entry
        # ---------------------------------------------------------

        entry_idx = i

        entry_price = close

        entry_atr = atr

        entry_score = score

        entry_probability = probability

        in_position = True

        accepted_entries.append(
            {
                "index": i,
                "timestamp": timestamp,
                "price": close,
                "probability": probability,
                "score": score,
                "components": components,
            }
        )

    # -------------------------------------------------------------
    # Force-close position at final-test end
    # -------------------------------------------------------------

    if in_position:

        i = end_idx

        timestamp = raw.iloc[
            i
        ]["timestamp"]

        exit_price = float(
            raw.iloc[i]["close"]
        )

        gross_return = (
            exit_price
            / entry_price
            - 1.0
        )

        net_return = (
            gross_return
            - 2.0 * cost_rate
        )

        equity *= (
            1.0 + net_return
        )

        trades.append(
            {
                "entry_index":
                    entry_idx,
                "entry_timestamp":
                    raw.iloc[
                        entry_idx
                    ]["timestamp"],
                "entry_price":
                    entry_price,
                "exit_index":
                    i,
                "exit_timestamp":
                    timestamp,
                "exit_price":
                    exit_price,
                "bars_held":
                    i - entry_idx,
                "exit_reason":
                    "timeout",
                "entry_probability":
                    entry_probability,
                "entry_score":
                    entry_score,
                "atr":
                    entry_atr,
                "sl_price":
                    entry_price
                    - sl_mult * entry_atr,
                "tp_price":
                    entry_price
                    + sl_mult
                    * risk_reward
                    * entry_atr,
                "gross_return":
                    gross_return,
                "transaction_cost":
                    2.0 * cost_rate,
                "net_return":
                    net_return,
                "equity":
                    equity,
            }
        )

    # -------------------------------------------------------------
    # Equity dataframe
    # -------------------------------------------------------------

    equity_df = pd.DataFrame(
        equity_records
    )

    if not equity_df.empty:
        equity_df["equity"] = (
            equity_df["equity"]
            .astype(float)
        )

        equity_df["peak"] = (
            equity_df["equity"]
            .cummax()
        )

        equity_df["drawdown"] = (
            equity_df["equity"]
            / equity_df["peak"]
            - 1.0
        )

    trades_df = pd.DataFrame(
        trades
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    number_of_trades = len(
        trades_df
    )

    if number_of_trades:

        tp_count = int(
            (
                trades_df[
                    "exit_reason"
                ] == "tp"
            ).sum()
        )

        sl_count = int(
            (
                trades_df[
                    "exit_reason"
                ] == "sl"
            ).sum()
        )

        timeout_count = int(
            (
                trades_df[
                    "exit_reason"
                ] == "timeout"
            ).sum()
        )

        wins = (
            trades_df[
                "net_return"
            ] > 0
        )

        win_rate = (
            float(wins.mean())
        )

        gross_profit = float(
            trades_df.loc[
                trades_df["net_return"] > 0,
                "net_return",
            ].sum()
        )

        gross_loss = float(
            -trades_df.loc[
                trades_df["net_return"] < 0,
                "net_return",
            ].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = (
                float("inf")
                if gross_profit > 0
                else 0.0
            )

        avg_trade_return = float(
            trades_df[
                "net_return"
            ].mean()
        )

        total_transaction_cost = (
            float(
                trades_df[
                    "transaction_cost"
                ].sum()
            )
        )

    else:

        tp_count = 0
        sl_count = 0
        timeout_count = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0
        total_transaction_cost = 0.0

    # -------------------------------------------------------------
    # Sharpe
    # -------------------------------------------------------------

    if (
        number_of_trades >= 2
        and trades_df["net_return"].std(
            ddof=1
        ) > 0
    ):

        sharpe = (
            trades_df[
                "net_return"
            ].mean()
            / trades_df[
                "net_return"
            ].std(
                ddof=1
            )
            * np.sqrt(
                number_of_trades
            )
        )

    else:
        sharpe = 0.0

    # -------------------------------------------------------------
    # Max drawdown
    # -------------------------------------------------------------

    if not equity_df.empty:
        max_drawdown = float(
            equity_df[
                "drawdown"
            ].min()
        )
    else:
        max_drawdown = 0.0

    total_return = (
        equity - 1.0
    )

    # -------------------------------------------------------------
    # Signal statistics
    # -------------------------------------------------------------

    accepted_count = len(
        accepted_entries
    )

    print()
    print("=" * 78)
    print(
        "SIGNAL PIPELINE"
    )
    print("=" * 78)

    print(
        f"Model-valid signals:  "
        f"{int(valid_model.iloc[start_idx:end_idx+1].sum())}"
    )

    print(
        f"V4 accepted entries:  "
        f"{accepted_count}"
    )

    print(
        f"Trades executed:      "
        f"{number_of_trades}"
    )

    print()
    print("=" * 78)
    print(
        "RESULT"
    )
    print("=" * 78)

    print(
        f"Strategy total return: "
        f"{total_return * 100:+.2f}%"
    )

    print(
        f"Number of trades:       "
        f"{number_of_trades}"
    )

    print(
        f"TP-first exits:         "
        f"{tp_count}"
    )

    print(
        f"SL-first exits:         "
        f"{sl_count}"
    )

    print(
        f"Timeout exits:          "
        f"{timeout_count}"
    )

    print(
        f"Win rate:               "
        f"{win_rate * 100:+.2f}%"
    )

    if math.isfinite(
        profit_factor
    ):
        print(
            f"Profit factor:          "
            f"{profit_factor:.2f}"
        )
    else:
        print(
            "Profit factor:          INF"
        )

    print(
        f"Sharpe ratio:           "
        f"{sharpe:+.2f}"
    )

    print(
        f"Max drawdown:           "
        f"{max_drawdown * 100:+.2f}%"
    )

    print(
        f"Avg trade return:       "
        f"{avg_trade_return * 100:+.2f}%"
    )

    print(
        f"Total transaction cost: "
        f"{total_transaction_cost * 100:+.2f}%"
    )

    # -------------------------------------------------------------
    # Filter diagnostics
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "V4 SCORE / FILTER DIAGNOSTICS"
    )
    print("=" * 78)

    for key, value in sorted(
        rejection_counts.items(),
        key=lambda x: -x[1],
    ):
        print(
            f"{key:<24} {value}"
        )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    output_dir = (
        artifact_dir
        / "backtest_v4"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_path = (
        output_dir
        / "trades.csv"
    )

    equity_path = (
        output_dir
        / "equity.csv"
    )

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "entry_index",
                "entry_timestamp",
                "entry_price",
                "exit_index",
                "exit_timestamp",
                "exit_price",
                "bars_held",
                "exit_reason",
                "entry_probability",
                "entry_score",
                "atr",
                "sl_price",
                "tp_price",
                "gross_return",
                "transaction_cost",
                "net_return",
                "equity",
            ]
        )

    trades_df.to_csv(
        trades_path,
        index=False,
    )

    equity_df.to_csv(
        equity_path,
        index=False,
    )

    metrics = {
        "version": "V4",
        "symbol": symbol,
        "timeframe": timeframe,
        "higher_timeframes":
            higher_timeframes,
        "side": side,
        "final_test_start":
            str(start_ts),
        "final_test_end":
            str(end_ts),
        "final_test_start_index":
            int(start_idx),
        "final_test_end_index":
            int(end_idx),
        "final_test_rows":
            int(end_idx - start_idx + 1),
        "sl_multiplier":
            sl_mult,
        "risk_reward":
            risk_reward,
        "tp_atr_multiplier":
            sl_mult * risk_reward,
        "max_holding":
            max_holding,
        "transaction_cost_bps":
            transaction_cost_bps,
        "confidence_threshold":
            confidence_threshold,
        "ensemble_mode":
            ensemble_mode,
        "xgb_weight":
            xgb_weight,
        "catboost_weight":
            catboost_weight,
        "min_score":
            min_score,
        "min_adx":
            min_adx,
        "rsi_min":
            rsi_min,
        "rsi_max":
            rsi_max,
        "max_distance_atr":
            max_distance_atr,
        "atr_percentile_min":
            atr_percentile_min,
        "atr_percentile_max":
            atr_percentile_max,
        "cooldown_bars":
            cooldown_bars,
        "total_return":
            total_return,
        "number_of_trades":
            number_of_trades,
        "tp_first":
            tp_count,
        "sl_first":
            sl_count,
        "timeout":
            timeout_count,
        "win_rate":
            win_rate,
        "profit_factor":
            profit_factor,
        "sharpe":
            sharpe,
        "max_drawdown":
            max_drawdown,
        "avg_trade_return":
            avg_trade_return,
        "total_transaction_cost":
            total_transaction_cost,
        "accepted_entries":
            accepted_count,
        "rejection_counts":
            rejection_counts,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            default=str,
        )

    print()
    print(
        f"Saved V4 trades:  "
        f"{trades_path}"
    )

    print(
        f"Saved V4 equity:  "
        f"{equity_path}"
    )

    print(
        f"Saved V4 metrics: "
        f"{metrics_path}"
    )

    if number_of_trades == 0:
        print()
        print(
            "V4 STATUS: NO TRADES."
        )
        print(
            "Lower --min-score or --confidence-threshold "
            "during DEVELOPMENT testing."
        )

    elif total_return > 0:
        print()
        print(
            "V4 STATUS: POSITIVE RETURN "
            "on this test window."
        )
        print(
            "Do not tune parameters against "
            "this final-test window."
        )

    else:
        print()
        print(
            "V4 STATUS: NEGATIVE RETURN "
            "on this final-test window."
        )
        print(
            "Do not tune parameters against "
            "this final-test window."
        )

    return metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "V4 weighted multi-timeframe "
            "TP-before-SL backtest."
        )
    )

    parser.add_argument(
        "--symbol",
        default="SOL/USDT",
    )

    parser.add_argument(
        "--timeframe",
        default="1h",
    )

    parser.add_argument(
        "--higher-timeframes",
        nargs="+",
        default=[
            "4h",
            "1d",
        ],
    )

    parser.add_argument(
        "--side",
        default="long",
        choices=[
            "long",
        ],
    )

    parser.add_argument(
        "--sl-mult",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--risk-reward",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--max-holding",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--ensemble-mode",
        choices=[
            "xgb_cat",
            "stacking",
        ],
        default="xgb_cat",
    )

    parser.add_argument(
        "--xgb-weight",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--catboost-weight",
        type=float,
        default=0.30,
    )

    # -------------------------------------------------------------
    # V4 scoring
    # -------------------------------------------------------------

    parser.add_argument(
        "--min-score",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-adx",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--rsi-min",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--rsi-max",
        type=float,
        default=72.0,
    )

    parser.add_argument(
        "--max-distance-atr",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--atr-percentile-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--atr-percentile-max",
        type=float,
        default=95.0,
    )

    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=4,
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        higher_timeframes=
            args.higher_timeframes,
        side=args.side,
        sl_mult=args.sl_mult,
        risk_reward=args.risk_reward,
        max_holding=args.max_holding,
        holdout_frac=args.holdout_frac,
        confidence_threshold=
            args.confidence_threshold,
        transaction_cost_bps=
            args.transaction_cost_bps,
        ensemble_mode=args.ensemble_mode,
        xgb_weight=args.xgb_weight,
        catboost_weight=
            args.catboost_weight,
        min_score=args.min_score,
        min_adx=args.min_adx,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        max_distance_atr=
            args.max_distance_atr,
        atr_percentile_min=
            args.atr_percentile_min,
        atr_percentile_max=
            args.atr_percentile_max,
        cooldown_bars=
            args.cooldown_bars,
    )


if __name__ == "__main__":
    main()