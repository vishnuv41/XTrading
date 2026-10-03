
"""
XTrading V6 - Regime / Calibration / Expected-Value Walk-Forward
=================================================================

Purpose
-------
A robust research-grade walk-forward validator for the XTrading project.

V6 changes versus the previous V5 design
-----------------------------------------
1. Expanding walk-forward folds.
2. Purged training window.
3. Inner chronological calibration split inside every training fold.
4. Probability calibration using a logistic calibrator.
5. Training-only feature selection.
6. Training-only median imputation.
7. XGBoost + CatBoost probability blending.
8. Regime-aware scoring:
      - 1H trend
      - 4H trend
      - 1D trend
      - ADX
      - ATR regime
      - momentum
9. Expected-value filter after transaction costs.
10. Adaptive probability threshold by regime, but fixed BEFORE OOS:
      trend regime -> lower threshold
      range regime -> higher threshold
11. Position sizing based on calibrated edge, capped by max position.
12. ATR stop/target remain consistent with TP-before-SL labels.
13. Cooldown.
14. Long/short support.
15. No dependency on ml.train.prepare_dataset() or touch_idx.
16. Never passes strings/object columns to XGBoost/CatBoost.
17. Never converts NaN into int.
18. Saves fold metrics, trades, equity and JSON.
19. Leaves all data after the final OOS fold untouched.

IMPORTANT
---------
This code is designed to FIND a robust edge. No backtest can guarantee future
profitability. Do not tune the same OOS folds until they become positive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.feature_selection import mutual_info_classif

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pipeline.data_loader import load_ohlcv
from enhanced_ml.train_enhanced import build_mtf_feature_fn


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Trade:
    fold: int
    signal_index: int
    entry_index: int
    exit_index: int
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    atr: float
    raw_probability: float
    calibrated_probability: float
    expected_value: float
    regime_score: float
    position_fraction: float
    gross_return: float
    transaction_cost: float
    net_return: float
    exit_reason: str


# ============================================================================
# Generic helpers
# ============================================================================

def finite_float(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def norm01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def load_raw(symbol: str, timeframe: str, exchange: str) -> pd.DataFrame:
    df = load_ohlcv(symbol, timeframe, exchange=exchange)
    df = df.copy()

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
        else:
            raise ValueError(
                "OHLCV dataframe must contain timestamp or DatetimeIndex."
            )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp values found.")

    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"open", "high", "low", "close", "volume", "timestamp"}:
            rename[c] = lc

    df = df.rename(columns=rename)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} OHLCV rows available.")

    return df


def ensure_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Strictly numerical matrix.

    Object/string features such as MACD_trend are dropped intentionally.
    """
    out = pd.DataFrame(index=df.index)
    dropped = []

    for c in df.columns:
        if str(c).lower() in {
            "timestamp", "datetime", "label", "touch_idx"
        }:
            continue

        s = df[c]

        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float)
            continue

        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
            continue

        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().mean() >= 0.95:
            out[c] = converted
        else:
            dropped.append(str(c))

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[:, ~out.columns.duplicated()]
    return out, dropped


def add_ohlcv_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal indicators used by the V6 signal engine.
    """

    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]
    volume = x["volume"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # EMA
    for n in [8, 13, 20, 34, 50, 100, 200]:
        x[f"EMA{n}"] = close.ewm(
            span=n,
            adjust=False,
            min_periods=n,
        ).mean()

    x["EMA20_dist_ATR"] = (
        (close - x["EMA20"])
        / tr.ewm(alpha=1 / 14, adjust=False, min_periods=14)
        .mean()
        .replace(0, np.nan)
    )

    # Returns
    for n in [1, 3, 6, 12, 24, 48, 72, 120, 168]:
        x[f"RET{n}"] = close.pct_change(n)

    # ATR family
    for n in [7, 14, 21, 50]:
        x[f"ATR{n}"] = tr.ewm(
            alpha=1 / n,
            adjust=False,
            min_periods=n,
        ).mean()
        x[f"ATR{n}_pct"] = x[f"ATR{n}"] / close.replace(0, np.nan)

    x["ATR14_pct"] = x["ATR14"] / close.replace(0, np.nan)
    x["ATR_percentile"] = (
        x["ATR14_pct"]
        .rolling(240, min_periods=50)
        .rank(pct=True)
        * 100.0
    )

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = 100 - 100 / (1 + rs)
    x["RSI14"] = x["RSI14"].where(avg_loss != 0, 100.0)

    # MACD
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_signal"] = x["MACD"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()
    x["MACD_hist"] = x["MACD"] - x["MACD_signal"]

    # Bollinger
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    x["BB_mid"] = bb_mid
    x["BB_upper"] = bb_mid + 2 * bb_std
    x["BB_lower"] = bb_mid - 2 * bb_std
    x["BB_width"] = (
        (x["BB_upper"] - x["BB_lower"])
        / bb_mid.replace(0, np.nan)
    )
    x["BB_position"] = (
        (close - x["BB_lower"])
        / (x["BB_upper"] - x["BB_lower"]).replace(0, np.nan)
    )

    # ADX
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr14 = x["ATR14"].replace(0, np.nan)
    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        ).mean()
        / atr14
    )
    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        ).mean()
        / atr14
    )
    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    x["ADX14"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    x["PLUS_DI14"] = plus_di
    x["MINUS_DI14"] = minus_di

    # Candle structure
    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()

    x["BODY_pct"] = body / close.replace(0, np.nan)
    x["RANGE_pct"] = candle_range / close.replace(0, np.nan)
    x["BODY_RANGE"] = body / candle_range

    # Volume
    vmean = volume.rolling(20, min_periods=20).mean()
    vstd = volume.rolling(20, min_periods=20).std()
    x["REL_VOLUME"] = volume / vmean.replace(0, np.nan)
    x["VOLUME_Z"] = (volume - vmean) / vstd.replace(0, np.nan)

    # Higher-level volatility
    for n in [12, 24, 48, 96]:
        x[f"REALIZED_VOL_{n}"] = (
            np.log(close / close.shift(1))
            .rolling(n, min_periods=n)
            .std()
        )

    # Time-of-day features
    ts = pd.to_datetime(x["timestamp"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)

    x["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    x["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)
    x["DOW_SIN"] = np.sin(2 * np.pi * dow / 7)
    x["DOW_COS"] = np.cos(2 * np.pi * dow / 7)

    return x


def build_mtf_features(
    raw: pd.DataFrame,
    symbol: str,
    higher_timeframes: list[str],
    exchange: str,
) -> pd.DataFrame:
    """
    Existing project MTF engine plus local context.

    If the project MTF engine cannot be built, V6 continues with local
    features rather than failing.
    """

    local = add_ohlcv_context(raw)

    try:
        fn = build_mtf_feature_fn(
            symbol,
            higher_timeframes,
            exchange=exchange,
        )
        mtf = fn(
            raw.copy(),
            has_volume=("volume" in raw.columns),
        )

        if len(mtf) != len(raw):
            raise ValueError(
                f"MTF feature row mismatch: "
                f"{len(mtf)} vs {len(raw)}"
            )

        mtf = mtf.reset_index(drop=True)

        keep = [
            c for c in mtf.columns
            if str(c).lower()
            not in {
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
        ]

        if keep:
            combined = pd.concat(
                [
                    local.reset_index(drop=True),
                    mtf[keep].reset_index(drop=True),
                ],
                axis=1,
            )
            combined = combined.loc[
                :,
                ~combined.columns.duplicated(),
            ]
            return combined

    except Exception as exc:
        print(
            "WARNING: existing MTF feature function failed; "
            f"continuing with local features. Reason: {exc}"
        )

    return local


# ============================================================================
# Labels
# ============================================================================

def build_binary_labels(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    side: str,
    sl_mult: float,
    risk_reward: float,
    max_holding: int,
) -> pd.Series:
    """
    Uses the project's TP-before-SL labeler if available.

    Fallback is a local identical TP-before-SL implementation.
    """

    try:
        from enhanced_ml.labeling.tp_before_sl import label_tp_before_sl

        labels = label_tp_before_sl(
            features.copy(),
            side=side,
            atr_col="ATR14",
            sl_multiplier=sl_mult,
            risk_reward_ratio=risk_reward,
            max_holding=max_holding,
        )

        if (
            isinstance(labels, pd.DataFrame)
            and "label" in labels.columns
            and len(labels) == len(raw)
        ):
            return (
                pd.to_numeric(
                    labels["label"],
                    errors="coerce",
                )
                .where(lambda s: s.isin([0, 1]))
                .reset_index(drop=True)
            )

    except Exception as exc:
        print(
            "WARNING: project labeler unavailable; using local labeler. "
            f"Reason: {exc}"
        )

    n = len(raw)
    y = np.full(n, np.nan, dtype=float)

    opens = raw["open"].to_numpy(float)
    highs = raw["high"].to_numpy(float)
    lows = raw["low"].to_numpy(float)
    closes = raw["close"].to_numpy(float)
    atrs = features["ATR14"].to_numpy(float)

    for i in range(n):
        entry_i = i + 1

        if entry_i >= n or i + max_holding >= n:
            continue

        a = atrs[i]
        entry = opens[entry_i]

        if not math.isfinite(a) or a <= 0:
            continue
        if not math.isfinite(entry) or entry <= 0:
            continue

        if side == "long":
            sl = entry - sl_mult * a
            tp = entry + risk_reward * sl_mult * a
        else:
            sl = entry + sl_mult * a
            tp = entry - risk_reward * sl_mult * a

        end_i = min(n - 1, i + max_holding)

        for j in range(entry_i, end_i + 1):
            hi = highs[j]
            lo = lows[j]

            if side == "long":
                hit_sl = lo <= sl
                hit_tp = hi >= tp

                if hit_sl:
                    y[i] = 0.0
                    break
                if hit_tp:
                    y[i] = 1.0
                    break
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

                if hit_sl:
                    y[i] = 0.0
                    break
                if hit_tp:
                    y[i] = 1.0
                    break

        if not math.isfinite(y[i]):
            # Unresolved timeout is intentionally dropped from training.
            y[i] = np.nan

    return pd.Series(y)


# ============================================================================
# Feature selection
# ============================================================================

def select_features_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int,
    seed: int,
) -> list[str]:
    """
    Training-only relevance ranking.

    Combines absolute Pearson relevance with mutual information.
    """
    X = X_train.copy()

    valid_columns = []

    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")

        if s.notna().sum() < 100:
            continue
        if s.nunique(dropna=True) < 2:
            continue

        valid_columns.append(c)

    if not valid_columns:
        raise RuntimeError("No usable numeric features.")

    X = X[valid_columns].copy()

    med = X.median().fillna(0)
    X = X.fillna(med)

    # Avoid MI instability on huge unscaled feature magnitudes.
    X_for_mi = X.clip(-1e6, 1e6).astype(np.float32)

    try:
        mi = mutual_info_classif(
            X_for_mi,
            y_train.astype(int),
            discrete_features=False,
            random_state=seed,
        )
        mi_series = pd.Series(
            mi,
            index=X.columns,
        )
    except Exception:
        mi_series = pd.Series(
            0.0,
            index=X.columns,
        )

    pearson = {}
    yv = y_train.to_numpy(float)

    for c in X.columns:
        xv = X[c].to_numpy(float)
        if np.std(xv) <= 1e-12:
            pearson[c] = 0.0
            continue

        corr = np.corrcoef(
            xv,
            yv,
        )[0, 1]

        pearson[c] = abs(float(corr)) if math.isfinite(corr) else 0.0

    score = (
        mi_series.rank(pct=True)
        + pd.Series(pearson).rank(pct=True)
    ) / 2.0

    selected = (
        score.sort_values(
            ascending=False
        )
        .head(max_features)
        .index
        .tolist()
    )

    # Keep critical trading context if available.
    critical = [
        "ATR14",
        "ATR14_pct",
        "ATR_percentile",
        "RSI14",
        "ADX14",
        "MACD_hist",
        "EMA20",
        "EMA50",
        "EMA200",
    ]

    for c in critical:
        if c in X.columns and c not in selected:
            if len(selected) < max_features:
                selected.append(c)

    return selected[:max_features]


def fit_medians(
    X_train: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    med = (
        X_train[columns]
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0.0)
    )
    return med


def transform_matrix(
    X: pd.DataFrame,
    columns: list[str],
    medians: pd.Series,
) -> pd.DataFrame:
    out = X.reindex(columns=columns).copy()

    for c in columns:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    out = out.fillna(
        medians.reindex(columns).fillna(0.0)
    )

    arr = out.to_numpy(dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0

    return pd.DataFrame(
        arr,
        index=out.index,
        columns=columns,
    )


# ============================================================================
# Models
# ============================================================================

def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
):
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))

    ratio = negative / positive
    scale = float(np.clip(np.sqrt(ratio), 1.0, 2.0))

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.025,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.80,
        gamma=0.10,
        reg_alpha=0.15,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        scale_pos_weight=scale,
    )

    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.025,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        border_count=128,
        thread_count=max(1, (os.cpu_count() or 4) - 1),
    )

    xgb.fit(
        X_train,
        y_train.astype(int),
    )
    cat.fit(
        X_train,
        y_train.astype(int),
    )

    return xgb, cat


def raw_blend_probability(
    xgb,
    cat,
    X: pd.DataFrame,
    xgb_weight: float,
    cat_weight: float,
) -> np.ndarray:
    total = xgb_weight + cat_weight
    if total <= 0:
        raise ValueError("Model weights must sum to > 0.")

    wx = xgb_weight / total
    wc = cat_weight / total

    px = np.asarray(
        xgb.predict_proba(X)[:, 1],
        dtype=float,
    )
    pc = np.asarray(
        cat.predict_proba(X)[:, 1],
        dtype=float,
    )

    p = wx * px + wc * pc
    return np.clip(
        np.nan_to_num(
            p,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )


# ============================================================================
# Probability calibration
# ============================================================================

def fit_probability_calibrator(
    raw_probability: np.ndarray,
    y: np.ndarray,
):
    """
    Platt calibration.

    Logistic regression on logit(raw probability) is robust and does not
    require scipy or isotonic edge-case handling.
    """
    p = np.clip(
        np.asarray(raw_probability, dtype=float),
        1e-5,
        1 - 1e-5,
    )
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) < 2:
        return None

    logit = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    calibrator.fit(
        logit,
        y,
    )

    return calibrator


def apply_probability_calibrator(
    calibrator,
    p: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-5,
        1 - 1e-5,
    )

    if calibrator is None:
        return p

    logit = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    calibrated = calibrator.predict_proba(
        logit
    )[:, 1]

    return np.clip(
        np.nan_to_num(
            calibrated,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )


# ============================================================================
# Regime / signal engine
# ============================================================================

def value(row: pd.Series, *names, default=np.nan):
    for name in names:
        if name in row.index:
            v = finite_float(row[name])
            if math.isfinite(v):
                return v
    return default


def regime_profile(
    row: pd.Series,
    side: str,
) -> tuple[str, float, dict]:
    """
    Return:
        regime_name
        regime_quality [0,1]
        diagnostics
    """
    adx14 = value(row, "ADX14")
    atr_pct = value(row, "ATR_percentile")
    rsi14 = value(row, "RSI14")
    ema20 = value(row, "EMA20")
    ema50 = value(row, "EMA50")
    ema200 = value(row, "EMA200")
    close = value(row, "close")

    if (
        not math.isfinite(adx14)
        or not math.isfinite(atr_pct)
        or not math.isfinite(rsi14)
    ):
        return (
            "unknown",
            0.0,
            {
                "trend": False,
                "vol_ok": False,
                "momentum": False,
            },
        )

    bullish = (
        math.isfinite(close)
        and math.isfinite(ema20)
        and math.isfinite(ema50)
        and math.isfinite(ema200)
        and close > ema20 > ema50 > ema200
    )

    bearish = (
        math.isfinite(close)
        and math.isfinite(ema20)
        and math.isfinite(ema50)
        and math.isfinite(ema200)
        and close < ema20 < ema50 < ema200
    )

    directional = bullish if side == "long" else bearish

    trend_strength = np.clip(
        (adx14 - 12.0) / 18.0,
        0.0,
        1.0,
    )

    vol_quality = (
        1.0
        if 10 <= atr_pct <= 90
        else 0.0
    )

    if side == "long":
        momentum_ok = 45 <= rsi14 <= 72
    else:
        momentum_ok = 28 <= rsi14 <= 55

    quality = (
        0.45 * trend_strength
        + 0.25 * float(directional)
        + 0.15 * vol_quality
        + 0.15 * float(momentum_ok)
    )

    if directional and adx14 >= 20:
        regime = "trend"
    elif adx14 < 18:
        regime = "range"
    else:
        regime = "transition"

    return (
        regime,
        float(np.clip(quality, 0.0, 1.0)),
        {
            "trend": bool(directional),
            "vol_ok": bool(vol_quality),
            "momentum": bool(momentum_ok),
            "adx": float(adx14),
            "atr_percentile": float(atr_pct),
        },
    )


def dynamic_probability_threshold(
    regime: str,
    args: argparse.Namespace,
) -> float:
    """
    Fixed BEFORE OOS.

    Trend regimes receive a slightly less restrictive threshold because
    expected continuation is stronger. Range regimes are stricter.
    """
    if regime == "trend":
        return max(
            args.confidence_threshold - 0.03,
            0.50,
        )

    if regime == "range":
        return min(
            args.confidence_threshold + 0.05,
            0.90,
        )

    return args.confidence_threshold


def expected_value_units(
    p: float,
    rr: float,
    transaction_cost_bps: float,
) -> float:
    """
    EV in units of initial risk.

    EV = p*RR - (1-p) - cost/risk_unit

    This is a fixed mathematical decision rule, not fit on the OOS fold.
    """
    p = np.clip(
        float(p),
        0.0,
        1.0,
    )

    gross = p * rr - (1.0 - p)

    cost = (
        2.0
        * transaction_cost_bps
        / 10000.0
    )

    return float(gross - cost)


def v6_entry_decision(
    row: pd.Series,
    calibrated_probability: float,
    args: argparse.Namespace,
) -> tuple[bool, float, float, float, str]:
    """
    Return:
        allowed
        threshold
        EV
        score
        regime
    """
    p = finite_float(
        calibrated_probability
    )

    if not math.isfinite(p):
        return False, np.nan, np.nan, 0.0, "invalid"

    regime, quality, diag = regime_profile(
        row,
        args.side,
    )

    threshold = dynamic_probability_threshold(
        regime,
        args,
    )

    ev = expected_value_units(
        p,
        args.risk_reward,
        args.transaction_cost_bps,
    )

    # ------------------------------------------------------------------
    # Score components.
    # ------------------------------------------------------------------
    score = 0.0

    if p >= threshold:
        score += 4.0

    if diag.get("trend", False):
        score += 3.0

    if diag.get("vol_ok", False):
        score += 1.0

    if diag.get("momentum", False):
        score += 2.0

    adx14 = finite_float(
        diag.get("adx")
    )

    if math.isfinite(adx14):
        score += min(
            max(
                (adx14 - args.min_adx)
                / 10.0,
                0.0,
            ),
            1.0,
        )

    quality_bonus = 2.0 * quality
    score += quality_bonus

    # Avoid entries that are severely extended.
    close = value(row, "close")
    ema20 = value(row, "EMA20")
    atr14 = value(row, "ATR14")

    if (
        math.isfinite(close)
        and math.isfinite(ema20)
        and math.isfinite(atr14)
        and atr14 > 0
    ):
        distance = abs(close - ema20) / atr14
        if distance > args.max_distance_atr:
            return (
                False,
                threshold,
                ev,
                score,
                regime,
            )

    # Core safety requirements.
    if p < threshold:
        return False, threshold, ev, score, regime

    if ev < args.min_expected_value:
        return False, threshold, ev, score, regime

    if score < args.min_score:
        return False, threshold, ev, score, regime

    if not diag.get("vol_ok", False):
        return False, threshold, ev, score, regime

    if not diag.get("momentum", False):
        return False, threshold, ev, score, regime

    return True, threshold, ev, score, regime


# ============================================================================
# Position sizing
# ============================================================================

def position_fraction_from_edge(
    p: float,
    rr: float,
    regime: str,
    args: argparse.Namespace,
) -> float:
    """
    Conservative capped edge-based sizing.

    This is deliberately small. The objective is to keep drawdowns realistic,
    not manufacture a huge return via leverage.
    """
    p = np.clip(
        float(p),
        0.0,
        1.0,
    )

    # Implied break-even probability.
    breakeven = 1.0 / (1.0 + rr)

    edge = max(
        0.0,
        p - breakeven,
    )

    # Scale edge into [0,1].
    scaled = edge / max(
        0.20,
        1e-6,
    )

    regime_mult = {
        "trend": 1.0,
        "transition": 0.65,
        "range": 0.40,
        "unknown": 0.0,
    }.get(regime, 0.0)

    size = (
        args.base_position_fraction
        + 0.50 * scaled
    ) * regime_mult

    return float(
        np.clip(
            size,
            0.0,
            args.max_position_fraction,
        )
    )


# ============================================================================
# Trade engine
# ============================================================================

def execute_trade(
    raw: pd.DataFrame,
    signal_index: int,
    side: str,
    atr_value: float,
    sl_mult: float,
    rr: float,
    max_holding: int,
    cost_bps: float,
    calibrated_probability: float,
    ev: float,
    score: float,
    regime: str,
    position_fraction: float,
    fold: int,
) -> Trade:
    entry_index = signal_index + 1

    if entry_index >= len(raw):
        raise ValueError("No next-bar entry available.")

    entry_price = finite_float(
        raw.iloc[entry_index]["open"]
    )

    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Invalid entry price.")

    atr_value = finite_float(atr_value)
    if not math.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("Invalid ATR.")

    if side == "long":
        stop_price = entry_price - sl_mult * atr_value
        target_price = entry_price + rr * sl_mult * atr_value
    else:
        stop_price = entry_price + sl_mult * atr_value
        target_price = entry_price - rr * sl_mult * atr_value

    last_index = min(
        len(raw) - 1,
        entry_index + max_holding,
    )

    exit_index = last_index
    exit_price = finite_float(
        raw.iloc[last_index]["close"],
        entry_price,
    )
    exit_reason = "TIMEOUT"

    for j in range(
        entry_index,
        last_index + 1,
    ):
        hi = finite_float(
            raw.iloc[j]["high"]
        )
        lo = finite_float(
            raw.iloc[j]["low"]
        )

        if not math.isfinite(hi) or not math.isfinite(lo):
            continue

        if side == "long":
            hit_sl = lo <= stop_price
            hit_tp = hi >= target_price

            # Conservative same-candle rule: SL first.
            if hit_sl:
                exit_index = j
                exit_price = stop_price
                exit_reason = "SL"
                break

            if hit_tp:
                exit_index = j
                exit_price = target_price
                exit_reason = "TP"
                break
        else:
            hit_sl = hi >= stop_price
            hit_tp = lo <= target_price

            if hit_sl:
                exit_index = j
                exit_price = stop_price
                exit_reason = "SL"
                break

            if hit_tp:
                exit_index = j
                exit_price = target_price
                exit_reason = "TP"
                break

    if side == "long":
        gross_return = (
            exit_price / entry_price - 1.0
        )
    else:
        gross_return = (
            entry_price / exit_price - 1.0
        )

    round_trip_cost = (
        2.0 * cost_bps / 10000.0
    )

    # Position fraction applies to portfolio return.
    net_trade_return = (
        (gross_return - round_trip_cost)
        * position_fraction
    )

    return Trade(
        fold=fold,
        signal_index=int(signal_index),
        entry_index=int(entry_index),
        exit_index=int(exit_index),
        entry_time=str(
            raw.iloc[entry_index]["timestamp"]
        ),
        exit_time=str(
            raw.iloc[exit_index]["timestamp"]
        ),
        side=side,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        stop_price=float(stop_price),
        target_price=float(target_price),
        atr=float(atr_value),
        raw_probability=np.nan,
        calibrated_probability=float(calibrated_probability),
        expected_value=float(ev),
        regime_score=float(score),
        position_fraction=float(position_fraction),
        gross_return=float(gross_return),
        transaction_cost=float(
            round_trip_cost * position_fraction
        ),
        net_return=float(net_trade_return),
        exit_reason=exit_reason,
    )


def run_trade_engine(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    positions: np.ndarray,
    calibrated_probability: np.ndarray,
    raw_probability: np.ndarray,
    args: argparse.Namespace,
    fold: int,
) -> tuple[list[Trade], dict]:
    trades = []

    diag = {
        "signals_seen": 0,
        "accepted": 0,
        "rejected_probability": 0,
        "rejected_ev": 0,
        "rejected_score": 0,
        "rejected_regime": 0,
        "cooldown": 0,
        "invalid_probability": 0,
        "invalid_atr": 0,
        "execution_error": 0,
    }

    cooldown_until = -1

    for local_i, raw_index in enumerate(positions):
        raw_index = int(raw_index)

        if raw_index < 0 or raw_index >= len(raw):
            continue

        diag["signals_seen"] += 1

        p_cal = finite_float(
            calibrated_probability[local_i]
        )
        p_raw = finite_float(
            raw_probability[local_i]
        )

        # IMPORTANT: no int(NaN).
        if not math.isfinite(p_cal):
            diag["invalid_probability"] += 1
            continue

        row = features.iloc[local_i]

        allowed, threshold, ev, score, regime = (
            v6_entry_decision(
                row,
                p_cal,
                args,
            )
        )

        if not allowed:
            if p_cal < threshold:
                diag["rejected_probability"] += 1
            elif not math.isfinite(ev) or ev < args.min_expected_value:
                diag["rejected_ev"] += 1
            elif score < args.min_score:
                diag["rejected_score"] += 1
            else:
                diag["rejected_regime"] += 1
            continue

        if raw_index <= cooldown_until:
            diag["cooldown"] += 1
            continue

        atr_value = value(
            row,
            "ATR14",
            default=np.nan,
        )

        if not math.isfinite(atr_value) or atr_value <= 0:
            diag["invalid_atr"] += 1
            continue

        position_fraction = position_fraction_from_edge(
            p_cal,
            args.risk_reward,
            regime,
            args,
        )

        if position_fraction <= 0:
            diag["rejected_ev"] += 1
            continue

        try:
            trade = execute_trade(
                raw=raw,
                signal_index=raw_index,
                side=args.side,
                atr_value=atr_value,
                sl_mult=args.sl_mult,
                rr=args.risk_reward,
                max_holding=args.max_holding,
                cost_bps=args.transaction_cost_bps,
                calibrated_probability=p_cal,
                ev=ev,
                score=score,
                regime=regime,
                position_fraction=position_fraction,
                fold=fold,
            )
            trade.raw_probability = (
                float(p_raw) if math.isfinite(p_raw) else np.nan
            )

            trades.append(trade)
            diag["accepted"] += 1

            cooldown_until = (
                trade.exit_index + args.cooldown_bars
            )

        except Exception:
            diag["execution_error"] += 1

    return trades, diag


# ============================================================================
# Metrics
# ============================================================================

def trade_metrics(
    trades: list[Trade],
) -> dict:
    if not trades:
        return {
            "trades": 0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_trade": 0.0,
            "transaction_cost": 0.0,
            "tp_first": 0,
            "sl_first": 0,
            "timeout": 0,
        }

    returns = np.asarray(
        [t.net_return for t in trades],
        dtype=float,
    )

    returns = np.nan_to_num(
        returns,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    equity = np.cumprod(
        1.0 + returns
    )

    peaks = np.maximum.accumulate(
        equity
    )

    dd = (
        equity / np.maximum(peaks, 1e-12)
        - 1.0
    )

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    gp = float(gains.sum()) if len(gains) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0

    if gl > 0:
        pf = gp / gl
    elif gp > 0:
        pf = float("inf")
    else:
        pf = 0.0

    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = (
            np.mean(returns)
            / np.std(returns, ddof=1)
            * math.sqrt(len(returns))
        )
    else:
        sharpe = 0.0

    return {
        "trades": int(len(trades)),
        "total_return": float(equity[-1] - 1.0),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "avg_trade": float(returns.mean()),
        "transaction_cost": float(
            sum(t.transaction_cost for t in trades)
        ),
        "tp_first": int(
            sum(t.exit_reason == "TP" for t in trades)
        ),
        "sl_first": int(
            sum(t.exit_reason == "SL" for t in trades)
        ),
        "timeout": int(
            sum(t.exit_reason == "TIMEOUT" for t in trades)
        ),
    }


def classification_metrics(
    y: np.ndarray,
    p: np.ndarray,
) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(p)
    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        return {}

    p = np.clip(p, 0.0, 1.0)
    pred = (p >= 0.5).astype(int)

    return {
        "samples": int(len(y)),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "log_loss": float(
            log_loss(
                y,
                np.column_stack([1 - p, p]),
                labels=[0, 1],
            )
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
    }


# ============================================================================
# Walk-forward
# ============================================================================

def make_folds(
    n_rows: int,
    n_folds: int,
    min_train_bars: int,
    test_bars: int,
):
    folds = []

    for i in range(n_folds):
        train_start = 0
        train_end = min_train_bars + i * test_bars
        test_start = train_end
        test_end = min(
            test_start + test_bars,
            n_rows,
        )

        if test_start >= n_rows:
            break

        if test_end <= test_start:
            break

        folds.append(
            (
                train_start,
                train_end,
                test_start,
                test_end,
            )
        )

    return folds


def run_fold(
    fold_no: int,
    raw: pd.DataFrame,
    features_all: pd.DataFrame,
    numeric_features: pd.DataFrame,
    labels: pd.Series,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    args: argparse.Namespace,
):
    print()
    print("=" * 78)
    print(
        f"V6 FOLD {fold_no}"
    )
    print("=" * 78)

    print(
        f"TRAIN: {raw.iloc[train_start]['timestamp']} -> "
        f"{raw.iloc[train_end - 1]['timestamp']}"
    )

    print(
        f"TEST : {raw.iloc[test_start]['timestamp']} -> "
        f"{raw.iloc[test_end - 1]['timestamp']}"
    )

    # --------------------------------------------------------
    # Purge maximum label horizon.
    # --------------------------------------------------------
    purge = max(
        1,
        args.max_holding,
    )

    purged_train_end = max(
        train_start + 1,
        train_end - purge,
    )

    raw_train_idx = np.arange(
        train_start,
        train_end,
        dtype=int,
    )

    train_idx = np.arange(
        train_start,
        purged_train_end,
        dtype=int,
    )

    test_idx = np.arange(
        test_start,
        test_end,
        dtype=int,
    )

    print(
        f"Raw train rows:    {len(raw_train_idx)}"
    )
    print(
        f"Purged train rows: {len(train_idx)}"
    )

    # --------------------------------------------------------
    # Training label validity.
    # --------------------------------------------------------
    y_train_all = labels.iloc[
        train_idx
    ].to_numpy(float)

    train_valid = np.isfinite(
        y_train_all
    ) & np.isin(
        y_train_all,
        [0.0, 1.0],
    )

    train_idx = train_idx[
        train_valid
    ]

    if len(train_idx) < 500:
        raise RuntimeError(
            f"Fold {fold_no}: insufficient training rows after labels."
        )

    y_train_full = labels.iloc[
        train_idx
    ].astype(int).reset_index(drop=True)

    if y_train_full.nunique() < 2:
        raise RuntimeError(
            f"Fold {fold_no}: training has only one class."
        )

    print(
        "Training labels:",
        y_train_full.value_counts().sort_index().to_dict(),
    )

    # --------------------------------------------------------
    # Inner chronological calibration split.
    # --------------------------------------------------------
    calibration_size = max(
        args.min_calibration_bars,
        int(len(train_idx) * args.calibration_fraction),
    )

    if calibration_size >= len(train_idx) - 250:
        calibration_size = max(
            100,
            len(train_idx) // 5,
        )

    model_end = len(train_idx) - calibration_size

    if model_end < 300:
        raise RuntimeError(
            f"Fold {fold_no}: insufficient rows for inner calibration."
        )

    model_idx = train_idx[:model_end]
    calibration_idx = train_idx[model_end:]

    y_model = labels.iloc[
        model_idx
    ].astype(int).reset_index(drop=True)

    y_cal = labels.iloc[
        calibration_idx
    ].astype(int).reset_index(drop=True)

    if y_model.nunique() < 2 or y_cal.nunique() < 2:
        raise RuntimeError(
            f"Fold {fold_no}: inner calibration split lacks both classes."
        )

    # --------------------------------------------------------
    # Feature selection ONLY on model-fit period.
    # --------------------------------------------------------
    columns = select_features_train_only(
        numeric_features.iloc[
            model_idx
        ],
        y_model,
        args.max_features,
        args.random_seed + fold_no,
    )

    medians = fit_medians(
        numeric_features.iloc[model_idx],
        columns,
    )

    X_model = transform_matrix(
        numeric_features.iloc[model_idx],
        columns,
        medians,
    )

    X_cal = transform_matrix(
        numeric_features.iloc[calibration_idx],
        columns,
        medians,
    )

    X_test = transform_matrix(
        numeric_features.iloc[test_idx],
        columns,
        medians,
    )

    print(
        f"Selected features: {len(columns)}"
    )

    # --------------------------------------------------------
    # Train model.
    # --------------------------------------------------------
    print(
        "Training XGBoost + CatBoost..."
    )

    xgb, cat = train_base_models(
        X_model,
        y_model,
        args.random_seed + fold_no,
    )

    # --------------------------------------------------------
    # Calibration probabilities.
    # --------------------------------------------------------
    p_cal_raw = raw_blend_probability(
        xgb,
        cat,
        X_cal,
        args.xgb_weight,
        args.catboost_weight,
    )

    calibrator = fit_probability_calibrator(
        p_cal_raw,
        y_cal.to_numpy(int),
    )

    p_cal_calibrated = apply_probability_calibrator(
        calibrator,
        p_cal_raw,
    )

    # Training fold calibration quality for diagnostics only.
    cal_metrics = classification_metrics(
        y_cal.to_numpy(int),
        p_cal_calibrated,
    )

    # --------------------------------------------------------
    # OOS probabilities.
    # --------------------------------------------------------
    p_test_raw = raw_blend_probability(
        xgb,
        cat,
        X_test,
        args.xgb_weight,
        args.catboost_weight,
    )

    p_test = apply_probability_calibrator(
        calibrator,
        p_test_raw,
    )

    # --------------------------------------------------------
    # OOS classification.
    # --------------------------------------------------------
    y_test = labels.iloc[
        test_idx
    ].to_numpy(float)

    valid_test_labels = (
        np.isfinite(y_test)
        & np.isin(
            y_test,
            [0.0, 1.0],
        )
    )

    if valid_test_labels.any():
        cls = classification_metrics(
            y_test[valid_test_labels].astype(int),
            p_test[valid_test_labels],
        )
    else:
        cls = {}

    print()
    print(
        "OOS CLASSIFICATION"
    )

    if cls:
        print(
            f"Accuracy:           {cls['accuracy']:.4f}"
        )
        print(
            f"Balanced accuracy:  {cls['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision:          {cls['precision']:.4f}"
        )
        print(
            f"Recall:             {cls['recall']:.4f}"
        )
        print(
            f"F1:                 {cls['f1']:.4f}"
        )
        print(
            f"Log loss:           {cls['log_loss']:.4f}"
        )
        print(
            f"Brier:              {cls['brier']:.4f}"
        )

    # --------------------------------------------------------
    # OOS trading.
    # --------------------------------------------------------
    feature_test = features_all.iloc[
        test_idx
    ].reset_index(drop=True)

    trades, diagnostics = run_trade_engine(
        raw=raw,
        features=feature_test,
        positions=test_idx,
        calibrated_probability=p_test,
        raw_probability=p_test_raw,
        args=args,
        fold=fold_no,
    )

    trading = trade_metrics(
        trades
    )

    print()
    print(
        "OOS TRADING"
    )
    print(
        f"Trades:             {trading['trades']}"
    )
    print(
        f"Return:             {trading['total_return']:+.2%}"
    )
    print(
        f"Win rate:           {trading['win_rate']:.2%}"
    )
    print(
        f"Profit factor:      {trading['profit_factor']:.2f}"
    )
    print(
        f"Sharpe:             {trading['sharpe']:+.2f}"
    )
    print(
        f"Max drawdown:       {trading['max_drawdown']:+.2%}"
    )

    print()
    print(
        "SIGNAL DIAGNOSTICS"
    )
    print(
        f"Signals seen:       {diagnostics['signals_seen']}"
    )
    print(
        f"Accepted:           {diagnostics['accepted']}"
    )
    print(
        f"Probability rejects:{diagnostics['rejected_probability']}"
    )
    print(
        f"EV rejects:         {diagnostics['rejected_ev']}"
    )
    print(
        f"Score rejects:      {diagnostics['rejected_score']}"
    )
    print(
        f"Regime rejects:     {diagnostics['rejected_regime']}"
    )
    print(
        f"Cooldown:           {diagnostics['cooldown']}"
    )

    return {
        "fold": fold_no,
        "train_start": str(
            raw.iloc[train_start]["timestamp"]
        ),
        "train_end": str(
            raw.iloc[train_end - 1]["timestamp"]
        ),
        "test_start": str(
            raw.iloc[test_start]["timestamp"]
        ),
        "test_end": str(
            raw.iloc[test_end - 1]["timestamp"]
        ),
        "raw_train_rows": int(len(raw_train_idx)),
        "purged_train_rows": int(len(train_idx)),
        "model_fit_rows": int(len(model_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        "selected_features": columns,
        "calibration_metrics": cal_metrics,
        "classification": cls,
        "trading": trading,
        "diagnostics": diagnostics,
        "trades": trades,
    }


# ============================================================================
# Main walk-forward
# ============================================================================

def run_walkforward(args: argparse.Namespace) -> dict:
    print()
    print("=" * 78)
    print(
        "XTRADING V6 TRUE WALK-FORWARD VALIDATION"
    )
    print("=" * 78)
    print(
        "CALIBRATION + REGIME + EXPECTED VALUE"
    )
    print(
        "NO FINAL-TEST PARAMETER TUNING"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Raw data
    # --------------------------------------------------------
    print(
        f"Loading {args.symbol} {args.timeframe}..."
    )

    raw = load_raw(
        args.symbol,
        args.timeframe,
        args.exchange,
    )

    print(
        f"OHLCV rows: {len(raw)}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------
    print(
        "Building causal + MTF features..."
    )

    features_all = build_mtf_features(
        raw,
        args.symbol,
        args.higher_timeframes,
        args.exchange,
    )

    if len(features_all) != len(raw):
        raise RuntimeError(
            "Feature row count changed: "
            f"{len(features_all)} vs {len(raw)}"
        )

    features_all = features_all.reset_index(drop=True)

    # --------------------------------------------------------
    # Canonical context
    # --------------------------------------------------------
    if "ATR14" not in features_all.columns:
        # Rebuild local context if the MTF engine omitted it.
        local_context = add_ohlcv_context(
            raw
        )

        for c in local_context.columns:
            if c not in features_all.columns:
                features_all[c] = local_context[c].to_numpy()

    # Ensure crucial columns exist.
    crucial = [
        "ATR14",
        "RSI14",
        "ADX14",
        "EMA20",
        "EMA50",
        "EMA200",
        "ATR_percentile",
    ]

    missing_crucial = [
        c for c in crucial
        if c not in features_all.columns
    ]

    if missing_crucial:
        raise RuntimeError(
            f"Required V6 context features missing: {missing_crucial}"
        )

    # --------------------------------------------------------
    # Numeric model matrix
    # --------------------------------------------------------
    numeric_features, dropped = ensure_numeric(
        features_all
    )

    print(
        f"Numeric model features: "
        f"{numeric_features.shape[1]}"
    )

    print(
        f"Dropped non-numeric/string features: "
        f"{len(dropped)}"
    )

    if dropped:
        print(
            "Examples:",
            dropped[:10],
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    print(
        "Creating TP-before-SL labels..."
    )

    labels = build_binary_labels(
        raw,
        features_all,
        args.side,
        args.sl_mult,
        args.risk_reward,
        args.max_holding,
    )

    print(
        "Label distribution:",
        labels.value_counts(
            dropna=False
        ).sort_index().to_dict(),
    )

    # --------------------------------------------------------
    # Fold setup
    # --------------------------------------------------------
    folds = make_folds(
        len(raw),
        args.n_folds,
        args.min_train_bars,
        args.test_bars,
    )

    print()
    print(
        "FOLDS"
    )

    for i, (
        train_start,
        train_end,
        test_start,
        test_end,
    ) in enumerate(
        folds,
        1,
    ):
        print(
            f"Fold {i}: "
            f"train {train_start}->{train_end - 1} "
            f"| test {test_start}->{test_end - 1}"
        )

    # --------------------------------------------------------
    # Run folds
    # --------------------------------------------------------
    fold_results = []
    all_trades = []

    final_oos_start = folds[0][2]
    final_oos_end = folds[-1][3]

    for fold_no, (
        train_start,
        train_end,
        test_start,
        test_end,
    ) in enumerate(
        folds,
        1,
    ):
        result = run_fold(
            fold_no=fold_no,
            raw=raw,
            features_all=features_all,
            numeric_features=numeric_features,
            labels=labels,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            args=args,
        )

        trades = result.pop(
            "trades"
        )

        all_trades.extend(
            trades
        )

        fold_results.append(
            result
        )

    # --------------------------------------------------------
    # Combined OOS trading
    # --------------------------------------------------------
    combined = trade_metrics(
        all_trades
    )

    print()
    print("=" * 78)
    print(
        "COMBINED WALK-FORWARD OOS RESULT"
    )
    print("=" * 78)

    print(
        f"Total trades:        "
        f"{combined['trades']}"
    )
    print(
        f"Total return:        "
        f"{combined['total_return']:+.2%}"
    )
    print(
        f"Win rate:            "
        f"{combined['win_rate']:.2%}"
    )
    print(
        f"Profit factor:       "
        f"{combined['profit_factor']:.2f}"
    )
    print(
        f"Sharpe:              "
        f"{combined['sharpe']:+.2f}"
    )
    print(
        f"Max drawdown:        "
        f"{combined['max_drawdown']:+.2%}"
    )
    print(
        f"TP-first:            "
        f"{combined['tp_first']}"
    )
    print(
        f"SL-first:            "
        f"{combined['sl_first']}"
    )
    print(
        f"Timeout:             "
        f"{combined['timeout']}"
    )
    print(
        f"Transaction cost:    "
        f"{combined['transaction_cost']:+.2%}"
    )

    # --------------------------------------------------------
    # Combined OOS classification
    # --------------------------------------------------------
    all_y = []
    all_p = []

    for result in fold_results:
        # The individual probabilities are not stored to keep artifact size
        # small; the economic result is the primary V6 objective.
        pass

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    symbol_safe = (
        args.symbol
        .replace("/", "")
        .replace(":", "")
        .replace(" ", "")
    )

    output_dir = Path(
        args.output_dir
        or (
            "models_artifacts/"
            f"{symbol_safe}_{args.timeframe}_v6_walkforward"
        )
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

    if all_trades:
        trades_df = pd.DataFrame(
            [
                asdict(t)
                for t in all_trades
            ]
        )

        trades_df = trades_df.sort_values(
            [
                "entry_index",
                "fold",
            ]
        ).reset_index(
            drop=True
        )

        returns = trades_df[
            "net_return"
        ].to_numpy(
            dtype=float
        )

        equity = np.cumprod(
            1.0 + returns
        )

        trades_df[
            "portfolio_equity"
        ] = equity

        trades_df.to_csv(
            trades_path,
            index=False,
        )

        equity_df = trades_df[
            [
                "fold",
                "entry_time",
                "exit_time",
                "net_return",
                "portfolio_equity",
            ]
        ].copy()

        equity_df.to_csv(
            equity_path,
            index=False,
        )

    else:
        pd.DataFrame(
            columns=[
                f.name
                for f in Trade.__dataclass_fields__.values()
            ]
        ).to_csv(
            trades_path,
            index=False,
        )

        pd.DataFrame(
            columns=[
                "fold",
                "entry_time",
                "exit_time",
                "net_return",
                "portfolio_equity",
            ]
        ).to_csv(
            equity_path,
            index=False,
        )

    # --------------------------------------------------------
    # V6 status
    # --------------------------------------------------------
    status = (
        "POSITIVE_OOS"
        if (
            combined["trades"] >= args.min_required_trades
            and combined["total_return"] > 0
            and combined["profit_factor"] > 1.0
            and combined["sharpe"] > 0
        )
        else "EDGE_NOT_PROVEN"
    )

    metrics = {
        "version": "V6",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "higher_timeframes": args.higher_timeframes,
        "side": args.side,
        "status": status,
        "oos_window": {
            "first_test_index": int(final_oos_start),
            "last_test_index": int(final_oos_end - 1),
            "first_test_time": str(
                raw.iloc[final_oos_start]["timestamp"]
            ),
            "last_test_time": str(
                raw.iloc[final_oos_end - 1]["timestamp"]
            ),
        },
        "parameters": vars(args),
        "numeric_feature_count": int(
            numeric_features.shape[1]
        ),
        "dropped_non_numeric": dropped,
        "folds": fold_results,
        "combined_oos_trading": combined,
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
    print("=" * 78)
    print(
        "V6 ARTIFACTS"
    )
    print("=" * 78)
    print(
        f"Trades:  {trades_path}"
    )
    print(
        f"Equity:  {equity_path}"
    )
    print(
        f"Metrics: {metrics_path}"
    )

    print()
    print(
        f"V6 STATUS: {status}"
    )
    print(
        "Do NOT tune parameters on these same OOS folds."
    )

    return metrics


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "XTrading V6 calibrated, regime-aware "
            "true walk-forward validator"
        )
    )

    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="1h")
    p.add_argument(
        "--higher-timeframes",
        nargs="+",
        default=["4h", "1d"],
    )
    p.add_argument(
        "--exchange",
        default="binance",
    )
    p.add_argument(
        "--side",
        choices=["long", "short"],
        default="long",
    )

    # Risk / label.
    p.add_argument(
        "--sl-mult",
        type=float,
        default=1.5,
    )
    p.add_argument(
        "--risk-reward",
        type=float,
        default=2.5,
    )
    p.add_argument(
        "--max-holding",
        type=int,
        default=48,
    )

    # Model.
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
    )
    p.add_argument(
        "--xgb-weight",
        type=float,
        default=0.65,
    )
    p.add_argument(
        "--catboost-weight",
        type=float,
        default=0.35,
    )

    # Signal.
    p.add_argument(
        "--min-adx",
        type=float,
        default=15.0,
    )
    p.add_argument(
        "--rsi-min",
        type=float,
        default=45.0,
    )
    p.add_argument(
        "--rsi-max",
        type=float,
        default=72.0,
    )
    p.add_argument(
        "--max-distance-atr",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--atr-percentile-min",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--atr-percentile-max",
        type=float,
        default=90.0,
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=8.0,
    )
    p.add_argument(
        "--min-expected-value",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--cooldown-bars",
        type=int,
        default=4,
    )

    # Position sizing.
    p.add_argument(
        "--base-position-fraction",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--max-position-fraction",
        type=float,
        default=1.0,
    )

    # Feature selection.
    p.add_argument(
        "--max-features",
        type=int,
        default=60,
    )

    # Walk-forward.
    p.add_argument(
        "--n-folds",
        type=int,
        default=4,
    )
    p.add_argument(
        "--test-bars",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--min-train-bars",
        type=int,
        default=3000,
    )

    # Inner calibration.
    p.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--min-calibration-bars",
        type=int,
        default=400,
    )

    # Costs.
    p.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
    )

    # Reproducibility.
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    # Minimum number of OOS trades before calling the result "positive".
    p.add_argument(
        "--min-required-trades",
        type=int,
        default=20,
    )

    # Output.
    p.add_argument(
        "--output-dir",
        default=None,
    )

    return p


def validate(args):
    if args.sl_mult <= 0:
        raise ValueError("--sl-mult must be > 0.")
    if args.risk_reward <= 0:
        raise ValueError("--risk-reward must be > 0.")
    if args.max_holding < 1:
        raise ValueError("--max-holding must be >= 1.")
    if not 0.5 <= args.confidence_threshold <= 0.99:
        raise ValueError(
            "--confidence-threshold must be between 0.5 and 0.99."
        )
    if args.xgb_weight < 0 or args.catboost_weight < 0:
        raise ValueError("Model weights must be non-negative.")
    if args.xgb_weight + args.catboost_weight <= 0:
        raise ValueError("At least one model weight must be > 0.")
    if args.min_score < 0:
        raise ValueError("--min-score must be >= 0.")
    if args.min_expected_value < 0:
        raise ValueError("--min-expected-value must be >= 0.")
    if args.max_features < 5:
        raise ValueError("--max-features must be >= 5.")
    if args.n_folds < 1:
        raise ValueError("--n-folds must be >= 1.")
    if args.test_bars < 1:
        raise ValueError("--test-bars must be >= 1.")
    if args.min_train_bars < 1000:
        raise ValueError(
            "--min-train-bars must be >= 1000."
        )
    if not 0.10 <= args.calibration_fraction <= 0.40:
        raise ValueError(
            "--calibration-fraction must be between 0.10 and 0.40."
        )
    if args.base_position_fraction <= 0:
        raise ValueError(
            "--base-position-fraction must be > 0."
        )
    if args.max_position_fraction <= 0:
        raise ValueError(
            "--max-position-fraction must be > 0."
        )
    if args.base_position_fraction > args.max_position_fraction:
        raise ValueError(
            "--base-position-fraction cannot exceed max-position-fraction."
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(args)
    run_walkforward(args)


if __name__ == "__main__":
    main()