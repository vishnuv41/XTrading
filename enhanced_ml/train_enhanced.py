"""
enhanced_ml/train_enhanced.py
--------------------------------
Adapter that wires enhanced_ml's TP-before-SL labeling and MTF features
into ml.train.run_training_pipeline via the label_fn / feature_fn /
label_map hooks added to that function. This module does NOT duplicate
any training logic — purged CV, the stacking ensemble, calibration, the
class-integrity check, and artifact saving all still run exactly as
they do for V1, from inside the same run_training_pipeline call.

What this file actually adds:
  1. build_tp_before_sl_label_fn() — wraps
     enhanced_ml.labeling.tp_before_sl.label_tp_before_sl (which takes
     side/atr_col/... kwargs and returns label/entry_price/stop_loss/
     take_profit/touch_type/holding_bars) into the single-arg
     `label_fn(feat_df) -> DataFrame[label, touch_idx]` shape
     run_training_pipeline expects — matching exactly what
     ml.labeling.triple_barrier.triple_barrier_labels returns, so
     prepare_dataset's downstream handling (dropna, PurgedKFold purging
     via touch_idx) needs no changes.

  2. build_mtf_feature_fn() — thin wrapper around
     enhanced_ml.features.feature_engine.FeatureEngine +
     enhanced_ml.features.mtf_features.make_mtf_feature_fn, matching the
     `feature_fn(df, has_volume=has_volume) -> feat_df` shape.

  3. run_enhanced_training() — calls run_training_pipeline with those
     two hooks plus label_map={0: 0, 1: 1} (TP-before-SL's labels are
     already binary 0/1, so this is an identity map — no remapping to
     the triple-barrier 3-class {-1,0,1}->{0,1,2} scheme). Defaults
     output_dir to models_artifacts/<SYMBOL>_<timeframe>_v2 — a
     DIFFERENT directory from V1's models_artifacts/<SYMBOL>_<timeframe>
     (see run_training_from_db.py's _default_output_dir), so V1
     artifacts are never touched or overwritten by an enhanced run.

touch_idx derivation: tp_before_sl's `holding_bars[i]` is the 1-indexed
count of bars from i+1 until the touching bar (first_tp/first_sl + 1).
The absolute positional index of that touch is therefore i + holding_bars
— this matches triple_barrier_labels' touch_idx convention exactly (see
ml/labeling/triple_barrier.py's docstring: "positional index where the
barrier was touched"), so PurgedKFold's purging logic needs no changes
either.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.train import run_training_pipeline
from enhanced_ml.labeling.tp_before_sl import label_tp_before_sl
from enhanced_ml.features.feature_engine import FeatureEngine
from enhanced_ml.features.mtf_features import make_mtf_feature_fn

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def build_tp_before_sl_label_fn(
    side: str = "long",
    atr_col: str = "ATR14",
    sl_multiplier: float = 2.0,
    risk_reward_ratio: float = 2.0,
    max_holding: int = 48,
):
    """
    Returns a `label_fn(feat_df) -> DataFrame[label, touch_idx]` closure
    matching run_training_pipeline's expected contract, built from
    label_tp_before_sl with the given side/sizing fixed.
    """

    def _label_fn(feat_df: pd.DataFrame) -> pd.DataFrame:
        raw = label_tp_before_sl(
            feat_df,
            side=side,
            atr_col=atr_col,
            sl_multiplier=sl_multiplier,
            risk_reward_ratio=risk_reward_ratio,
            max_holding=max_holding,
        )

        n = len(feat_df)
        positions = np.arange(n)
        touch_idx = positions + raw["holding_bars"].to_numpy()  # NaN stays NaN where holding_bars is NaN

        return pd.DataFrame(
            {"label": raw["label"].values, "touch_idx": touch_idx},
            index=feat_df.index,
        )

    _label_fn.__name__ = f"tp_before_sl_label_fn(side={side})"
    return _label_fn


def build_mtf_feature_fn(symbol: str, higher_timeframes: list[str], exchange: str):
    """
    Returns a `feature_fn(df, has_volume=has_volume) -> feat_df` closure:
    V1's base feature set (via FeatureEngine, which itself just calls
    ml.utils.preprocessing.build_feature_matrix — same 115 features,
    unchanged) plus the curated higher-timeframe columns from
    mtf_features.make_mtf_feature_fn registered on top.
    """

    def _feature_fn(df: pd.DataFrame, has_volume: bool = True) -> pd.DataFrame:
        engine = FeatureEngine()
        engine.register(make_mtf_feature_fn(symbol, higher_timeframes, exchange=exchange, has_volume=has_volume))
        return engine.build(df, has_volume=has_volume)

    _feature_fn.__name__ = f"mtf_feature_fn({symbol},{higher_timeframes})"
    return _feature_fn


def default_v2_output_dir(symbol: str, timeframe: str) -> str:
    """
    models_artifacts/<SYMBOL>_<timeframe>_v2 — deliberately a sibling of,
    never the same as, V1's models_artifacts/<SYMBOL>_<timeframe> (see
    run_training_from_db.py's _default_output_dir). No V1 artifact
    directory is ever written to by this module.
    """
    return f"models_artifacts/{symbol.replace('/', '')}_{timeframe}_v2"


def run_enhanced_training(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    higher_timeframes: list[str],
    exchange: str,
    side: str = "long",
    sl_multiplier: float = 2.0,
    risk_reward_ratio: float = 2.0,
    max_holding: int = 48,
    n_splits: int = 5,
    output_dir: str = None,
) -> dict:
    """
    Runs the enhanced (TP-before-SL label + MTF features) pipeline
    through the SAME run_training_pipeline used by V1 — purged CV,
    stacking ensemble, calibration, integrity check, and artifact
    saving are byte-identical code paths to V1's, just fed different
    features/labels via the hooks.

    Returns the same dict run_training_pipeline returns:
    {'ensemble', 'calibrator', 'X_columns', 'metrics'}.
    """
    output_dir = output_dir or default_v2_output_dir(symbol, timeframe)

    label_fn = build_tp_before_sl_label_fn(
        side=side, sl_multiplier=sl_multiplier,
        risk_reward_ratio=risk_reward_ratio, max_holding=max_holding,
    )
    feature_fn = build_mtf_feature_fn(symbol, higher_timeframes, exchange=exchange)

    return run_training_pipeline(
        df,
        n_splits=n_splits,
        output_dir=output_dir,
        feature_fn=feature_fn,
        label_fn=label_fn,
        label_map={0: 0, 1: 1},  # tp_before_sl labels are already binary — identity map
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the enhanced (TP-before-SL + MTF) model. Never touches V1 artifacts."
    )
    parser.add_argument("--symbol", default=settings.symbols.symbols[0], help="e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="e.g. 1h, 15m, 1d")
    parser.add_argument("--exchange", default=settings.exchange.default_exchange)
    parser.add_argument("--higher-timeframes", nargs="+", default=["4h", "1d"])
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--risk-reward", type=float, default=2.0)
    parser.add_argument("--max-holding", type=int, default=48)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", default=None,
                         help="Defaults to models_artifacts/<SYMBOL>_<timeframe>_v2 — "
                              "never the V1 directory.")
    args = parser.parse_args()
    output_dir = args.output_dir or default_v2_output_dir(args.symbol, args.timeframe)

    logger.info("Loading %s %s from %s...", args.symbol, args.timeframe, args.exchange)
    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)

    logger.info("Training enhanced model (side=%s, higher_tf=%s) on %d candles...",
                args.side, args.higher_timeframes, len(df))
    result = run_enhanced_training(
        df, symbol=args.symbol, timeframe=args.timeframe,
        higher_timeframes=args.higher_timeframes, exchange=args.exchange,
        side=args.side, sl_multiplier=args.sl_mult,
        risk_reward_ratio=args.risk_reward, max_holding=args.max_holding,
        n_splits=args.n_splits, output_dir=output_dir,
    )

    print("\nEnhanced training complete.")
    print(json.dumps({k: v for k, v in result["metrics"].items()}, indent=2, default=str))
    print(f"Artifacts saved to: {output_dir}/  (V1 artifacts untouched)")


if __name__ == "__main__":
    main()
