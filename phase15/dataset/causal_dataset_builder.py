"""
phase15/dataset/causal_dataset_builder.py
-----------------------------------------
Point-in-Time Causal Dataset Builder for Phase 15.

Generates leakage-audited causal market snapshot datasets for Phase 15 modeling:
1. Loads historical OHLCV from DB or CSV.
2. Computes causal indicators and market regimes.
3. Formulates triple-barrier outcome labels (TP 3.0x ATR, SL 1.5x ATR, max holding 48 bars).
4. Formats discrete financial tokens for Mini-LLM / QLoRA sequence tokenization.
5. Runs TemporalLeakageAuditor to verify 100% lookahead-free integrity before saving.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from indicators import calculate_all_indicators
from regime import calculate_market_state
from ml.labeling import triple_barrier_labels
from phase15.audit.leakage_auditor import TemporalLeakageAuditor

logger = logging.getLogger(__name__)


def discretize_rsi(rsi: float) -> str:
    if np.isnan(rsi): return "RSI_NEUTRAL"
    if rsi < 30: return "RSI_OVERSOLD"
    if rsi < 45: return "RSI_WEAK"
    if rsi <= 55: return "RSI_NEUTRAL"
    if rsi <= 70: return "RSI_BULLISH"
    return "RSI_OVERBOUGHT"


def discretize_atr_pct(atr_pct: float) -> str:
    if np.isnan(atr_pct): return "VOL_NORMAL"
    if atr_pct < 0.008: return "VOL_LOW"
    if atr_pct < 0.020: return "VOL_NORMAL"
    if atr_pct < 0.035: return "VOL_HIGH"
    return "VOL_EXTREME"


def discretize_volume_z(z: float) -> str:
    if np.isnan(z): return "VOLUMENUM_NORMAL"
    if z < -1.0: return "VOLUMENUM_DRY"
    if z <= 1.0: return "VOLUMENUM_NORMAL"
    if z <= 2.5: return "VOLUMENUM_HIGH"
    return "VOLUMENUM_SPIKE"


def build_causal_snapshot(row: pd.Series, symbol: str) -> Dict[str, Any]:
    """Builds a point-in-time tokenized market snapshot dictionary for a single candle row."""
    symbol_token = f"<{symbol.replace('/', '')}>"
    state_token = f"<{row.get('market_state', 'NORMAL').upper()}>"
    rsi_tok = f"<{discretize_rsi(row.get('RSI14', 50))}>"
    vol_tok = f"<{discretize_atr_pct(row.get('ATR14', 0) / row.get('close', 1))}>"
    volum_tok = f"<{discretize_volume_z(row.get('VOLUME_ZSCORE_20', 0))}>"
    
    tokens = [symbol_token, state_token, rsi_tok, vol_tok, volum_tok]
    
    return {
        "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else str(row["ts"]),
        "symbol": symbol,
        "tokens": tokens,
        "features": {
            "close": float(row.get("close", 0)),
            "rsi14": float(row.get("RSI14", 50)) if not np.isnan(row.get("RSI14", 50)) else 50.0,
            "atr14": float(row.get("ATR14", 0)) if not np.isnan(row.get("ATR14", 0)) else 0.0,
            "market_state": str(row.get("market_state", "NORMAL"))
        }
    }


def generate_causal_dataset(
    df: pd.DataFrame,
    symbol: str = "BTC/USDT",
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 48,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes indicators, labels, token snapshots, and runs leakage audit.
    """
    logger.info("Building causal dataset for %s (%d rows)...", symbol, len(df))
    
    df["symbol"] = symbol
    if "timestamp" in df.columns and "ts" not in df.columns:
        df["ts"] = df["timestamp"]
        
    # 1. Compute Indicators & Regime
    df = calculate_all_indicators(df)
    df = calculate_market_state(df)
    
    # 2. Compute Triple-Barrier Outcome Labels
    volatility = df["ATR14"] / df["close"] if "ATR14" in df.columns else None
    tb_labels = triple_barrier_labels(
        df, volatility=volatility, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding
    )
    
    df["label_raw"] = tb_labels["label"]
    # Map label: 1 (PT hit) -> ACCEPT; -1 (SL hit) or 0 (timeout) -> REJECT
    df["label"] = df["label_raw"].map({1: "ACCEPT", -1: "REJECT", 0: "REJECT"})
    
    # Drop rows without labels or indicators
    df_clean = df.dropna(subset=["close", "RSI14", "ATR14", "label"]).copy().reset_index(drop=True)
    
    # 3. Run Leakage Audit
    auditor = TemporalLeakageAuditor()
    audit_res = auditor.audit_dataframe(df_clean, timestamp_col="ts", label_col="label")
    
    if not audit_res["passed"]:
        raise ValueError(f"Dataset failed temporal leakage audit: {audit_res['errors']}")
        
    logger.info("Leakage audit passed cleanly. Clean dataset size: %d samples.", len(df_clean))
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_parquet(output_path, index=False)
        logger.info("Saved causal dataset to %s", output_path)
        
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Self-test with synthetic OHLCV
    dates = pd.date_range("2020-01-01", periods=500, freq="1h", tz="UTC")
    close = np.cumsum(np.random.normal(0, 1, 500)) + 100
    df = pd.DataFrame({
        "ts": dates,
        "open": close + np.random.normal(0, 0.2, 500),
        "high": close + np.random.random(500) * 1.0,
        "low": close - np.random.random(500) * 1.0,
        "close": close,
        "volume": np.random.randint(100, 10000, 500)
    })
    
    ds = generate_causal_dataset(df, symbol="BTC/USDT")
    print(f"Sample generated row:\n{ds[['ts', 'close', 'RSI14', 'label']].head(3)}")
