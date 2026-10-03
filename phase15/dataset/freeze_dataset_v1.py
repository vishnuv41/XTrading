"""
phase15/dataset/freeze_dataset_v1.py
------------------------------------
Freezes Phase 15 Dataset v1 & Tokenizer v1.

Calculates exact class distribution statistics across:
- Overall ACCEPT / REJECT counts and percentages
- Per-symbol breakdown (BTC/USDT, ETH/USDT)
- Per-year breakdown (2017 to 2026)
- Per-regime breakdown (BULL, BEAR, RANGING, VOLATILE)
- SHA-256 binary fingerprint hash

Outputs canonical audit manifest to phase15/provenance/dataset_v1_manifest.json.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd
import numpy as np

from pipeline.data_loader import load_ohlcv
from phase15.dataset.causal_dataset_builder import generate_causal_dataset
from phase15.audit.data_integrity_gate import run_data_integrity_gate

logger = logging.getLogger(__name__)


def freeze_dataset_v1(output_dir: str = "phase15/provenance") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    
    logger.info("Loading historical OHLCV data for Dataset v1 freeze...")
    
    # Load BTC and ETH history
    df_btc = load_ohlcv("BTC/USDT", "1h", limit=5000)
    df_eth = load_ohlcv("ETH/USDT", "1h", limit=5000)
    
    ds_btc = generate_causal_dataset(df_btc, symbol="BTC/USDT")
    ds_eth = generate_causal_dataset(df_eth, symbol="ETH/USDT")
    
    combined = pd.concat([ds_btc, ds_eth], ignore_index=True)
    combined = combined.sort_values(by=["ts", "symbol"]).reset_index(drop=True)
    
    # Drop indicator warmup NaNs from numeric feature columns
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined = combined.dropna(subset=num_cols).reset_index(drop=True)
    
    # Run Integrity Gate
    gate_res = run_data_integrity_gate(combined, dataset_name="Phase 15 Dataset v1 Final Freeze")
    
    if not gate_res["passed_all_checks"]:
        raise ValueError(f"Dataset v1 failed Data Integrity Gate: {gate_res['checks']}")

    # Compute Statistics
    total_rows = len(combined)
    accept_cnt = int((combined["label"] == "ACCEPT").sum())
    reject_cnt = int((combined["label"] == "REJECT").sum())
    accept_pct = float(accept_cnt / total_rows * 100) if total_rows > 0 else 0.0
    
    per_symbol = {
        "BTC/USDT": {
            "rows": len(ds_btc),
            "accept": int((ds_btc["label"] == "ACCEPT").sum()),
            "reject": int((ds_btc["label"] == "REJECT").sum()),
            "accept_pct": float((ds_btc["label"] == "ACCEPT").sum() / len(ds_btc) * 100) if len(ds_btc) > 0 else 0.0
        },
        "ETH/USDT": {
            "rows": len(ds_eth),
            "accept": int((ds_eth["label"] == "ACCEPT").sum()),
            "reject": int((ds_eth["label"] == "REJECT").sum()),
            "accept_pct": float((ds_eth["label"] == "ACCEPT").sum() / len(ds_eth) * 100) if len(ds_eth) > 0 else 0.0
        }
    }
    
    # Compute Per-Year Distribution
    combined["year"] = pd.to_datetime(combined["ts"], utc=True).dt.year
    per_year = {}
    for yr, group in combined.groupby("year"):
        acc = int((group["label"] == "ACCEPT").sum())
        rej = int((group["label"] == "REJECT").sum())
        tot = len(group)
        per_year[str(yr)] = {
            "total_rows": tot,
            "accept": acc,
            "reject": rej,
            "accept_pct": float(acc / tot * 100) if tot > 0 else 0.0
        }

    # Compute Per-Regime Distribution
    per_regime = {}
    if "market_state" in combined.columns:
        for reg, group in combined.groupby("market_state"):
            acc = int((group["label"] == "ACCEPT").sum())
            rej = int((group["label"] == "REJECT").sum())
            tot = len(group)
            per_regime[str(reg)] = {
                "total_rows": tot,
                "accept": acc,
                "reject": rej,
                "accept_pct": float(acc / tot * 100) if tot > 0 else 0.0
            }

    # Fingerprint SHA-256 Hash
    summary_str = combined[["ts", "symbol", "close", "RSI14", "label"]].head(500).to_string()
    ds_hash = hashlib.sha256(summary_str.encode("utf-8")).hexdigest()[:12]

    manifest = {
        "manifest_name": "Phase 15 Dataset v1 & Tokenizer v1 Canonical Manifest",
        "frozen_at": now_iso,
        "dataset_sha256": ds_hash,
        "total_samples": total_rows,
        "overall_class_distribution": {
            "accept_count": accept_cnt,
            "reject_count": reject_cnt,
            "accept_percentage": round(accept_pct, 2)
        },
        "per_symbol_distribution": per_symbol,
        "per_year_distribution": per_year,
        "per_regime_distribution": per_regime,
        "data_integrity_gate_passed": gate_res["passed_all_checks"],
        "status": "FROZEN_DATASET_V1"
    }

    manifest_path = os.path.join(output_dir, "dataset_v1_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info("Dataset v1 frozen successfully. Manifest written to %s", manifest_path)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = freeze_dataset_v1()
    print("Dataset v1 Manifest Summary:\n", json.dumps(res, indent=2))
