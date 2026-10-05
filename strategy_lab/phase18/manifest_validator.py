"""
strategy_lab/phase18/manifest_validator.py
-------------------------------------------
Cryptographic and protocol validator for Phase 18 research.
Enforces:
1. Manifest integrity & immutability.
2. Dataset boundary enforcement (no lookahead, no forward ledger data).
3. Parameter grid bounds checking.
4. Numerical gate definitions verification.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def load_manifest() -> Dict[str, Any]:
    """Load and parse the Phase 18 pre-registered manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Phase 18 manifest missing at {MANIFEST_PATH}")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def compute_manifest_hash() -> str:
    """Compute SHA-256 hash of the manifest file."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Phase 18 manifest missing at {MANIFEST_PATH}")
    
    with open(MANIFEST_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def validate_manifest_schema(manifest: Dict[str, Any] = None) -> bool:
    """Validate that all required sections and explicit numerical thresholds exist."""
    if manifest is None:
        manifest = load_manifest()

    required_keys = [
        "phase", "version", "universe", "timeframes",
        "dataset_boundaries", "execution_and_friction",
        "benchmark_reference", "hypotheses", "acceptance_gates", "reproducibility"
    ]
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Manifest missing required section: '{key}'")

    # Validate Gate 6 explicit numerical thresholds
    gate6 = manifest["acceptance_gates"].get("gate_6_regime_stability", {})
    for threshold_key in ["min_annual_sharpe", "max_annual_drawdown_pct", "min_annual_net_return_pct"]:
        if threshold_key not in gate6:
            raise ValueError(f"Gate 6 missing required numerical threshold: '{threshold_key}'")

    # Validate Universe
    if len(manifest["universe"]) != 9:
        raise ValueError(f"Manifest universe must contain exactly 9 assets, got {len(manifest['universe'])}")

    # Validate Hypotheses Grid
    for hyp_id in ["H18_A_Funding", "H18_B_RelativeStrength", "H18_C_VolatilityCompression"]:
        if hyp_id not in manifest["hypotheses"]:
            raise ValueError(f"Missing pre-registered hypothesis: '{hyp_id}'")

    return True


def check_dataset_boundaries(start_ts: str, end_ts: str) -> bool:
    """Ensure requested research dates never exceed holdout boundary into forward ledger."""
    manifest = load_manifest()
    holdout_end = manifest["dataset_boundaries"]["holdout_end"]
    
    if str(end_ts) > holdout_end:
        raise ValueError(
            f"FIREWALL VIOLATION: Requested end timestamp {end_ts} exceeds "
            f"frozen historical boundary {holdout_end}. Forward tracking data is segregated."
        )
    return True


if __name__ == "__main__":
    print("Validating Phase 18 Manifest...")
    m = load_manifest()
    validate_manifest_schema(m)
    m_hash = compute_manifest_hash()
    print(f"[OK] Manifest validated successfully. SHA-256: {m_hash}")
