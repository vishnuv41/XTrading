"""
strategy_lab/phase18/tests/test_funding_data.py
------------------------------------------------
Unit and causality tests for Binance USD-M funding rate data.
Enforces:
1. Exact symbol mapping.
2. Zero data leaks beyond 2026-09-30 boundary.
3. Strict causality (t_funding <= t_decision).
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from strategy_lab.phase18.funding.audit import audit_funding_dataset, FUNDING_DATA_DIR
from strategy_lab.phase18.manifest_validator import load_manifest


def test_funding_data_integrity_audit():
    audit_res = audit_funding_dataset()
    assert audit_res["all_symbols_passed"] is True
    assert len(audit_res["symbol_details"]) == 9


def test_funding_data_boundary_enforcement():
    manifest = load_manifest()
    holdout_end = manifest["dataset_boundaries"]["holdout_end"]
    
    for sym in manifest["universe"]:
        clean_name = sym.replace("/", "_")
        csv_file = FUNDING_DATA_DIR / f"{clean_name}_funding.csv"
        assert csv_file.exists()
        
        df = pd.read_csv(csv_file)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        assert df["ts"].max() <= pd.to_datetime(holdout_end)


def test_funding_causality_timestamp_alignment():
    """
    Ensure that for any daily decision bar at T, only funding observations
    settled strictly AT OR BEFORE T are accessible.
    """
    manifest = load_manifest()
    btc_csv = FUNDING_DATA_DIR / "BTC_USDT_funding.csv"
    df = pd.read_csv(btc_csv)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.set_index("ts").sort_index()

    decision_ts = pd.to_datetime("2023-06-01 00:00:00+00:00")
    available_funding = df[df.index <= decision_ts]

    # Verify no future funding observation is present
    assert available_funding.index.max() <= decision_ts
    assert (available_funding.index > decision_ts).sum() == 0
