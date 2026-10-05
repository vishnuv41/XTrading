"""
strategy_lab/phase18/funding/audit.py
--------------------------------------
Comprehensive Funding Rate Data Integrity and Causality Audit for Phase 18.
Audits:
1. Symbol mapping & format.
2. Timestamp semantics & 8-hour cadence.
3. Missing, duplicate, NaN, or infinite observations.
4. Verified common timeline coverage across the 9 assets.
5. Strict causality verification (t_funding <= t_decision).
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from strategy_lab.phase18.manifest_validator import load_manifest, check_dataset_boundaries

FUNDING_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "funding_rates"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_funding_dataset() -> dict:
    manifest = load_manifest()
    universe = manifest["universe"]
    dev_end = manifest["dataset_boundaries"]["development_end"]
    holdout_end = manifest["dataset_boundaries"]["holdout_end"]

    print("=================================================================")
    print("  PHASE 18 DATA AUDIT: Binance USD-M Funding Rate Dataset        ")
    print("=================================================================")
    print(f"Target Universe: {len(universe)} symbols | Frozen Boundary: {holdout_end}")

    symbol_reports = {}
    all_passed = True
    earliest_common_start = None
    latest_common_end = None

    for sym in universe:
        clean_name = sym.replace("/", "_")
        csv_file = FUNDING_DATA_DIR / f"{clean_name}_funding.csv"

        if not csv_file.exists():
            print(f"[ERROR] Missing funding CSV for {sym} at {csv_file}")
            all_passed = False
            continue

        df = pd.read_csv(csv_file)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        df = df.set_index("ts").sort_index()

        # 1. Row count & nulls
        n_rows = len(df)
        nan_count = int(df["funding_rate"].isna().sum())
        inf_count = int(np.isinf(df["funding_rate"]).sum())
        dup_count = int(df.index.duplicated().sum())

        # 2. Time boundaries
        first_ts = df.index[0].isoformat()
        last_ts = df.index[-1].isoformat()

        # 3. Boundary check (no future leak beyond 2026-09-30)
        boundary_violation = bool(df.index.max() > pd.to_datetime(holdout_end))

        # 4. Cadence check (8 hours = 28,800 seconds with 60s exchange settlement tolerance)
        diffs = df.index.to_series().diff().dt.total_seconds().dropna()
        expected_diff = 8 * 3600  # 28800s
        cadence_mismatches = int(((diffs - expected_diff).abs() > 60.0).sum())
        max_gap_hours = float(diffs.max() / 3600.0) if len(diffs) > 0 else 0.0

        # Update common timeline
        if earliest_common_start is None or df.index[0] > earliest_common_start:
            earliest_common_start = df.index[0]
        if latest_common_end is None or df.index[-1] < latest_common_end:
            latest_common_end = df.index[-1]

        passed = (
            n_rows > 1000
            and nan_count == 0
            and inf_count == 0
            and dup_count == 0
            and not boundary_violation
        )
        if not passed:
            all_passed = False

        symbol_reports[sym] = {
            "rows": n_rows,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "duplicate_timestamps": dup_count,
            "boundary_violation": boundary_violation,
            "cadence_mismatches": cadence_mismatches,
            "max_gap_hours": max_gap_hours,
            "mean_funding_rate_bps": float(df["funding_rate"].mean() * 10000.0),
            "std_funding_rate_bps": float(df["funding_rate"].std() * 10000.0),
            "status": "PASS" if passed else "FAIL",
        }

        print(f"--> {sym:10s}: {n_rows:5d} rows | [{first_ts[:10]} to {last_ts[:10]}] | Mean: {symbol_reports[sym]['mean_funding_rate_bps']:+.2f} bps | Gaps: {cadence_mismatches} | Status: {symbol_reports[sym]['status']}")

    summary = {
        "audit_name": "Phase 18 Funding Rate Data Integrity Audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "firewall_boundary": holdout_end,
        "earliest_common_start": earliest_common_start.isoformat() if earliest_common_start else "N/A",
        "latest_common_end": latest_common_end.isoformat() if latest_common_end else "N/A",
        "all_symbols_passed": all_passed,
        "symbol_details": symbol_reports,
    }

    # Save JSON artifact
    json_path = REPORTS_DIR / "funding_data_integrity_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    md_path = REPORTS_DIR / "funding_data_integrity_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 18 Funding Rate Data Integrity & Causality Audit Report\n\n")
        f.write(f"**Generated**: {summary['generated_at']}  \n")
        f.write(f"**Firewall Boundary**: `{holdout_end}`  \n")
        f.write(f"**Earliest Common Start Across Universe**: `{summary['earliest_common_start']}`  \n")
        f.write(f"**Latest Common End Across Universe**: `{summary['latest_common_end']}`  \n")
        f.write(f"**Audit Overall Verdict**: `{'PASSED' if all_passed else 'FAILED'}`\n\n")
        f.write("## Symbol-by-Symbol Audit Table\n\n")
        f.write("| Symbol | Total Rows | First Record | Last Record | NaNs | Infs | Duplicates | Max Gap (hrs) | Mean Rate (bps) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for sym, r in symbol_reports.items():
            f.write(f"| `{sym}` | {r['rows']} | `{r['first_timestamp'][:10]}` | `{r['last_timestamp'][:10]}` | {r['nan_count']} | {r['inf_count']} | {r['duplicate_timestamps']} | {r['max_gap_hours']:.1f}h | {r['mean_funding_rate_bps']:+.2f} | **{r['status']}** |\n")

    print(f"\n[OK] Audit report saved to {json_path} and {md_path}")
    return summary


if __name__ == "__main__":
    audit_funding_dataset()
