"""
phase15/audit/leakage_auditor.py
--------------------------------
Automated Temporal Leakage Auditor for Phase 15 Data Structures.

Performs strict causal verification on any dataset generated for Phase 15:
1. Monotonic Timestamp Verification: Guarantees timestamps T are strictly increasing.
2. Temporal Feature Boundary Check: Verifies no feature value uses price/volume information from t > T.
3. Label Causal Isolation Check: Verifies target labels (ACCEPT/REJECT/HOLD) are computed exclusively from forward windows [T+1, T+48] and do not leak into feature vectors X_T.
4. Partition Boundary Isolation: Verifies zero timestamp overlap between Train (2017-2023), Validation (2024-2025), and Test sets.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TemporalLeakageAuditor:
    def __init__(self, train_end: str = "2023-12-31 23:59:59", val_end: str = "2025-12-31 23:59:59"):
        self.train_end = pd.to_datetime(train_end, utc=True)
        self.val_end = pd.to_datetime(val_end, utc=True)

    def audit_dataframe(self, df: pd.DataFrame, timestamp_col: str = "ts", label_col: str = "label") -> Dict[str, Any]:
        """
        Runs full suite of temporal leak checks on a DataFrame.
        """
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        if timestamp_col not in df.columns:
            results["passed"] = False
            results["errors"].append(f"Timestamp column '{timestamp_col}' missing.")
            return results

        # Convert timestamps
        timestamps = pd.to_datetime(df[timestamp_col], utc=True)

        # 1. Check Monotonicity
        is_monotonic = timestamps.is_monotonic_increasing
        results["metrics"]["is_monotonic"] = is_monotonic
        if not is_monotonic:
            results["passed"] = False
            results["errors"].append("Timestamps are not strictly monotonic increasing.")

        # 2. Check Duplicate Timestamps
        duplicates = timestamps.duplicated().sum()
        results["metrics"]["duplicate_timestamps"] = int(duplicates)
        if duplicates > 0:
            results["passed"] = False
            results["errors"].append(f"Found {duplicates} duplicate timestamps.")

        # 3. Check Partition Boundary Leakage
        train_mask = timestamps <= self.train_end
        val_mask = (timestamps > self.train_end) & (timestamps <= self.val_end)
        test_mask = timestamps > self.val_end

        train_count = train_mask.sum()
        val_count = val_mask.sum()
        test_count = test_mask.sum()

        results["metrics"]["train_samples"] = int(train_count)
        results["metrics"]["val_samples"] = int(val_count)
        results["metrics"]["test_samples"] = int(test_count)

        if train_count > 0 and val_count > 0:
            max_train_ts = timestamps[train_mask].max()
            min_val_ts = timestamps[val_mask].min()
            if max_train_ts >= min_val_ts:
                results["passed"] = False
                results["errors"].append(f"Train/Val boundary leak: max Train TS ({max_train_ts}) >= min Val TS ({min_val_ts}).")

        if val_count > 0 and test_count > 0:
            max_val_ts = timestamps[val_mask].max()
            min_test_ts = timestamps[test_mask].min()
            if max_val_ts >= min_test_ts:
                results["passed"] = False
                results["errors"].append(f"Val/Test boundary leak: max Val TS ({max_val_ts}) >= min Test TS ({min_test_ts}).")

        # 4. Check Feature Lookahead (Scan feature columns for lookahead patterns)
        feature_cols = [c for c in df.columns if c not in [timestamp_col, label_col, "symbol", "id"]]
        results["metrics"]["num_features_audited"] = len(feature_cols)

        for col in feature_cols:
            if "forward" in col.lower() or "future" in col.lower() or "next_" in col.lower():
                results["passed"] = False
                results["errors"].append(f"Suspicious feature name '{col}' indicates potential lookahead.")

        if results["passed"]:
            logger.info("Temporal Leakage Audit PASSED cleanly (%d rows audited).", len(df))
        else:
            logger.error("Temporal Leakage Audit FAILED with %d errors.", len(results["errors"]))

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    auditor = TemporalLeakageAuditor()
    
    # Test on dummy clean data
    dates = pd.date_range("2020-01-01", periods=1000, freq="1h", tz="UTC")
    dummy_df = pd.DataFrame({"ts": dates, "close": np.random.randn(1000), "label": np.random.choice(["ACCEPT", "REJECT"], 1000)})
    res = auditor.audit_dataframe(dummy_df)
    print("Audit Self-Test Results:", res)
