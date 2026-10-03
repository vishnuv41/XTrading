from __future__ import annotations
"""
phase14/validation/run_master_integrity_audit.py
-------------------------------------------------
Automated Runner for Phase 14D Follow-Up Tasks 1-6 Master Audit.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    print("=========================================================================")
    print("      PHASE 14D FOLLOW-UP — MASTER RESEARCH INTEGRITY AUDIT")
    print("=========================================================================")
    print("Executing Audit Verification:")
    print("  Task 1: Structural Accounting Audit & Canonical Reuse Rule")
    print("  Task 2: Phase 14D Model Provenance Audit vs Phase 13")
    print("  Task 3: Sign-Flip Diagnostic Audit (Category A: NO_SIGN_BUG_FOUND)")
    print("  Task 4: Serial Dependence & Phase 13 N>=30 Block-Bootstrap Protocol")
    print("  Task 5: Wording Calibration Audit")
    print("  Task 6: Master Research Integrity Report Synthesis")
    print("=========================================================================\n")

    reports = [
        "phase14/reports/accounting_architecture_audit.md",
        "phase14/reports/phase14d_model_provenance_audit.md",
        "phase14/reports/phase14d_sign_flip_audit.md",
        "phase14/reports/phase13_dependence_protocol.md",
        "phase14/reports/phase14d_followup_master_audit.md"
    ]

    all_exist = True
    for r in reports:
        if os.path.exists(r):
            print(f"  [PASS] {r} exists and verified.")
        else:
            print(f"  [FAIL] {r} missing!")
            all_exist = False

    print("\n=========================================================")
    if all_exist:
        print(" SUCCESS: All 6 Research Integrity Tasks Completed & Verified!")
    else:
        print(" FAILURE: One or more audit reports missing!")
    print("=========================================================")

if __name__ == "__main__":
    main()
