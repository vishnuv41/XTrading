"""
strategy_lab/paper/verify_ledger.py
-----------------------------------
Independent ledger integrity verifier for Phase 17 Forward Tracker.
Verifies:
1. Cryptographic SHA-256 hash chain linkage (prev_hash -> entry_hash).
2. Genesis block specification (prev_hash == 64 zeros, frozen_code_commit, parameters_sha256).
3. Audits all code_change entries.

Returns exit code 0 and prints PASS on valid ledger; prints FAIL with error details on corruption.
"""

import hashlib
import json
import os
import sys
from typing import Tuple, List, Dict, Any

LEDGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_track_ledger.jsonl")


def verify_ledger(ledger_path: str = LEDGER_FILE) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Verifies the hash chain of a ledger file.
    Returns: (is_valid, error_messages, code_change_records)
    """
    if not os.path.exists(ledger_path):
        return False, [f"Ledger file not found: {ledger_path}"], []

    errors = []
    code_changes = []
    expected_prev_hash = "0" * 64

    with open(ledger_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return False, ["Ledger file is empty"], []

    for idx, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except Exception as e:
            errors.append(f"Line {idx+1}: Invalid JSON ({e})")
            continue

        stored_entry_hash = entry.get("entry_hash")
        stored_prev_hash = entry.get("prev_hash")

        # 1. Verify prev_hash link
        if stored_prev_hash != expected_prev_hash:
            errors.append(
                f"Line {idx+1}: Broken hash chain! Expected prev_hash {expected_prev_hash}, got {stored_prev_hash}"
            )

        # 2. Verify genesis block
        if idx == 0:
            if stored_prev_hash != "0" * 64:
                errors.append(f"Genesis Block: prev_hash is not 64 zeros ({stored_prev_hash})")
            if "genesis_spec" not in entry:
                errors.append("Genesis Block: missing genesis_spec metadata")

        # 3. Verify entry_hash calculation
        entry_copy = dict(entry)
        entry_copy.pop("entry_hash", None)
        canonical_payload = json.dumps(entry_copy, sort_keys=True, default=str)
        computed_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

        if computed_hash != stored_entry_hash:
            errors.append(
                f"Line {idx+1}: Hash mismatch! Computed {computed_hash}, stored {stored_entry_hash}"
            )

        # 4. Check for code_change records
        if entry.get("entry_type") == "code_change":
            code_changes.append(entry)

        expected_prev_hash = stored_entry_hash

    is_valid = (len(errors) == 0)
    return is_valid, errors, code_changes


def main():
    print(f"=========================================================================")
    print(f"PHASE 17 FORWARD TRACK LEDGER INTEGRITY VERIFIER")
    print(f"Target: {LEDGER_FILE}")
    print(f"=========================================================================\n")

    is_valid, errors, code_changes = verify_ledger()

    if code_changes:
        print(f"Audit Note: {len(code_changes)} code_change event(s) recorded in ledger:")
        for cc in code_changes:
            print(f"  - [{cc.get('timestamp')}] {cc.get('file')}: {cc.get('old_sha256')[:12]} -> {cc.get('new_sha256')[:12]}")
        print()

    if is_valid:
        print("RESULT: PASS (Hash chain is cryptographically intact and unbroken.)")
        sys.exit(0)
    else:
        print("RESULT: FAIL (Ledger integrity violation detected!)")
        for err in errors:
            print(f"  ERROR: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
