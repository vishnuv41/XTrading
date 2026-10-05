"""
tests/test_verify_ledger.py
---------------------------
Unit tests for the Phase 17 Forward Track Ledger Verifier.
"""

import os
import json
import pytest

from strategy_lab.paper.forward_baseline_tracker import (
    append_to_hash_chained_ledger,
    LEDGER_FILE,
)
from strategy_lab.paper.verify_ledger import verify_ledger


def test_verify_valid_ledger(tmp_path):
    test_ledger = str(tmp_path / "valid_ledger.jsonl")
    
    # Generate 3 valid chained entries
    from unittest.mock import patch
    with patch("strategy_lab.paper.forward_baseline_tracker.LEDGER_FILE", test_ledger):
        append_to_hash_chained_ledger({"entry_type": "daily_bar", "bar": 1})
        append_to_hash_chained_ledger({"entry_type": "daily_bar", "bar": 2})
        append_to_hash_chained_ledger({"entry_type": "code_change", "file": "test.py", "old_sha256": "aaa", "new_sha256": "bbb"})

    is_valid, errors, code_changes = verify_ledger(test_ledger)
    assert is_valid is True
    assert len(errors) == 0
    assert len(code_changes) == 1


def test_verify_corrupted_hash_chain(tmp_path):
    test_ledger = str(tmp_path / "corrupted_ledger.jsonl")
    
    from unittest.mock import patch
    with patch("strategy_lab.paper.forward_baseline_tracker.LEDGER_FILE", test_ledger):
        append_to_hash_chained_ledger({"entry_type": "daily_bar", "bar": 1})
        append_to_hash_chained_ledger({"entry_type": "daily_bar", "bar": 2})

    # Tamper with the file
    with open(test_ledger, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Modify data in first line
    entry0 = json.loads(lines[0])
    entry0["bar"] = 999  # Tampering with payload
    lines[0] = json.dumps(entry0) + "\n"

    with open(test_ledger, "w", encoding="utf-8") as f:
        f.writelines(lines)

    is_valid, errors, _ = verify_ledger(test_ledger)
    assert is_valid is False
    assert len(errors) > 0
