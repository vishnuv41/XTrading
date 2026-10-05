"""
tests/test_forward_tracker.py
-----------------------------
Unit tests for isolated forward paper tracker and SHA-256 hash chaining.
"""

import os
import json
import pytest
import pandas as pd
from unittest.mock import patch

from strategy_lab.paper.forward_baseline_tracker import (
    initialize_or_load_state,
    save_state,
    get_last_ledger_hash,
    append_to_hash_chained_ledger,
    step_daily_tracker,
)


def test_state_initialization(tmp_path):
    temp_state_file = str(tmp_path / "test_state.json")
    with patch("strategy_lab.paper.forward_baseline_tracker.STATE_FILE", temp_state_file):
        state = initialize_or_load_state()
        assert "arms" in state
        assert "Trend_EMA_50" in state["arms"]
        assert "Buy_and_Hold" in state["arms"]
        assert "Static_Exposure_45pct" in state["arms"]
        assert state["arms"]["Trend_EMA_50"]["current_equity"] == 10000.0


def test_hash_chaining(tmp_path):
    temp_ledger = str(tmp_path / "test_ledger.jsonl")
    with patch("strategy_lab.paper.forward_baseline_tracker.LEDGER_FILE", temp_ledger):
        # 1. First entry (genesis)
        h0 = get_last_ledger_hash()
        assert h0 == "0" * 64

        rec1 = {"timestamp": "2026-10-06T00:00:00Z", "val": 100}
        append_to_hash_chained_ledger(rec1)
        
        # 2. Second entry (chained)
        h1 = get_last_ledger_hash()
        assert h1 != "0" * 64
        assert len(h1) == 64

        rec2 = {"timestamp": "2026-10-07T00:00:00Z", "val": 200}
        append_to_hash_chained_ledger(rec2)
        h2 = get_last_ledger_hash()
        assert h2 != h1
        assert len(h2) == 64
