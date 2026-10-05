"""
tests/test_forward_tracker.py
-----------------------------
Unit tests for isolated forward paper tracker.
"""

import os
import json
import pytest
import pandas as pd
from unittest.mock import patch

from strategy_lab.paper.forward_baseline_tracker import (
    initialize_or_load_state,
    save_state,
    step_daily_tracker,
    STATE_FILE,
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
