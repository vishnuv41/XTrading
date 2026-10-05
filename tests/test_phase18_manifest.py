"""
tests/test_phase18_manifest.py
--------------------------------
Unit and protocol tests for Phase 18 manifest integrity, boundary enforcement, and firewall checks.
"""

import pytest
from strategy_lab.phase18.manifest_validator import (
    load_manifest,
    compute_manifest_hash,
    validate_manifest_schema,
    check_dataset_boundaries,
)


def test_phase18_manifest_schema():
    manifest = load_manifest()
    assert validate_manifest_schema(manifest) is True
    assert len(manifest["universe"]) == 9
    assert manifest["reproducibility"]["random_seed"] == 42


def test_phase18_manifest_hash_stability():
    h1 = compute_manifest_hash()
    h2 = compute_manifest_hash()
    assert len(h1) == 64
    assert h1 == h2


def test_phase18_firewall_boundary_enforcement():
    # Valid date within holdout
    assert check_dataset_boundaries("2020-01-01", "2026-09-30 23:59:59+00:00") is True

    # Firewall violation: attempting to query beyond frozen holdout
    with pytest.raises(ValueError, match="FIREWALL VIOLATION"):
        check_dataset_boundaries("2020-01-01", "2026-10-06 00:00:00+00:00")
