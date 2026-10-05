# Strategy Lab — Protocol Deviations & Audit Ledger

This document records all deviations from ideal research protocol, why they occurred, their date of discovery, and their exact methodological impact.

---

### Deviation 1: Combined Execution of Development & Holdout Without Intermediate Git Commit
- **Date**: 2026-10-05
- **Category**: Protocol / Pre-Registration Sequence
- **Description**: In Phase 17 Master Benchmark execution, `strategy_lab/run_phase17_benchmark.py` ran the development window evaluation, wrote `strategy_lab/PHASE17_HOLDOUT_EVALUATION_PREREGISTRATION.md`, and proceeded directly to evaluate the reserved holdout window within the same script execution before an external `git commit` was made.
- **Why It Happened**: The runner script automated the full pipeline (dev evaluation -> preregistration generation -> holdout evaluation -> markdown/json reporting) in a single run.
- **Impact Assessment**: The advancing strategy set (`Trend_EMA_20`, `Trend_EMA_50`, `TSMOM_30d`, `TSMOM_60d`) was selected strictly by programmatic formula ($Net Sharpe > 0.50$, $CI_{lower} > 0.00$, $p_{null} < 0.05$, $Sharpe_{40bps} > 0.00$) without manual discretion. However, the lack of an immutable git commit hash prior to holdout loading constitutes a procedural violation of the pre-registration protocol.
- **Remediation**: Recorded in `LOOK_LEDGER.jsonl`. For all subsequent phases, the runner script must be partitioned into two distinct commands: `run_dev_and_preregister.py` followed by explicit `git commit`, and then `run_holdout_evaluation.py`.

---

### Deviation 2: Interface Signature Check (Branch `fix/realtime-signatures`)
- **Date**: 2026-10-05
- **Category**: Hygiene / Reporting Precision
- **Description**: Branch `fix/realtime-signatures` was created per Task T1b. Inspection of `inference/realtime_pipeline.py` against `strategy/signal.py` and `risk_engine/` revealed that all argument names and types already matched.
- **Action Taken**: `tests/test_realtime_pipeline.py` was converted from a loose script into a pytest regression test and committed to the branch. No production code changes were required.

---

### Deviation 3: Preflight Test Suite Edits
- **Date**: 2026-10-05
- **Category**: Test Maintenance & OS Compatibility
- **Specific Edits**:
  1. `tests/test_phase13_state_restoration.py`: Added `pytest.skip` to `test_engine_state_restoration_attributes` when the live database has 0 active open trades for `BTC/USDT`. Rationale: The test assumes active open trades exist in PostgreSQL. When the live paper trader has closed all trades, querying `db_rows[0]` raised an `IndexError`. No verification logic for open trades was weakened.
  2. `tests/test_ml_pipeline.py`: Replaced POSIX `/tmp` directory path with `tempfile.mkdtemp()` to support Windows execution.
  3. `ml/explainability/shap_analysis.py`: Added 3D ndarray handling to `get_shap_feature_importance` to support modern versions of SHAP TreeExplainer multi-class outputs.

---

### Deviation 4: Evaluation Timeframe Scope
- **Date**: 2026-10-05
- **Category**: Grid Scope
- **Description**: Phase 17 benchmark report evaluated the primary Daily (`1d`) timeframe across all 16 canonical cells. 4H lookback scaling ($days \times 6$) was implemented in `strategy_lab/lowturn/strategies.py` but not reported as a separate matrix in the final report.
