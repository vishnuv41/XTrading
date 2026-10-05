# Strategy Lab — Protocol Deviations & Audit Ledger

This document records all deviations from ideal research protocol, why they occurred, their date of discovery, and their exact methodological impact.

---

### Deviation 1: Combined Execution of Development & Holdout Without Intermediate Git Commit
- **Date**: 2026-10-05
- **Category**: Protocol / Pre-Registration Sequence
- **Description**: In Phase 17 Master Benchmark execution, `strategy_lab/run_phase17_benchmark.py` ran the development window evaluation, wrote `strategy_lab/PHASE17_HOLDOUT_EVALUATION_PREREGISTRATION.md`, and proceeded directly to evaluate the reserved holdout window within the same script execution before an external `git commit` was made.
- **Why It Happened**: The runner script automated the full pipeline (dev evaluation -> preregistration generation -> holdout evaluation -> markdown/json reporting) in a single run.
- **Impact Assessment**: The advancing strategy set (`Trend_EMA_20`, `Trend_EMA_50`, `TSMOM_30d`, `TSMOM_60d`) was selected strictly by programmatic formula ($Net Sharpe > 0.50$, $CI_{lower} > 0.00$, $p_{null} < 0.05$, $Sharpe_{40bps} > 0.00$) without manual discretion. However, the lack of an immutable git commit hash prior to holdout loading constitutes a procedural violation of the pre-registration protocol.
- **Remediation**: For all subsequent evaluations (including the 2018–2020 out-of-time check), pre-registration markdown files are committed to git *before* evaluation scripts are executed.

---

### Deviation 2: Advancement Filter Criteria Discrepancy
- **Date**: 2026-10-05
- **Category**: Selection Criteria
- **Description**: The programmatic advancement rule applied in the development table required $p_{\text{null}} < 0.05$ and $CI_{\text{lower}} > 0.00$, whereas the high-level specification mentioned $p_{\text{null}} < 0.10$ with plateau and per-symbol checks.
- **Audit Findings**:
  1. No candidate lies in the $0.05 \le p < 0.10$ range; the same 4 strategies qualify under either $p$-value threshold.
  2. Under multiple testing Bonferroni correction ($p_{\text{adj}} = p \times 16$), only `TSMOM_30d` ($p_{\text{adj}} = 0.048$) remains statistically significant at $\alpha = 0.05$.
  3. Paired Delta-Sharpe bootstrap CIs against Buy & Hold $[\Delta_{lo}, \Delta_{hi}]$ include zero for all four advancing strategies, confirming that risk-adjusted performance is not statistically distinguishable from Buy & Hold on the development window.
  4. Plateau verification confirms smooth decay across lookback neighborhoods without cliff-edge parameter artifacts.

---

### Deviation 3: Interface Signature Check (Branch `fix/realtime-signatures`)
- **Date**: 2026-10-05
- **Category**: Hygiene / Reporting Precision
- **Description**: Branch `fix/realtime-signatures` was created per Task T1b. Inspection of `inference/realtime_pipeline.py` against `strategy/signal.py` and `risk_engine/` revealed that all argument names and types already matched.
- **Action Taken**: `tests/test_realtime_pipeline.py` was converted from a loose script into a pytest regression test and committed to the branch. No production code changes were required.

---

### Deviation 4: Preflight Test Suite Edits & Fixture Seeding
- **Date**: 2026-10-05
- **Category**: Test Maintenance & Deterministic Fixturing
- **Specific Edits**:
  1. `tests/test_phase13_state_restoration.py`: Converted from `pytest.skip` to a deterministic mock/seeded fixture (`fake_open_trades`) ensuring 100% test execution and full attribute verification without skipping when live DB has zero active positions.
  2. `tests/test_ml_pipeline.py`: Replaced POSIX `/tmp` directory path with `tempfile.mkdtemp()` to support Windows execution.
  3. `ml/explainability/shap_analysis.py`: Added 3D ndarray handling to `get_shap_feature_importance` to support modern versions of SHAP TreeExplainer multi-class outputs.

---

### Deviation 5: Universe Survivorship Bias
- **Date**: 2026-10-05
- **Category**: Methodological Context
- **Description**: The 9-symbol universe consists of top crypto assets surviving as of 2026 (SOL, DOGE, BNB, LINK, ADA, LTC, XRP, BTC, ETH), which elevates baseline Buy & Hold annualized returns (102% in development). Absolute returns are subject to survivorship; relative metrics ($\Delta \text{Sharpe}$, relative Max Drawdown reduction) serve as the valid comparative benchmark.
