# PHASE 15 — RESEARCH BRANCH BLUEPRINT, PROVENANCE MANIFEST & MINI-LLM BASELINE BENCHMARKING REPORT

> **Official Classification**: **Phase 15C-v1 Mini-LLM Baseline Trained & Benchmarked against Baselines; Untouched Test Set Preserved.**
>
> **Model Status**: **`FinancialMiniLLM-35M-Baseline-v1` (34.6M Parameters) trained on frozen Dataset v1 (`ec4491e01bd7`) and benchmarked on chronological validation era (Achieved Highest Recall: 63.83% & Highest F1: 0.5636).**

---

## 1. Dual-Track Operational Boundary

```text
XTrading System Architecture
│
├── PHASE 13 (FROZEN PROSPECTIVE CONTROL)
│   ├── Model           : Aug 07–10 3-Model Stacking Classifier (XGB + LGBM + CatBoost)
│   ├── Signal Gate     : Top 1.0% (≥99.0th percentile rank)
│   ├── Accounting      : PostgreSQL Shared Portfolio ($10,000)
│   ├── Live State      : N = 5 / 30 Valid | P&L -$45.41 | PF 0.82 | Equity $9,954.59
│   └── Status          : RUNNING AUTONOMOUSLY (N ≥ 6) — ZERO TOUCH
│
└── PHASE 15 (OFFLINE PARALLEL RESEARCH BRANCH)
    ├── Level 1 (Software Correctness) : ✅ Implemented & 39/39 Tests Passing
    ├── Level 2 (Dataset Integrity)    : ✅ Dataset v1 Frozen (`ec4491e01bd7`) & 10-Point Gate Passed
    ├── Shadow Context Reader           : ✅ `phase15/model/shadow_llm_analyst.py` Ingests History & News
    ├── Level 3 (Baseline 15C Benchmarking): ✅ `FinancialMiniLLM-35M-Baseline-v1` Benchmarked vs 4 Baselines
    └── Status                         : FROZEN BASELINE V1 (Zero Impact on Phase 13)
```

---

## 2. Phase 15C Experiment B Baseline Comparison Matrix (`phase15/model/benchmark_baselines.py`)

All models were evaluated strictly on the chronological validation era (6,655 training samples, 1,427 validation samples):

| Model Name | Val Accuracy | Val Bal Acc | Val Precision | Val Recall | Val F1 Score | Brier Score | Log Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Classifier** | 50.60% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.4940 | 17.8071 |
| **Logistic Regression** | **54.03%** | 53.61% | **61.61%** | 18.44% | 0.2838 | 0.2478 | 0.6889 |
| **Multi-Layer Perceptron (MLP)** | 50.88% | 50.93% | 50.26% | 55.74% | 0.5286 | 0.4888 | 16.6103 |
| **CatBoost Meta-Filter** | 53.89% | **53.84%** | 53.61% | 49.50% | 0.5147 | **0.2466** | **0.6865** |
| **FinancialMiniLLM (35M Transformer)** | 51.16% | 51.31% | 50.45% | **63.83%** | **0.5636** | 0.2623 | 0.7194 |

### Key Benchmark Insights:
1. **Highest Recall & F1 Score**: `FinancialMiniLLM (35M Transformer)` achieved **highest Recall (63.83%)** and **highest F1 Score (0.5636)**.
2. **Lowest Brier & Log Loss**: `CatBoost Meta-Filter` achieved lowest Brier score (`0.2466`) and lowest Log Loss (`0.6865`).
3. **Highest Precision**: `Logistic Regression` achieved highest precision (`61.61%`) at low recall (`18.44%`).
4. Recorded in [baseline_comparison_matrix.json](file:///d:/all/XTrading_combined%20%281%29/XTrading/phase15/artifacts/baseline_comparison_matrix.json).

---

## 3. Canonical Provenance Registry (`phase15/provenance/`)

1. `dataset_manifest.json`: Source data hashes and zero-prospective-leakage rules.
2. `dataset_v1_manifest.json`: Frozen Dataset v1 SHA-256 fingerprint (`ec4491e01bd7`) and class breakdown (46.61% ACCEPT / 53.39% REJECT).
3. `feature_manifest.json`: 115 causal feature definitions ($t_{\text{feature}} \le T_{\text{candle\_close}}$).
4. `tokenizer_manifest.json`: Discretized token vocabulary mapping (`vocab_size=2048`).
5. `split_manifest.json`: Chronological partition boundaries (Train `2017-2023`, Val `2024-2025`, Test `2026+`).
6. `model_provenance.json`: Recorded 34.6M Mini-LLM baseline weights (`FinancialMiniLLM-35M-Baseline-v1`) and validation metrics.
7. `data_integrity_gate_report.json`: Automated 10-point data integrity audit report (`passed_all_checks = true`).

---

## 4. System Status Summary

```text
Phase 13 Prospection : RUNNING AUTONOMOUSLY (N ≥ 6)
Phase 13 Ledger      : N = 5 / 30 Valid | P&L -$45.41 | PF 0.82 | Equity $9,954.59
Phase 13 Model      : FROZEN (Aug 07–10, 2026 Binaries — BTC SHA256 2efaaefb65e2, ETH SHA256 4c864a5eed22)
Phase 15 Dataset v1 : FROZEN & AUDITED (SHA256 ec4491e01bd7 | 46.61% ACCEPT / 53.39% REJECT)
Phase 15C Baseline  : TRAINED & BENCHMARKED (`FinancialMiniLLM-35M-Baseline-v1` | Highest F1 0.5636 & Recall 63.83%)
Shadow LLM Analyst  : BUILT & VERIFIED (`phase15/model/shadow_llm_analyst.py` — Non-Interfering)
Test Suite Status   : 39 / 39 PASSED
```
