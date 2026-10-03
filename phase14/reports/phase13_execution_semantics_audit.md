# Phase 13 Execution Semantics Audit Report

**Audit Date:** 2026-09-10  
**Status:** `AUDIT COMPLETE — N=5 VALID`  
**Classification:** `VALID_TOP1_EXECUTION` + `TELEMETRY_DISPLAY_MISLABELING`  
**Subject:** Forensic investigation into stored confidence values (`0.559417`, `0.642599`) for the 5 executed Phase 13 trades and reconciliation with the Top-1% Gating Protocol.

---

## Executive Summary

A comprehensive, 9-task forensic audit was conducted on the execution pipeline, database schema, model prediction logs, and dashboard telemetry.

### Key Audit Finding
The apparent inconsistency—where executed trades appeared to have "percentiles" of **55.94%** and **64.26%** despite a Top-1% cutoff (99.00%)—is **100% resolved as a Telemetry Display Mislabeling bug in `dashboard.py`**.

1. **What is stored in `prediction_log.confidence`?**  
   The `confidence` column stores the **raw/calibrated model class probability** (e.g. `0.642599` = 64.26% winning-class probability), **NOT** the rolling percentile rank.
2. **How was the Top-1% gate actually evaluated during execution?**  
   In `inference/realtime_pipeline.py`, the Top-1% cutoff is calculated over a rolling 250-bar window of model predictions:
   $$\text{top\_k\_cutoff} = \text{np.partition}(\text{pred\_series}, -k)[-k]$$
   For all 5 executed trades, the model's confidence probability was the **highest score in the 250-bar window**, corresponding to a **100.00th percentile rank** (`gate_pass = True`).
3. **What caused the dashboard error?**  
   `experiments/dashboard.py` took the raw probability confidence `p[confidence]` (e.g. `0.642599`) and multiplied it by 100 (`0.642599 * 100.0 = 64.26%`), mistakenly rendering raw probability as if it were a percentile rank score.

---

## Reconstruction Ledger of 5 Executed Trades

| Trade # | Symbol | Entry TS | Stored Prob (`confidence`) | 250-Bar Cutoff | Reconstructed Window Rank | Top-1% Gate Pass | Execution Status | Audit Classification |
|---|---|---|---|---|---|---|---|---|
| **1** | ETH/USDT | 2026-09-06 14:00 | **0.642599** (64.26% prob) | 0.642599 | **100.00%** | **`True` (PASS)** | EXECUTED | `VALID_TOP1_EXECUTION` |
| **2** | BTC/USDT | 2026-09-06 14:00 | **0.559417** (55.94% prob) | 0.559417 | **100.00%** | **`True` (PASS)** | EXECUTED | `VALID_TOP1_EXECUTION` |
| **3** | ETH/USDT | 2026-09-07 04:00 | **0.642599** (64.26% prob) | 0.642599 | **100.00%** | **`True` (PASS)** | EXECUTED | `VALID_TOP1_EXECUTION` |
| **4** | ETH/USDT | 2026-09-07 07:00 | **0.660701** (66.07% prob) | 0.660701 | **100.00%** | **`True` (PASS)** | EXECUTED | `VALID_TOP1_EXECUTION` |
| **5** | ETH/USDT | 2026-09-09 18:00 | **0.642599** (64.26% prob) | 0.642599 | **100.00%** | **`True` (PASS)** | EXECUTED | `VALID_TOP1_EXECUTION` |

---

## Detailed Task Breakdown

### Task 1 & 3 — Gate Calculation & Rank Reconstruction
For each of the 5 trades, the 250-bar rolling inference window preceding the trade timestamp was loaded and passed through `ml.predict.predict()`. 
* In every single case, the trade was triggered because the model prediction achieved the **maximum confidence score** in that 250-bar window ($100.00^{\text{th}}$ percentile).
* The Top-1% gate (`gate_pass = True`) operated strictly as pre-registered.

### Task 2 — Distinguish Confidence vs Percentile Rank
* **`confidence` (stored in `prediction_log`):** Calibrated probability score output by `ml.predict.predict()` for the winning class (range $[0.0, 1.0]$).
* **Percentile Rank:** Calculated in memory inside `inference/realtime_pipeline.py` by ranking `confidence` against recent rolling prediction history:
  $$\text{percentile} = \frac{\sum (\text{pred\_series} \le \text{confidence})}{N} \times 100\%$$
* **Conclusion:** `prediction_log` stores model probability confidence, not percentile rank.

### Task 4 & 5 — Provenance & Configuration Run Verification
* All 5 trades were created after `2026-09-06 14:00:00+00` (`created_at` timestamps between Sep 6 and Sep 10).
* None of the 5 trades belong to `PHASE13_RUN_001` (which was previously invalidated).
* All 5 trades belong 100% to the official, pre-registered **Phase 13 Candidate Run 002**.

### Task 6 — Repeated Prediction Values Analysis
* The prediction confidence for ETH frequently evaluates to exact numbers like `0.642599` or `0.545788`.
* **Root Cause:** The probability calibrator uses **Isotonic Regression** (`ProbabilityCalibrator`), which is a non-decreasing step function. Isotonic regression maps continuous raw model outputs into discrete calibrated probability steps. Different feature vectors falling within the same step interval output identical calibrated probabilities. This is standard and expected behavior for Isotonic calibration.

### Task 7 — Closed-Candle Verification
* All 5 trades were evaluated at exact closed 1H bar boundaries (e.g. 14:00 candle evaluated at 15:00:00 UTC).
* No partial or forming candles entered the feature pipeline.

---

## Final Decision & Impact on $N=5$

### 1. Classification
* **Execution Engine:** `VALID_TOP1_EXECUTION`
* **Telemetry Display:** `LOGGING/TELEMETRY_BUG`

### 2. Impact on Prospective Sample ($N=5$)
* **`N=5 VALID`**
* The 5 prospective trades ($N=5/30$, 2W / 3L, $-\$45.41$, PF `0.82`) are **100% valid** members of the frozen Phase 13 prospective evaluation sample.
* Zero strategy rules, thresholds, models, or database records require modification.

---

## Required Corrective Action

### Dashboard Telemetry Fix
Update `experiments/dashboard.py` so that:
1. It explicitly labels model prediction confidence as **Model Win Probability** (e.g. `64.26% Prob`).
2. When displaying Top-1% Gate status, it formats the rolling rank accurately (e.g. `100.00% Rank [PASS]`).
