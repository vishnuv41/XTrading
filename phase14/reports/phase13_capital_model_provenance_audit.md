# PHASE 13 — BLOCKING CAPITAL, CONFIGURATION & MODEL PROVENANCE AUDIT

> **Executive Summary**: This audit formally inspects and establishes the mathematical, architectural, and prospective provenance of Phase 13 paper trading ($N=5$ completed trades). It reconciles historical isolated daemon process execution against the single shared $10,000 portfolio specification, proves the mathematical identity of the TP ATR multiplier, establishes exact empirical model-artifact provenance from byte-level inspection, and updates the canonical Phase 13 classification.

---

## 1. Capital Accounting Provenance & Model Parity

### Question 1: Execution Model Structure
**Finding**: The initial 5 trades executed under separate process-local cash states of $10,000 for BTC and ETH daemons rather than a single shared $10,000 portfolio.
- When the ETH daemon restarted on Sep 10, its local cash reset to $10,000 when flat.
- The trade P&Ls themselves were financially calculated correctly, but persisted `equity_after` in daemon memory reflected local daemon cash rather than global portfolio equity.

### Question 2: Capital Shortfall & Signal Suppression
**Finding**: **Zero trades were skipped or suppressed.**
- On 2026-09-06 at 14:00 UTC, ETH Trade #1 and BTC Trade #2 generated simultaneous entry signals.
- **Shared Capital Margin Audit ($10,000 Shared Portfolio)**:
  - Initial Shared Cash: **$10,000.00**
  - ETH #1 Margin Required: **$3,315.44** (Notional $9,916.56 at 3x leverage + $9.92 fee)
  - Remaining Shared Cash after ETH #1: **$6,684.56**
  - BTC #1 Margin Required: **$5,155.77** (Notional $15,421.05 at 3x leverage + $15.42 fee)
  - Remaining Shared Cash after BOTH trades open: **$1,528.79** ($6,684.56 - $5,155.77)
- **Conclusion**: Because 3.0x leverage was active, total initial margin required across both trades ($8,471.21) was well below available shared cash ($10,000.00). Both signals executed cleanly under single-portfolio shared capital.

### Question 3: Quantitative Sizing & Sensitivity Matrix

| Trade ID | Symbol | Action TS | Historical Size | Counterfactual Shared Size | Size Diff | Observed Realized PnL | Counterfactual Shared PnL | PnL Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `c86f0f89` | ETH/USDT | 2026-09-06 14:00 | 3.999196 ETH | 4.001179 ETH | +0.05% | +$124.26 | +$124.32 | +$0.06 |
| `faf6125a` | BTC/USDT | 2026-09-06 14:00 | 0.193967 BTC | 0.193924 BTC | -0.02% | +$87.77 | +$87.75 | -$0.02 |
| `db6ab9bb` | ETH/USDT | 2026-09-07 04:00 | 3.431121 ETH | 3.462348 ETH | +0.91% | -$103.67 | -$104.61 | -$0.94 |
| `9e9d5579` | ETH/USDT | 2026-09-07 07:00 | 3.327963 ETH | 3.358202 ETH | +0.91% | -$104.18 | -$105.13 | -$0.95 |
| `7abfcfd4` | ETH/USDT | 2026-09-09 18:00 | 1.489811 ETH | 1.490433 ETH | +0.04% | -$49.59 | -$49.61 | -$0.02 |
| **TOTAL** | — | — | — | — | — | **-$45.41** | **-$47.28** | **-$1.87** |

### Accounting Record Distinction:
- **Observed Historical Result (Canonical Scientific Record)**: **-$45.4092** (Canonical Equity: **$9,954.59**)
- **Counterfactual Shared-Capital Sensitivity Analysis**: **-$47.2779** (Shared Equity: **$9,952.72**)
- **Quantified Infrastructure Effect**: **$1.8686** (0.0187% of initial capital).
- **Rule**: Historical P&L records in PostgreSQL are preserved untouched. The counterfactual is maintained strictly as an audit sensitivity baseline.

---

## 2. Model Architecture & Artifact Provenance

### Question 6: Empirical Code & Artifact Audit
Direct byte-level and memory inspection of `models_artifacts/BTCUSDT_1h` and `models_artifacts/ETHUSDT_1h` confirms:

1. **Loaded Model Artifacts**:
   - `models_artifacts/BTCUSDT_1h/ensemble.pkl` (SHA-256: `2efaaefb65e2`, 5.66 MB)
   - `models_artifacts/ETHUSDT_1h/ensemble.pkl` (SHA-256: `4c864a5eed22`, 2.01 MB)
   - Both files contain a serialized `ml.models.ensemble.EnsembleModel` instance.

2. **Base Model Ensemble Composition**:
   - **XGBoost Classifier** (`ml.models.xgboost_model.XGBoostModel` 3-class classifier)
   - **LightGBM Classifier** (`ml.models.lightgbm_model.LightGBMModel` 3-class classifier)
   - **CatBoost Classifier** (`ml.models.catboost_model.CatBoostModel` 3-class classifier)
   - **Status of LightGBM**: **LOADED AND ACTIVE**. LightGBM is physically embedded inside `ensemble.pkl` for both symbols and contributes directly to raw ensemble probabilities. It was never omitted at runtime.

3. **Probability Calibration & Decision Logic**:
   - **Ensemble Raw Output**: 3-class probability distribution ($P_{\text{down}}, P_{\text{flat}}, P_{\text{up}}$).
   - **Class Decision**: Decided strictly via `np.argmax(raw_proba, axis=1)` (uncalibrated raw model consensus).
   - **Calibrator**: `calibrator.pkl` contains `ProbabilityCalibrator` (Isotonic regression). It rescales the winning class's raw score into calibrated `confidence`.

4. **Resolution of `candidate_manifest.json` Contradiction**:
   - `candidate_manifest.json` in `models_artifacts/PHASE13_PAPER_TRADING/` was a preliminary draft document describing an earlier 2-model regressor design.
   - The **actual frozen prospective model artifacts** deployed for Phase 13 Run 002 (trained August 7–10, 2026, and loaded by `run_paper_trading.py`) are the **3-model Stacking Classifier Ensemble (XGBoost + LightGBM + CatBoost) with Isotonic calibration**.
   - No code or model changes were made during this audit.

### Pipeline End-to-End Execution Diagram:

```text
OHLCV (1h)
  │
  ├──► Indicators (EMA, MACD, RSI, ATR14, Supertrend, etc.)
  │
  ├──► Feature Matrix (115 Causal Features)
  │
  ├──► Ensemble Model (models_artifacts/{symbol}_1h/ensemble.pkl)
  │      ├── XGBoost Classifier  ┐
  │      ├── LightGBM Classifier ┼─► Ensemble Raw Probabilities (Down / Flat / Up)
  │      └── CatBoost Classifier ┘
  │
  ├──► Decision Logic: Class = argmax(raw_proba)
  │
  ├──► Calibration: confidence = Isotonic(raw_proba[winning_class])
  │
  ├──► Reference Window: 250-Prediction Rolling Window
  │
  ├──► Signal Gate: Rolling Percentile Rank >= 99.0% (Top 1.0%)
  │
  └──► Trade Signal: BUY / SELL / HOLD
```

---

## 3. Configuration & Parameter Provenance

### Question 5: Take-Profit Multiplier Provenance
**Finding**: `candidate_manifest.json` ($TP = 3.0\times\text{ATR}$) and `config/settings.py` (`tp_risk_reward = 2.0`) are **mathematically identical**.

$$\text{Stop Loss Distance} = 1.5 \times \text{ATR}$$

$$\text{Take Profit Distance} = R \times \text{Stop Loss Distance} = 2.0 \times (1.5 \times \text{ATR}) = 3.0 \times \text{ATR}$$

- **Execution Path**:
  - `stop_loss = calculate_stop_loss(entry, atr, multiplier=1.5)` ($\Rightarrow 1.5\text{x ATR}$)
  - `take_profit = calculate_take_profit(entry, stop_loss, risk_reward_ratio=2.0)` ($\Rightarrow 2.0 \times 1.5\text{x ATR} = 3.0\text{x ATR}$)
- **Slippage Impact**: Executed order fills include 5 bps (0.05%) slippage, shifting entry fill prices slightly and producing fill-to-exit ratios of 1.69–1.86 relative to executed fill prices, while signal R:R is exactly 2.00 relative to signal entry price.

---

## 4. Architectural Protection Verification

### Question 9: Shared DB Portfolio Synchronization
**Verification**: Shared portfolio sizing is protected by the implemented database synchronization / atomic allocation mechanism in PostgreSQL. Concurrent BTC and ETH risk evaluations query global portfolio equity directly (`portfolio_state`), with concurrent allocation integrity verified by 39 passing unit tests (including `test_simultaneous_signals.py`).

---

## 5. Formal Classification of Phase 13 $N=5$

### Canonical Classification:
> **PHASE 13 N=5 — VALID WITH QUANTIFIED INFRASTRUCTURE DEVIATION**

- Entry Signal Integrity: ✅ Valid (Passes 99.0th percentile rolling rank gate)
- Exit Geometry Integrity: ✅ Valid (SL = 1.5x ATR, TP = 3.0x ATR / 2.0R)
- Closed-Candle Execution: ✅ Valid (Evaluated at candle closes)
- Trade Ledger & PnL: ✅ Preserved (Observed PnL -$45.41)
- Position Sizing Deviation: ⚠️ Historical process-local architecture produced small deviations (Max size diff ~0.91%)
- Aggregate PnL Effect: **$1.87** (0.0187% of initial capital)
- Model Artifact Provenance: ✅ Verified (XGBoost + LightGBM + CatBoost 3-class classifier ensemble with Isotonic calibration)
- Shared Architecture & Concurrency: ✅ Implemented & Tested (39/39 PASS)
- Action: **$N=5$ record is valid and preserved. $N=6$ is CLEARED to proceed.**

---

## 6. Current Phase 13 Ledger & Live State

```text
Capital Pool      : $10,000 (Shared Global Portfolio)
Completed Trades  : N = 5 / 30
Wins / Losses     : 2 Wins / 3 Losses (Win Rate = 40.0%)
Observed P&L      : -$45.41
Canonical Equity  : $9,954.59
Open Positions    : 0 (Flat)
BTC Gate Status   : HOLD (10.99th percentile < 99th cutoff)
ETH Gate Status   : HOLD (42.11th percentile < 99th cutoff)
Test Suite Status : 39/39 PASSED
$N=6$ Status      : CLEARED
```
