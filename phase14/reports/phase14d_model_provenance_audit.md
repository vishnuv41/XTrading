# Phase 14D Model Provenance Audit Report

*Date*: 2026-09-10  
*Auditor*: Forensic Verification Engine  
*Scope*: Complete lineage trace comparing Phase 14D research predictions against the frozen Phase 13 candidate.

---

## 1. Lineage Comparison Table

| Component | Phase 13 Frozen Candidate | Phase 14D Research Audit | Same? | Notes / Distinction |
| :--- | :--- | :--- | :---: | :--- |
| **Model Artifact** | `models_artifacts/BTCUSDT_1h`<br>`models_artifacts/ETHUSDT_1h` | `models_artifacts/BTCUSDT_1h`<br>`models_artifacts/ETHUSDT_1h` | **YES** | Identical trained ensemble binary artifacts (`ensemble.pkl`). |
| **Target Variable** | 1H Continuous Log Return | 1H Continuous Log Return | **YES** | Identical 1H forward log return target $y_{t+1} = \ln(p_{t+1}/p_t)$. |
| **Feature Stack** | 66 Causal Technical Features | 66 Causal Technical Features | **YES** | Identical 66 feature columns (`feature_columns.pkl`). |
| **Forecast Horizon** | 1 Bar ($1 \text{ hour}$) | 1 Bar ($1 \text{ hour}$) | **YES** | Identical 1H forecast horizon. |
| **Training Data** | Historical 1H Data ($\le 2026\text{-}06$) | Historical 1H Data ($\le 2026\text{-}06$) | **YES** | Identical training history window. |
| **Ensemble Architecture**| 50/50 XGBoost + CatBoost Blend | 50/50 XGBoost + CatBoost Blend | **YES** | Identical base models and blending weights. |
| **Prediction Evaluation**| Live 250-prediction rolling window | All 2,000 historical bars | **NO** | Phase 13 ranks predictions live; Phase 14D evaluates full historical distribution. |
| **Ranking Procedure** | Rolling Percentile Rank ($\text{Rank} \ge 0.9900$) | Static Quantile Bins ($Q_1 \dots Q_5$) | **NO** | Phase 13 uses live rolling Top-1% gate; Phase 14D evaluates static quantile bins. |
| **Execution Gate** | LONG-ONLY ($\text{Rank} \ge 0.9900 \rightarrow \text{LONG}$) | Diagnostic Bins ($Q_1 \dots Q_5$ & $P_{90}\dots P_{99}$) | **NO** | Phase 13 executes prospective trades; Phase 14D evaluates diagnostic returns. |

---

## 2. Explicit Provenance Answers

1. **Are Phase 14D predictions generated from the exact same model artifact as Phase 13?**  
   **YES**. Phase 14D loaded `models_artifacts/BTCUSDT_1h` and `models_artifacts/ETHUSDT_1h` via `ml/predict.py` (the identical model artifacts used in Phase 13).
2. **What differs?**  
   The **prediction evaluation scope and ranking procedure**. Phase 13 ranks predictions dynamically over a rolling 250-prediction window to enforce a live Top-1% gate. Phase 14D evaluated the model's predictions across all 2,000 historical bars divided into static quantile bins ($Q_1 \dots Q_5$).
3. **Configuration Classification**:  
   Phase 14D evaluated an **un-gated diagnostic ranking configuration of the exact frozen Phase 13 model**.
4. **Generalization Constraint**:  
   The finding *"No monotonic ordinal ranking power was demonstrated in this audit"* applies to the historical 5-bin quantile distribution of predictions. It confirms that raw prediction values $\hat{r}$ carry minimal monotonic ordinal ranking power across all bars, but does NOT alter the frozen Phase 13 candidate or its ongoing prospective validation loop.
