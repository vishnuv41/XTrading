# Phase 14C — Pre-Registration Specification Document (v3)

*Status*: PRE-REGISTERED EXPERIMENTAL SPECIFICATION  
*Date*: 2026-09-10  
*Purpose*: Define baselines B1–B4, candidate experiments Exp 1–4, cost model, walk-forward protocol, and rejection criteria BEFORE running Phase 14C model validation experiments.

> [!IMPORTANT]
> **Zero Phase 13 Contamination**: Phase 14C validation experiments are conducted strictly on historical development data (`BTC/USDT`, `ETH/USDT`). Phase 13 prospective trading remains 100% frozen.

---

## 1. Baselines & Candidate Experiments

### Baseline Controls (B1–B4)
* **B1 (Naive Zero)**: $\hat{r} = 0 \Rightarrow$ Always HOLD.
* **B2 (Simple Rule-Based Mean Reversion)**: `RET_1` $< -1.0\%$ or `BB_pct` $< 0.1 \rightarrow$ LONG; `RET_1` $> +1.0\%$ or `BB_pct` $> 0.9 \rightarrow$ SHORT.
* **B3 (Simple Rule-Based Momentum)**: EMA 20 $> $ EMA 50 $\rightarrow$ LONG; EMA 20 $< $ EMA 50 $\rightarrow$ SHORT.
* **B4 (Original Phase 13 ML)**: 1H continuous log return ensemble ($50/50$ XGBoost + CatBoost).

### Controlled Experiments (Exp 1–4)
* **Exp 1 (Target Normalization)**: Train model using ATR-normalized target $y_{t+h} = \frac{r_{t+h}}{\text{ATR}_t}$.
* **Exp 2 (Mean Reversion Feature Augmentation)**: Train model augmented with explicit mean-reversion features (`RET_1`, `RET_3`, `RET_5`, `BB_pct`, `MFI14`, `Williams %R`).
* **Exp 3 (Horizon Expansion)**: Evaluate models trained on $h = 2\text{h}$ and $h = 4\text{h}$ targets.
* **Exp 4 (Derivatives Augmentation)**: Evaluate models augmented with Funding Rate, Open Interest Change, and Premium Index features.

---

## 2. Cost Model & Hurdle Thresholds

* **Taker Fee**: 5 BPS per side ($10 \text{ BPS}$ roundtrip).
* **Slippage**: 5 BPS per side ($10 \text{ BPS}$ roundtrip).
* **Bid-Ask Spread**: 2 BPS roundtrip.
* **Funding Rate**: 1 BPS per 8h.
* **Total Baseline Friction**: 22–30 BPS roundtrip.
* **Pre-registered Hurdle Threshold**: $T = 0.0030$ ($30 \text{ BPS}$).

---

## 3. Pre-Registered Rejection Criteria

A candidate experiment is **REJECTED** if:
1. Net Profit Factor (PF) after costs $\le 1.15$.
2. Net PnL after costs $\le 0.00\%$.
3. Directional sign accuracy $\le 52.5\%$.
4. Performance does not beat Baselines B1–B4 in 2 or more out of 3 Walk-Forward folds.
