# Phase 13 Serial Dependence & N=30 Evaluation Protocol

*Status*: PRE-REGISTERED STATISTICAL PROTOCOL FOR $N \ge 30$  
*Date*: 2026-09-10  
*Purpose*: Define statistical dependence treatment, block-bootstrapping rules, and trade clustering metrics BEFORE Phase 13 reaches $N=30$.

> [!IMPORTANT]
> **Immutable Gate Protection**: The original preregistered Phase 13 gate criteria ($N \ge 30$, PF $\ge 1.30$, Sharpe $\ge 1.00$, Win Rate $\ge 60\%$, Bootstrap Loss Probability $< 15\%$) remain **100% frozen**. Dependence-aware block-bootstrapping is added as an explicit robustness diagnostic, NOT a post-hoc replacement.

---

## 1. Trade Clustering & Temporal Dependence Diagnostics

When Phase 13 reaches $N \ge 30$, the following temporal clustering diagnostics will be reported alongside realized PnL:
1. **Trade Interval Metrics**: Mean gap, median gap, minimum gap, maximum gap (in hours / bars).
2. **Consecutive Signal Clusters**: Count of trades occurring within $\le 48 \text{ hours}$ of previous trade close.
3. **Simultaneous Exposure**: Count of concurrent open trades across `BTC/USDT` and `ETH/USDT`.

---

## 2. Pre-Registered $N=30$ Statistical Protocol

```text
                  PHASE 13 EVALUATION PROTOCOL (N ≥ 30)
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────────────────────┐                         ┌──────────────────────────────┐
│       PRIMARY METRIC         │                         │    ROBUSTNESS DIAGNOSTIC     │
├──────────────────────────────┤                         ├──────────────────────────────┤
│ • Preregistered IID          │                         │ • 5-Trade Block Bootstrap    │
│   Bootstrap (N=1000, seed=42)│                         │   (resamples 5-trade blocks) │
│ • Target: Loss Prob < 15%    │                         │ • Cluster-adjusted CI bounds │
└──────────────────────────────┘                         └──────────────────────────────┘
```

### Protocol Specifications
* **Primary Metric**: Preregistered IID trade bootstrap ($N=1000$ resamples, seed=42).
* **Robustness Metric**: **5-Trade Block-Bootstrap** ($N=1000$ resamples, seed=42). Resamples overlapping 5-trade chronological blocks to preserve serial dependence.
* **Overlapping / Clustered Trades**: Trades occurring in identical market trends are block-resampled together.
* **Interpretation Rule**:
  > *"The Phase 13 gate decision will be evaluated on the primary preregistered IID metric. If the primary metric passes but the block-bootstrap robustness metric fails (Loss Prob $\ge 15\%$), the result will be flagged as 'PASS WITH TEMPORAL CLUSTERING RISK' requiring an additional 15 prospective trades before capital allocation."*
