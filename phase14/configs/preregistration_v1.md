# Phase 14 — Pre-Registration Specification Document (v1)

*Status*: PRE-REGISTERED HYPOTHESIS & PROTOCOL  
*Date*: 2026-09-10  
*Purpose*: Define evaluation metrics, forecast horizon, candidate hurdle thresholds $T$, cost assumptions, walk-forward folds, and rejection criteria BEFORE running model deconstruction experiments.

> [!IMPORTANT]
> **Zero Phase 13 Contamination**: Phase 14 deconstruction experiments are conducted strictly on historical development data (`BTC/USDT`, `ETH/USDT` 1H OHLCV). Phase 13 prospective data is completely excluded.

---

## 1. Learning Task & Target Definition

* **Target Variable ($y_{t+1}$)**: Log return of next 1-hour candle:
  $$y_{t+1} = \ln\left(\frac{\text{Close}_{t+1}}{\text{Close}_t}\right)$$
* **Forecast Horizon**: 1 bar ($1 \text{ hour}$).
* **Symbol Universe**: `BTC/USDT`, `ETH/USDT`.
* **Timeframe**: `1h`.

---

## 2. Pre-Registered Hurdle Threshold Grid ($T$)

To evaluate predictive performance without post-hoc threshold selection, the following pre-registered threshold grid $T$ is specified:

$$T \in \{0.0010 \text{ (10 BPS)}, 0.0020 \text{ (20 BPS)}, 0.0030 \text{ (30 BPS)}, 0.0040 \text{ (40 BPS)}, 0.0050 \text{ (50 BPS)}\}$$

---

## 3. Pre-Registered Cost & Friction Schedule

Costs are modeled explicitly layer-by-layer:
* **Exchange Taker Fee**: $0.0005$ ($5 \text{ BPS}$) per side $\rightarrow 10 \text{ BPS}$ roundtrip.
* **Slippage**: $0.0005$ ($5 \text{ BPS}$) per side $\rightarrow 10 \text{ BPS}$ roundtrip.
* **Bid-Ask Spread**: $0.0002$ ($2 \text{ BPS}$) roundtrip.
* **Perpetual Funding Rate**: $0.0001$ ($1 \text{ BPS}$) per $8 \text{ hours}$.
* **Total Baseline Roundtrip Friction ($F_{\text{roundtrip}}$)**: $0.0022$ to $0.0030$ ($22\text{--}30 \text{ BPS}$).

---

## 4. Evaluation Metrics & Protocol

1. **Experiment A (Long Tail)**: $E[R_{\text{actual}} \mid \hat{r} \ge +T]$ (N, mean return, win rate, net return %, 95% CI, bootstrap loss probability).
2. **Experiment B (Short Tail)**: $E[-R_{\text{actual}} \mid \hat{r} \le -T]$ (N, mean short return, win rate, net short return %, 95% CI, bootstrap loss probability).
3. **Experiment C (Magnitude Correlation)**: Pearson $r$ & Spearman $\rho$ of $|\hat{r}|$ vs. $|r_{\text{actual}}|$.
4. **Experiment D (Directional Accuracy)**: Directional sign accuracy $\text{sign}(\hat{r}) == \text{sign}(r_{\text{actual}})$, confusion matrix, balanced accuracy.
5. **Walk-Forward Validation**: 3 non-overlapping chronological folds.

---

## 5. Pre-Registered Rejection Criteria

A tail signal (Long or Short) is **REJECTED** as a tradable edge if:
1. Sample size $N < 30$ across the historical dataset.
2. Net expected return after friction $\le 0.0000$.
3. Bootstrap loss probability $> 25\%$.
4. Directional accuracy $\le 52.0\%$.
5. Performance fails in 2 or more out of 3 Walk-Forward Out-of-Sample folds.
