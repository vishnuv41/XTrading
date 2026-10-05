# Phase 19 Research Pre-Registration: Causal Market Regime Classifier

**Protocol Version**: 1.0.0  
**Created Timestamp**: 2026-10-05T12:50:00Z  
**Research Boundary**: Frozen Historical Development Window (2020-08-11 to 2024-12-31 UTC, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `XRP/USDT`, `ADA/USDT`, `LTC/USDT`, `SOL/USDT`, `DOGE/USDT`, `LINK/USDT`)  
**Comparator Baseline**: `Trend_EMA_50` Equal-Weighted (Unfiltered)  
**Isolation Rule**: Phase 19 operates in strict isolation. No feedback or data flow from the Phase 17 forward tracker is permitted.

---

## 1. Core Economic Hypothesis

> **Hypothesis H19-A (Regime-Conditional Trend Following)**:  
> Cryptographic trend-following alpha is concentrated in identifiable market regimes (broad market trend alignment and moderate volatility). Filtering `Trend_EMA_50` entries to execute only when causal regime conditions are favorable produces statistically significant incremental risk-adjusted return ($\Delta\text{Sharpe} \ge +0.20$, paired bootstrap 95% CI lower bound $> 0$) and friction robustness across 0 to 50 bps, without relying merely on cash drag.

---

## 2. Pre-Registered Regime Feature Specifications

All regime features are computed **strictly causally** using data available at bar close $T$ (with execution at open of $T+1$).

### Feature 1: Market-Wide Trend Breadth ($B_T$)
$$\text{Breadth}_T = \frac{1}{9} \sum_{i=1}^{9} \mathbb{I}(\text{Close}_{i, T} > \text{EMA}_{50, i, T})$$

**Pre-Registered Categorization**:
- **Adverse / Severe Chop**: $\text{Breadth}_T < 0.30$ ($<3$ of 9 assets trending).
- **Neutral**: $0.30 \le \text{Breadth}_T < 0.60$ ($3$ to $5$ assets trending).
- **Favorable / Broad Bull Trend**: $\text{Breadth}_T \ge 0.60$ ($\ge 6$ of 9 assets trending).

### Feature 2: Asset-Specific Trend Quality ($Q_{i, T}$)
An asset's trend is considered high-quality if:
1. $\text{Close}_{i, T} > \text{EMA}_{50, i, T}$
2. $\text{EMA}_{20, i, T} > \text{EMA}_{50, i, T}$
3. $\text{Slope}_{10}(\text{EMA}_{50, i, T}) > 0$

### Feature 3: Volatility State ($V_{i, T}$)
$$\text{ATR Percentile}_{i, T} = \text{PercentileRank}_{60\text{d}}(\text{ATR}_{14, i, T})$$
- **Low Volatility**: $\text{ATR Percentile} < 20\%$
- **Normal Volatility**: $20\% \le \text{ATR Percentile} \le 80\%$
- **Extreme Panic Volatility**: $\text{ATR Percentile} > 80\%$

---

## 3. Pre-Registered Candidate Filter Configurations

We evaluate 3 discrete, pre-registered regime filter rules:

1. **`Regime_Breadth_Filter`**:
   - Long asset $i$ if $\text{Close}_{i, T} > \text{EMA}_{50, i, T}$ **AND** $\text{Breadth}_T \ge 0.30$ (avoids entering during complete market-wide chop).
2. **`Regime_Strong_Breadth_Filter`**:
   - Long asset $i$ if $\text{Close}_{i, T} > \text{EMA}_{50, i, T}$ **AND** $\text{Breadth}_T \ge 0.60$ (enters only during confirmed market-wide bull expansions).
3. **`Regime_Composite_Filter`**:
   - Long asset $i$ if $\text{Close}_{i, T} > \text{EMA}_{50, i, T}$ **AND** $\text{EMA}_{20, i, T} > \text{EMA}_{50, i, T}$ **AND** $\text{Breadth}_T \ge 0.30$ **AND** $\text{ATR Percentile}_{i, T} \le 80\%$ (trend alignment + breadth + avoids panic volatility).

---

## 4. Multi-Friction Sensitivity Grid

Friction scenarios evaluated for every candidate:
$$\text{Friction} \in \{0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0\}\text{ bps round-trip}$$

---

## 5. Formal Acceptance Gates (Must Pass All 6)

1. **Gate 1 (Incremental Net Return)**: $\Delta\text{Net Annual Return} > 0.0\%$ at canonical 30.0 bps.
2. **Gate 2 (Incremental Sharpe Lift)**: $\Delta\text{Net Sharpe} \ge +0.20$ vs baseline.
3. **Gate 3 (Paired Bootstrap CI)**: Paired bootstrap 95% CI lower bound for $\Delta\text{Sharpe} > 0.0$ ($N=10,000, \text{seed}=42$).
4. **Gate 4 (Severe Friction Survival)**: $\text{Net Sharpe} > 0.0$ at 45.0 bps and 50.0 bps friction.
5. **Gate 5 (Turnover Efficiency)**: Annual turnover friction drag $\le 50.0\%$ of gross excess return.
6. **Gate 6 (Sub-Period Stability)**:
   - Min Annual Sharpe $\ge 0.0$
   - Max Annual Drawdown $\le 45.0\%$
   - Min Annual Return $\ge -20.0\%$ across all calendar years.
