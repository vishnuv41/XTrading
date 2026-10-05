# Phase 20 Research Pre-Registration: Portfolio Construction & Risk Management

**Protocol Version**: 1.0.0  
**Created Timestamp**: 2026-10-05T12:55:00Z  
**Research Boundary**: Frozen Historical Development Window (2020-08-11 to 2024-12-31 UTC, $N=1,604$ Daily Bars)  
**Universe**: 9 Assets (`BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `XRP/USDT`, `ADA/USDT`, `LTC/USDT`, `SOL/USDT`, `DOGE/USDT`, `LINK/USDT`)  
**Underlying Signal Generator**: `Trend_EMA_50` (Long if $\text{Close}_{i, t} > \text{EMA}_{50, i, t}$)  
**Comparator Baseline**: `Trend_EMA_50` Equal-Weighted (1/9th per asset, Unconstrained)  
**Isolation Rule**: Phase 20 operates strictly in isolation. Phase 17 forward tracker is isolated and untouched.

---

## 1. Core Economic Hypothesis

> **Hypothesis H20 (Portfolio Construction & Volatility Normalization)**:  
> In cryptocurrency trend-following, asset returns share heavy common market risk but carry vastly different idiosyncratic volatilities (e.g. BTC vs DOGE/SOL). Replacing naive equal weighting with **volatility-inverse risk parity sizing** and **continuous breadth-linked exposure scaling** improves the portfolio's risk-adjusted economics ($\Delta\text{Sharpe} \ge +0.20$, paired bootstrap 95% CI lower bound $> 0$) and significantly reduces maximum drawdown, while maintaining friction robustness across 0 to 50 bps.

---

## 2. Pre-Registered Portfolio Construction Mechanisms

All portfolio weights are computed **strictly causally** at daily bar close $T$ using data up to $T$ (with execution at open of $T+1$).

### Mechanism 1: Volatility-Inverse Risk Parity Sizing (`Risk_Parity_Sizing`)
For all assets where $\text{Close}_{i, T} > \text{EMA}_{50, i, T}$:
$$w_{i, T} \propto \frac{1}{\text{ATR}_{14, i, T} / \text{Close}_{i, T}}$$
Normalized such that $\sum_{i \in \text{Open}} w_{i, T} = 1.0$ (or target portfolio exposure).
*Economic Rationale*: Equalizes risk contribution across high-beta altcoins and lower-volatility majors, preventing volatile meme/altcoins from dominating portfolio drawdowns.

### Mechanism 2: Continuous Breadth-Linked Exposure Scaling (`Breadth_Scaled_Exposure`)
Instead of binary on/off switching, total portfolio exposure $E_T \in [0.0, 1.0]$ scales continuously with market-wide participation:
$$E_T = \text{clip}\left(\frac{\text{Breadth}_T}{0.60}, 0.20, 1.0\right)$$
Where $\text{Breadth}_T = \frac{1}{9}\sum_{i=1}^9 \mathbb{I}(\text{Close}_{i, T} > \text{EMA}_{50, i, T})$.
*Economic Rationale*: Naturally de-leverages the portfolio into cash during fragmented chop without missing trend expansions.

### Mechanism 3: Composite Volatility-Parity + Breadth Scaling (`Composite_Risk_Portfolio`)
Combines Mechanism 1 (Volatility-Inverse Sizing) with Mechanism 2 (Continuous Breadth Exposure Scaling):
$$w_{i, T} = E_T \times \frac{w_{\text{RiskParity}, i, T}}{\sum_j w_{\text{RiskParity}, j, T}}$$

---

## 3. Multi-Friction Sensitivity Grid

Every candidate is evaluated across:
$$\text{Friction} \in \{0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0\}\text{ bps round-trip}$$

---

## 4. Formal Acceptance Gates (Must Pass All 6)

1. **Gate 1 (Incremental Net Return)**: $\Delta\text{Net Annual Return} > 0.0\%$ at canonical 30.0 bps.
2. **Gate 2 (Incremental Sharpe Lift)**: $\Delta\text{Net Sharpe} \ge +0.20$ vs equal-weighted baseline.
3. **Gate 3 (Paired Bootstrap CI)**: Paired bootstrap 95% CI lower bound for $\Delta\text{Sharpe} > 0.0$ ($N=10,000, \text{seed}=42$).
4. **Gate 4 (Severe Friction Survival)**: $\text{Net Sharpe} > 0.0$ across 45.0 bps and 50.0 bps friction.
5. **Gate 5 (Turnover Efficiency)**: Annual turnover friction drag $\le 50.0\%$ of gross excess alpha.
6. **Gate 6 (Sub-Period Stability)**:
   - Min Annual Sharpe $\ge 0.0$
   - Max Annual Drawdown $\le 45.0\%$
   - Min Annual Return $\ge -20.0\%$ across all calendar years.
