# Future Research Backlog (Isolated Queue)

## Overview
This backlog holds prospective research ideas, feature enhancements, and strategy variations for future evaluation.

> [!IMPORTANT]
> **ISOLATION POLICY**: None of the items listed in this backlog may be tested or implemented on the live Phase 13 pipeline. Phase 13 parameters are 100% frozen until $N \ge 30$ prospective trades complete.

---

## 💡 Candidate Research Proposals

### Proposal R-01: 5m High-Frequency Execution Sub-Layer
- **Concept**: Use 5m timeframe bars for intrabar entry timing refinement and dynamic trailing stop adjustments while keeping 1h signal direction.
- **Objective**: Reduce initial entry slippage and compress stop-loss distance.
- **Evaluation Criteria**: Out-of-sample backtest vs 1h baseline.

### Proposal R-02: Multi-Asset Expansion (SOL, AVAX, NEAR, MATIC)
- **Concept**: Extend the 66 MTF feature engine and XGBoost+CatBoost ensemble to altcoins with liquid futures markets.
- **Objective**: Increase monthly trade frequency and test cross-asset strategy generalization.
- **Evaluation Criteria**: Cross-validation Sharpe ratio $> 1.50$ across 4 altcoins.

### Proposal R-03: Order Book Depth & Liquidations Imbalance Features
- **Concept**: Incorporate real-time order book depth bids/asks asymmetry and forced liquidation volume spikes into the model feature vector.
- **Objective**: Capture microstructural liquidity squeezes prior to large directional breakouts.

### Proposal R-04: Conformal Prediction & Dynamic Gate Thresholds
- **Concept**: Replace fixed 0.9900 percentile gating with adaptive conformal prediction error rate control ($1 - \alpha = 0.95$).
- **Objective**: Dynamically contract signal frequency during high-uncertainty market regimes.

### Proposal R-05: Multi-Objective Portfolio Sizing (Kelly Criterion + Volatility Parity)
- **Concept**: Scale position risk heat dynamically based on calibrated model win probability $P$ and current regime volatility.
- **Objective**: Maximize logarithmic wealth growth while capping maximum portfolio drawdown below 10%.
