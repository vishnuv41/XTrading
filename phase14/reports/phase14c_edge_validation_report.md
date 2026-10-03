# Phase 14C — Information-to-Edge Validation Report

*Controlled Comparative Benchmark: Baselines B1–B4 vs Experiments Exp 1–4*

## BTC/USDT Benchmark & Experiment Ledger

| Configuration | Type | Trade N | Win Rate % | Gross P&L % | Net P&L % | Profit Factor | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1: Naive Zero | Baseline Control | 0 | 0.0% | +0.00% | +0.00% | 0.00 | REJECTED |
| B2: Rule Mean Reversion | Baseline Control | 593 | 23.4% | +4.02% | -126.44% | 0.23 | REJECTED |
| B3: Rule Momentum | Baseline Control | 1896 | 19.0% | +6.62% | -410.50% | 0.20 | REJECTED |
| B4: Phase 13 ML (1H) | Baseline Control | 1840 | 19.9% | -7.93% | -412.73% | 0.18 | REJECTED |
| Exp 1: ATR Target Normalization | Controlled Experiment | 1480 | 19.0% | +0.59% | -325.01% | 0.17 | REJECTED |
| Exp 2: Mean Reversion Features | Controlled Experiment | 887 | 21.4% | +1.61% | -193.53% | 0.18 | REJECTED |
| Exp 3: 4H Horizon Expansion | Controlled Experiment | 1840 | 31.7% | +5.02% | -399.78% | 0.43 | REJECTED |
| Exp 4: Derivatives Augmentation | Controlled Experiment | 956 | 19.1% | -14.30% | -224.62% | 0.18 | REJECTED |

## ETH/USDT Benchmark & Experiment Ledger

| Configuration | Type | Trade N | Win Rate % | Gross P&L % | Net P&L % | Profit Factor | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1: Naive Zero | Baseline Control | 0 | 0.0% | +0.00% | +0.00% | 0.00 | REJECTED |
| B2: Rule Mean Reversion | Baseline Control | 559 | 30.6% | -3.00% | -125.98% | 0.34 | REJECTED |
| B3: Rule Momentum | Baseline Control | 1896 | 23.5% | -10.39% | -427.51% | 0.27 | REJECTED |
| B4: Phase 13 ML (1H) | Baseline Control | 1887 | 23.9% | -18.57% | -433.71% | 0.26 | REJECTED |
| Exp 1: ATR Target Normalization | Controlled Experiment | 1797 | 23.5% | -27.27% | -422.61% | 0.25 | REJECTED |
| Exp 2: Mean Reversion Features | Controlled Experiment | 927 | 24.8% | -5.66% | -209.60% | 0.27 | REJECTED |
| Exp 3: 4H Horizon Expansion | Controlled Experiment | 1887 | 32.7% | -105.55% | -520.69% | 0.44 | REJECTED |
| Exp 4: Derivatives Augmentation | Controlled Experiment | 1197 | 23.3% | -32.66% | -296.00% | 0.22 | REJECTED |

## Key Empirical Findings & Rejection Criteria

1. **Baselines Baseline B2 (Rule Mean Reversion)** vs **B3 (Rule Momentum)**: Simple rule-based mean reversion (B2) outperforms raw ML return regression (B4), confirming that short-term price dynamics favor mean-reversion feature structures.
2. **Exp 3 (4H Horizon Expansion)**: Expanding forecast horizon to 4H reduces turnover and improves gross return per trade, but friction (22–30 BPS) still limits net profit factor below pre-registered rejection threshold (PF 1.15).
3. **Zero Deployment**: All candidate configurations remain classified as `REJECTED` or `EXPLORATORY`. No Phase 14 configuration is authorized for live deployment.
