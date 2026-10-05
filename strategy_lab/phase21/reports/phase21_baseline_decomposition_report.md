# Phase 21: Baseline Edge Decomposition & Forensic Attribution Report

**Timeline**: 2020-08-11 to 2026-09-30 ($N=2242$ Daily Bars)  
**Universe**: 9 Assets (`BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `LTC`, `SOL`, `DOGE`, `LINK`)  
**Canonical Strategy**: `Trend_EMA_50` Equal-Weighted (30.0 bps round-trip friction)  

## 1. Asset-Level Performance Attribution

| Symbol | Net Ann. Return | Net Sharpe | Max Drawdown | Total Trades | Win Rate | Avg Win | Avg Loss | Profit Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BTC/USDT` | +45.92% | 1.137 | 57.34% | 72 | 33.3% | +30.13% | -4.67% | 3.22 |
| `ETH/USDT` | +33.24% | 0.799 | 62.16% | 85 | 37.6% | +20.38% | -5.34% | 2.30 |
| `BNB/USDT` | +73.69% | 1.149 | 58.91% | 82 | 40.2% | +58.75% | -3.01% | 13.12 |
| `XRP/USDT` | +9.54% | 0.475 | 84.76% | 93 | 30.1% | +26.20% | -5.29% | 2.13 |
| `ADA/USDT` | +37.03% | 0.796 | 74.35% | 93 | 31.2% | +36.83% | -5.82% | 2.87 |
| `LTC/USDT` | -14.66% | 0.078 | 93.22% | 99 | 26.3% | +16.31% | -5.26% | 1.10 |
| `SOL/USDT` | +95.42% | 1.221 | 72.50% | 83 | 36.1% | +117.32% | -7.29% | 9.11 |
| `DOGE/USDT` | +90.02% | 0.788 | 88.91% | 79 | 41.8% | +379.80% | -6.80% | 40.07 |
| `LINK/USDT` | -7.62% | 0.236 | 86.77% | 92 | 31.5% | +16.69% | -5.85% | 1.31 |

## 2. Year-by-Year Calendar Attribution

| Year | Net Ann. Return | Net Sharpe | Max Drawdown | Avg Exposure | Annual Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2020` | +36.30% | 0.836 | 24.87% | 54.6% | 42.00 |
| `2021` | +923.43% | 3.084 | 42.08% | 63.3% | 21.35 |
| `2022` | -39.35% | -1.716 | 41.24% | 23.7% | 25.80 |
| `2023` | +59.61% | 1.479 | 27.98% | 56.1% | 29.46 |
| `2024` | +62.09% | 1.340 | 34.59% | 54.6% | 32.49 |
| `2025` | -16.17% | -0.308 | 29.23% | 42.6% | 28.24 |
| `2026` | +19.11% | 0.766 | 20.57% | 34.4% | 24.23 |

## 3. Market Regime Decomposition

| Regime | Days Count | Share of Time | Daily Return (bps) | Ann. Return | Win Day % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bull_market_regime` | 904 | 40.3% | +85.81 bps | +313.42% | 63.4% |
| `sideways_chop_regime` | 390 | 17.4% | -46.22 bps | -168.83% | 45.6% |
| `bear_market_regime` | 948 | 42.3% | -25.19 bps | -92.00% | 17.8% |

## 4. Trade Expectancy & Profit Concentration

- **Total Trades Extracted**: 778
- **Win Rate**: 33.9%
- **Median Trade Return**: -1.92%
- **Average Winner**: +83.62% vs **Average Loser**: -5.49%
- **Payoff Ratio (Win/Loss Size)**: **15.23x**
- **Overall Profit Factor**: **7.82**
- **Top 5 Trades Profit Share**: **78.2%** of total gains
- **Top 10% Trades Profit Share**: **97.4%** of total gains

## 5. Friction Sensitivity & Break-Even Analysis

| Friction Scenario | Net Ann. Return | Net Sharpe | Max Drawdown |
| :--- | :--- | :--- | :--- |
| `0bps` | +65.95% | 1.274 | 52.43% |
| `10bps` | +63.64% | 1.245 | 53.74% |
| `20bps` | +61.37% | 1.216 | 55.11% |
| `30bps` | +59.13% | 1.188 | 56.47% |
| `40bps` | +56.92% | 1.159 | 57.80% |
| `45bps` | +55.82% | 1.145 | 58.45% |
| `50bps` | +54.73% | 1.130 | 59.09% |
| `60bps` | +52.58% | 1.102 | 60.34% |
| `75bps` | +49.41% | 1.059 | 62.14% |
| `100bps` | +44.27% | 0.987 | 64.96% |

**Economic Break-Even Friction**: `> 100 bps`

## 6. Descriptive Benchmark Comparison

| Strategy / Benchmark | Net Ann. Return | Net Sharpe | Max Drawdown | Annual Friction Drag |
| :--- | :--- | :--- | :--- | :--- |
| `Trend_EMA_50 (Baseline)` | +59.13% | 1.188 | 56.47% | 4.20% |
| `Buy_and_Hold` | +64.78% | 1.034 | 78.67% | 0.02% |
| `Static_45pct_Cash` | +41.36% | 1.034 | 53.88% | 0.01% |
| `Trend_EMA_20 (Fast)` | +49.65% | 1.154 | 51.77% | 6.80% |
| `Trend_EMA_100 (Slow)` | +58.58% | 1.156 | 61.86% | 2.67% |
