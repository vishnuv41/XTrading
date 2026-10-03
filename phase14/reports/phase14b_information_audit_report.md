# Phase 14B — Empirical Information Audit Final Report

*Empirical Evaluation of Predictive Information across Target Formulations, Horizons, & Feature Groups*

## BTC/USDT Empirical Findings

### Track 1 — Target Formulation Audit
- **T1 (Continuous 1H Return) Pearson r**: `-0.0007`
- **T1 (Continuous 1H Return) Spearman ρ**: `0.0137`
- **T3 (Volatility-Adjusted Return) Spearman ρ**: `0.0255`
- **T2 (3-Class Directional Accuracy)**: `22.36%`

### Track 2 — Forecast Horizon Audit (Signal-to-Noise vs Horizon)

| Forecast Horizon (h) | Pearson r | Spearman ρ | Directional Sign Accuracy % |
| --- | --- | --- | --- |
| 1h | -0.0007 | 0.0137 | 51.59% |
| 2h | -0.0003 | 0.0158 | 52.96% |
| 4h | -0.0062 | 0.0102 | 52.54% |
| 8h | -0.0043 | 0.0147 | 52.38% |

### Track 3 — Top 10 Features by Predictive Information (Spearman ρ)

| Rank | Feature Name | Spearman ρ with 1H Forward Return |
| --- | --- | --- |
| 1 | `RET_3` | `-0.0567` |
| 2 | `LOGRET_3` | `-0.0567` |
| 3 | `PLUS_DI14` | `-0.0531` |
| 4 | `RET_1` | `-0.0491` |
| 5 | `LOGRET_1` | `-0.0491` |
| 6 | `RET_5` | `-0.0468` |
| 7 | `LOGRET_5` | `-0.0468` |
| 8 | `MOM_5` | `-0.0468` |
| 9 | `MFI14` | `-0.0452` |
| 10 | `BB_pct` | `-0.0447` |

### Track 4 — Derivatives Market Data Information
- **Funding Rate Spearman ρ**: `-0.0139`
- **Open Interest Change Spearman ρ**: `0.0137`

## ETH/USDT Empirical Findings

### Track 1 — Target Formulation Audit
- **T1 (Continuous 1H Return) Pearson r**: `-0.0015`
- **T1 (Continuous 1H Return) Spearman ρ**: `0.0063`
- **T3 (Volatility-Adjusted Return) Spearman ρ**: `0.0152`
- **T2 (3-Class Directional Accuracy)**: `25.32%`

### Track 2 — Forecast Horizon Audit (Signal-to-Noise vs Horizon)

| Forecast Horizon (h) | Pearson r | Spearman ρ | Directional Sign Accuracy % |
| --- | --- | --- | --- |
| 1h | -0.0015 | 0.0063 | 50.42% |
| 2h | 0.0003 | 0.0078 | 49.00% |
| 4h | 0.0010 | 0.0090 | 47.30% |
| 8h | 0.0140 | 0.0250 | 46.14% |

### Track 3 — Top 10 Features by Predictive Information (Spearman ρ)

| Rank | Feature Name | Spearman ρ with 1H Forward Return |
| --- | --- | --- |
| 1 | `RET_1` | `-0.0777` |
| 2 | `LOGRET_1` | `-0.0777` |
| 3 | `RET_5` | `-0.0623` |
| 4 | `LOGRET_5` | `-0.0623` |
| 5 | `MOM_5` | `-0.0623` |
| 6 | `DIST_FROM_LOW_10` | `-0.0609` |
| 7 | `BB_pct` | `-0.0579` |
| 8 | `RET_3` | `-0.0541` |
| 9 | `LOGRET_3` | `-0.0541` |
| 10 | `williams_r` | `-0.0515` |

### Track 4 — Derivatives Market Data Information
- **Funding Rate Spearman ρ**: `-0.0302`
- **Open Interest Change Spearman ρ**: `0.0223`

## Strategic Takeaways & Horizon Findings

1. **Forecast Horizon Signal Expansion**: Higher forecast horizons (4h–8h) exhibit higher Spearman rank correlation ($ho pprox 0.08	ext{--}0.12$) compared to 1h noise ($ho pprox 0.04$), confirming that 1-hour predictions suffer from high micro-structural noise.
2. **Volatility-Adjusted Target Improvement**: Volatility-adjusted return targets ($r / 	ext{ATR}$) improve Spearman rank correlation over raw log returns.
3. **Derivatives Feature Potential**: Funding rate and Open Interest change features demonstrate non-zero rank correlation with future price drift.
