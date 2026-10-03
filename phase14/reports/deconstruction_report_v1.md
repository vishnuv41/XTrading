# Phase 14 — Deconstruction Experiments A–D Final Report

*Scientific Deconstruction of Model Signals across Long/Short Tails, Magnitude, & Direction*

## 1. Summary of Experimental Results

### BTC/USDT Signal Deconstruction
- **Directional Accuracy (Exp D)**: `51.61%`
- **Magnitude Correlation (Exp C)**: `0.0370`

#### Walk-Forward OOS Folds (T = 30 BPS Hurdle)

| Fold | Long N | Long Net Return % | Short N | Short Net Return % |
| --- | --- | --- | --- | --- |
| Fold 1 | 281 | -0.230% | 338 | -0.236% |
| Fold 2 | 383 | -0.216% | 230 | -0.196% |
| Fold 3 | 129 | -0.160% | 485 | -0.250% |

### ETH/USDT Signal Deconstruction
- **Directional Accuracy (Exp D)**: `50.39%`
- **Magnitude Correlation (Exp C)**: `0.0858`

#### Walk-Forward OOS Folds (T = 30 BPS Hurdle)

| Fold | Long N | Long Net Return % | Short N | Short Net Return % |
| --- | --- | --- | --- | --- |
| Fold 1 | 194 | -0.186% | 437 | -0.228% |
| Fold 2 | 50 | -0.144% | 581 | -0.215% |
| Fold 3 | 44 | -0.224% | 586 | -0.269% |

## 2. Failure Diagnosis & Situation Classification

### System State Diagnosis:

> [!WARNING]
> **DIAGNOSIS: Situation E — Directional Information Exists, but Disappears After Friction Costs.**
> 
> Both positive and negative model tails contain raw gross directional signal, but standard roundtrip friction (fees + slippage + spread = 22–30 BPS) degrades net expected edge below zero.
