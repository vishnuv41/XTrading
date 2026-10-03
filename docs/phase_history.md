# XTrading Research Phase Chronology

## Historical Summary of Research Iterations

| Phase | Description / Experiment Focus | Outcome / Status | Key Metrics / Decision Rationale |
| :--- | :--- | :--- | :--- |
| **Phase 1-4** | Initial single-timeframe features & basic ML baselines | Baseline Established | Established OHLCV validation and PostgreSQL storage schema. |
| **Phase 5-8** | Multi-timeframe indicator expansions & model exploration | Feature Set Frozen | Standardized on 66 MTF causal features and XGBoost+CatBoost ensemble. |
| **Phase 9** | Walk-forward cross validation & rank gate formulation | Validated Gating | Identified 250-prediction rolling Top 1.0% percentile threshold ($0.9900$) as optimal signal filter. |
| **Phase 10** | Protocol bootstrap & strict rank gate validation | **REJECTED** | Bootstrap test failed pre-registered criteria (PF 1.21, Sharpe +0.82, Loss Prob 20.3%). |
| **Phase 11** | Dynamic Supertrend trailing exit variation | **REJECTED** | Bootstrap test failed pre-registered criteria (PF 0.75, Sharpe -1.17, Loss Prob 87.4%). |
| **Phase 12** | Model recalibration & continuous-return regression setup | Completed | Cleaned historical artifacts and prepared Phase 13 prospective baseline parameters. |
| **Phase 13** | Canonical prospective paper trading ($N \ge 30$) | **ONGOING (Active)** | 50/50 XGBoost + CatBoost continuous-return regression → 66 causal MTF features → 250-prediction rolling Top-1% rank gate → 1.5 ATR SL / 3.0 ATR TP → 48-bar timeout → 4-bar cooldown → 10 BPS fees/slippage. Current baseline: $N = 4 / 30$ ($+\$4.18$ net P&L). |

---

## Reconciliation of Historical Phase 10 & 11 Rejections

A comprehensive audit reconciled file timestamps and execution scripts:
- **Phase 10 Protocol Bootstrap**: Script `phase10_protocol_bootstrap.py` generated `phase10_top_1.0pct_trades.csv` (`2026-09-06 08:48 UTC`). Results confirmed Profit Factor 1.21, Sharpe +0.82, and 20.3% loss probability, failing the pre-registered requirement ($PF > 1.50$, $LossProb < 5.0\%$). Status: **REJECTED**.
- **Phase 11 Dynamic Trailing Exit**: Script `phase11_dynamic_exit_bootstrap.py` generated `phase11_dynamic_exit_trades.csv` (`2026-09-06 08:58 UTC`). Results confirmed Profit Factor 0.75, Sharpe -1.17, and 87.4% loss probability. Status: **REJECTED**.
- **Conclusion**: Both Phase 10 and Phase 11 were correctly rejected based on empirical bootstrap statistics. An earlier text note mentioning $PF = 1.54 / 1.62$ was identified as a transcription artifact; underlying CSV data and code remain 100% consistent with the REJECTED determination.
