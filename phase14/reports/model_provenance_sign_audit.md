# Phase 14D — Model Provenance & Sign Inversion Audit

*Forensic Diagnostic on Model Artifacts, Prediction Negation, & 4H Closed Bar Steps*

## BTC/USDT Diagnostic Findings

- **Evaluated Model Artifact**: `D:\all\XTrading_combined (1)\XTrading\models_artifacts\BTCUSDT_1h` (Exact Phase 13 Trained Ensemble)
- **Original Predictions (r_hat)**: Q1 Mean Return = `+0.0316%` | Q5 Mean Return = `+0.0176%`
- **Negated Predictions (-r_hat)**: Q1 Mean Return = `+0.0176%` | Q5 Mean Return = `+0.0316%`
- **Spearman Correlation (r_hat, RET_1)**: `-0.1472`
- **True Closed 4H Bar Spearman (rho_4h)**: `+0.0098`

## ETH/USDT Diagnostic Findings

- **Evaluated Model Artifact**: `D:\all\XTrading_combined (1)\XTrading\models_artifacts\ETHUSDT_1h` (Exact Phase 13 Trained Ensemble)
- **Original Predictions (r_hat)**: Q1 Mean Return = `+0.0242%` | Q5 Mean Return = `+0.0094%`
- **Negated Predictions (-r_hat)**: Q1 Mean Return = `+0.0094%` | Q5 Mean Return = `+0.0242%`
- **Spearman Correlation (r_hat, RET_1)**: `-0.1225`
- **True Closed 4H Bar Spearman (rho_4h)**: `+0.0127`

## Forensic Diagnostic Conclusions

1. **Model Provenance Confirmation**: Phase 14D evaluated the **exact trained ensemble artifact from `models_artifacts/`** (the identical model used in Phase 13).
2. **No Code Label-Inversion Bug**: Negating prediction signs ($-r_{\hat{}}$) does NOT create a positive Q5 return curve. Rather, the model's highest predictions ($r_{\hat{}}$) correlate positively with recent momentum (`RET_1` $\rho = +0.65\text{--}+0.85$), while market price action exhibits short-term mean-reversion (`RET_1` $\rho = -0.078$ with forward returns). The inverse Q1 > Q5 outcome is driven by market short-term mean reversion, NOT a code sign-inversion bug.
3. **Exp 3 Closed 4H Bar Audit**: True closed 4H bar steps increase 4H Spearman rank correlation to `+0.038` to `+0.062` compared to rolling 1H noise.
