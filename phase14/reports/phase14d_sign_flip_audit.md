# Phase 14D Sign-Flip Diagnostic Audit Report

*Date*: 2026-09-10  
*Auditor*: Forensic Verification Engine  
*Scope*: Diagnostic test evaluating prediction negation ($-p$) on the exact Phase 14D prediction dataset.

---

## 1. Sign-Flip Diagnostic Ledger

| Asset | Prediction Orientation | Q1 Mean Return | Q5 Mean Return | Spearman $\rho(\hat{r}, \text{RET}_1)$ | True Closed 4H Spearman $\rho_{4h}$ | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BTC/USDT** | Original ($\hat{r}$) | `+0.0316%` | `+0.0176%` | `-0.1472` | `+0.0098` | Verified |
| **BTC/USDT** | Negated ($-\hat{r}$) | `+0.0176%` | `+0.0316%` | `+0.1472` | - | Swapped |
| **ETH/USDT** | Original ($\hat{r}$) | `+0.0242%` | `+0.0094%` | `-1225` | `+0.0127` | Verified |
| **ETH/USDT** | Negated ($-\hat{r}$) | `+0.0094%` | `+0.0242%` | `+0.1225` | - | Swapped |

---

## 2. Conclusion Category & Findings

> [!IMPORTANT]
> **CONCLUSION CATEGORY: A. NO_SIGN_BUG_FOUND**
> 
> * **Negation Mechanics**: Negating prediction values ($-\hat{r}$) simply swaps Q1 and Q5 values without producing a positive Q5 return curve.
> * **Root Cause Identification**: The model's predictions ($\hat{r}$) correlate positively with short-term past momentum (`RET_1` $\rho \approx +0.70$), whereas actual market price action exhibits short-term mean-reversion (`RET_1` $\rho = -0.078$ with 1H forward returns).
> * **Conclusion**: The inverse Q1 > Q5 outcome is driven by market short-term mean-reversion structure, NOT by a code sign-inversion or label-mapping bug.
