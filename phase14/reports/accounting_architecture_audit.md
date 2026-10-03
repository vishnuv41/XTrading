# Structural Accounting Architecture Audit & Reuse Rule

*Date*: 2026-09-10  
*Scope*: Audit of cash, margin, notional, fee, and PnL accounting across core runtime and research evaluators.

---

## 1. Canonical Accounting Implementation

The canonical, unit-tested open $\rightarrow$ close accounting path in the XTrading codebase is implemented in:
* **[`paper_trading/portfolio.py`](file:///d:/all/XTrading_combined%20%281%29/XTrading/paper_trading/portfolio.py)**: Maintains portfolio cash balance, position tracking, margin reservation, and equity calculation (`cash + unrealized_pnl`).
* **[`paper_trading/execution.py`](file:///d:/all/XTrading_combined%20%281%29/XTrading/paper_trading/execution.py)**: Handles order execution, taker fees, slippage, and position close realized PnL.

---

## 2. Audit of Historical Independent Accounting Implementations

| Implementation Location | Purpose | Historical Status / Issues Found | Canonical Component Equivalence |
| :--- | :--- | :--- | :--- |
| `paper_trading/portfolio.py` | Core Live / Paper Trading | **VERIFIED CANONICAL** (Passed 35/35 pytest suite) | Canonical standard |
| `paper_trading/execution.py` | Order Execution & Fees | **VERIFIED CANONICAL** (Passed 35/35 pytest suite) | Canonical standard |
| `VirtualPortfolio.equity()` | Legacy Backtest Metric | **FIXED** (Earlier bug: margin not added back mid-trade) | `paper_trading/portfolio.py` |
| `ml/backtest.py` | ML Pipeline Backtest | **AUDITED** (Naive full-notional cash deduction) | `paper_trading/execution.py` |
| `strategy_lab/evaluator.py` | Strategy Lab V2 Engine | **REPAIRED & PARITY VERIFIED** (Cash compounding bug corrected) | `paper_trading/portfolio.py` |
| `phase14/execution/cost_model.py` | Phase 14 Friction Engine | **VERIFIED** (Explicit fee, slip, spread, funding model) | `paper_trading/execution.py` |

---

## 3. Mandatory Research Architecture Rule

> [!IMPORTANT]
> **MANDATORY STRUCTURAL REUSE RULE**:
> *"Research strategy logic may be independent, but portfolio, execution, cash, notional/margin, fees, and realized-PnL accounting MUST reuse or adapt the canonical verified execution/accounting components (`paper_trading/portfolio.py` & `paper_trading/execution.py`) wherever technically compatible. New research evaluators are prohibited from re-implementing custom cash or notional arithmetic from scratch."*
