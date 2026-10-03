# Phase 13 Portfolio State Architecture Report

**Audit Date:** 2026-09-10  
**Status:** `IMPLEMENTED & VERIFIED` (38/38 Tests Passing)  
**Subject:** Unified Global Portfolio State Architecture & Dashboard Data Contract for Phase 13 Paper Trading.

---

## Executive Summary

To resolve the per-daemon cash isolation and daemon restart cash reset defect (the **$4.18 discrepancy**), the paper trading engine state management was upgraded from process-local in-memory state to a **Unified Global Paper Portfolio** backed by PostgreSQL as the persistent source of truth.

### Key Architectural Enhancements
1. **Model A Capital Semantics ($10,000 Total Portfolio):**  
   The initial account capital of **$10,000.00** is defined as the single shared capital pool across all traded symbol daemons (BTC/USDT and ETH/USDT combined).
2. **PostgreSQL Global State Resolution (`load_global_portfolio_state`):**  
   Global available cash and portfolio equity are dynamically resolved from `trade_log`:
   $$\text{Global Cash} = \$10,000.00 + \sum_{\text{closed}} \text{realized\_pnl} - \sum_{\text{active open}} (\text{margin} + \text{fee})$$
3. **Restart Cash Continuity:**  
   Daemon process restarts now query `load_global_portfolio_state()` upon initialization (`restore_state_from_db`), restoring exact global trailing cash balance even when active open positions count is 0. Cash never resets to $10,000.00 on restart.
4. **Multi-Daemon Cash Synchronization:**  
   Symbol daemons re-sync available cash from PostgreSQL prior to risk evaluation and order entries, ensuring trades closed by one daemon (e.g. BTC) are instantly recognized in available cash by other daemons (e.g. ETH).
5. **Zero Historical Record Rewriting:**  
   All 5 completed trade PnLs ($+87.77, +\$124.26, -\$103.67, -\$104.18, -\$49.59 \rightarrow \text{Net } -\$45.41$) and existing `trade_log` database rows remain 100% untouched and preserved.

---

## State Ownership & Lifecycle Architecture

```text
                           PostgreSQL (trade_log)
                                     │
                    load_global_portfolio_state()
                                     │
               ┌─────────────────────┴─────────────────────┐
               │                                           │
       BTC/USDT Engine                             ETH/USDT Engine
     (Process / Daemon)                          (Process / Daemon)
               │                                           │
  - Syncs available cash                      - Syncs available cash
  - Shares $10k global pool                   - Shares $10k global pool
  - Preserves trailing PnL                    - Preserves trailing PnL
```

### 1. State Recovery Flow on Process Restart
When an engine starts up:
1. `restore_state_from_db()` executes `db_logger.load_global_portfolio_state()`.
2. Computes total realized PnL of all past closed trades across all symbols.
3. Computes tied margin and fees of all active open positions across all symbols.
4. Sets `self.portfolio.cash = available_cash`.
5. Loads active open positions for the engine's target symbol/timeframe.

### 2. Multi-Daemon Cash Synchronization Flow
During live bar processing (`on_bar`):
1. Step 1 exits evaluate and close any active position for that symbol.
2. Step 1.5 re-syncs `self.portfolio.cash` from `load_global_portfolio_state()` to capture any PnL realized by concurrent symbol engines.
3. Step 2 fresh prediction and Step 4 entry attempt evaluate risk sizing against exact global portfolio equity and available cash.

---

## Data Contract for Real-Time Dashboard

The real-time dashboard (`experiments/dashboard.py`) must follow the strict data contract defined below to ensure absolute accounting integrity:

### 1. Canonical Portfolio Equity vs Reconciliation Equity
* **CANONICAL EQUITY (Source of Truth):**
  $$\text{Equity}_{\text{canonical}} = \$10,000.00 + \sum_{\text{closed trades in DB}} \text{realized\_pnl} + \sum_{\text{open positions}} \text{unrealized\_pnl}$$
* **RECONCILIATION EQUITY (Audit Verification):**
  $$\text{Equity}_{\text{audit}} = \$10,000.00 + \sum_{i=1}^{N} \text{realized\_pnl}_i$$
* **STATUS:** `RECONCILED` if $|\text{Equity}_{\text{canonical}} - \text{Equity}_{\text{audit}}| < \$0.01$.

### 2. Dashboard Display Rules
1. **Never read `trade_log.equity_after` from single DB rows as the portfolio equity.** Affected historical rows contain process-local cash observations prior to this architectural upgrade.
2. **Display explicit Accounting Integrity card:**
   ```text
   ACCOUNTING INTEGRITY
   --------------------
   Canonical Equity:    $9,954.59
   Reconciled Equity:   $9,954.59
   Status:              ✓ RECONCILED
   Active Daemons:      BTC/USDT, ETH/USDT
   ```

---

## Verification & Test Results

The architecture fix was verified using the pytest suite (`tests/test_paper_trading.py` and `tests/test_validator.py`):

```powershell
============================= 38 passed in 2.15s ==============================
```

### Verified Test Cases
1. `test_global_portfolio_state_loader`: Verified dynamic global cash calculation over closed trades.
2. `test_daemon_restart_preserves_cash_when_zero_positions`: Verified daemon restart with 0 open positions preserves exact trailing cash ($10,124.26).
3. `test_historical_five_trade_reconciliation_exact_equity`: Verified exact 5-trade global equity reconstruction ($9,954.590778).
4. `35/35` pre-existing paper trading and validation unit tests passed with 100% success.
