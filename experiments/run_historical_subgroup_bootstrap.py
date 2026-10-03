"""
experiments/run_historical_subgroup_bootstrap.py
-------------------------------------------------
Prospective & Retrospective Bootstrap Verification Utility.

PURPOSE & DATA SOURCE DISCREPANCY NOTE:
- For Historical Phase 9B Subgroup Audit (N=22, 44, 50, 100):
  The canonical historical trade logs live in CSV format under:
  models_artifacts/TARGET_FORMULATION_AUDIT/
  (Executed via execute_phase9b_audit.py).

- For Prospective Phase 13 Trade Log Audit (Live DB):
  This script queries PostgreSQL `trade_log` for closed prospective trades.
  During early prospective validation (N < 5), it strictly FAILS LOUDLY with a
  ValueError to prevent any silent synthetic data substitution.

Once Phase 13 prospective trading accumulates N >= 30 completed trades in PostgreSQL,
this script will perform the pre-registered decision gate bootstrap on live trades.
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from database.connection import get_engine

def run_bootstrap_for_sample(returns: np.ndarray, sample_size: int, n_bootstraps: int = 10000, seed: int = 42):
    np.random.seed(seed)
    
    bootstrap_pnls = []
    bootstrap_pfs = []
    bootstrap_sharpes = []
    
    for _ in range(n_bootstraps):
        sample = np.random.choice(returns, size=sample_size, replace=True)
        tot_pnl = np.sum(sample)
        
        wins = sample[sample > 0]
        losses = sample[sample < 0]
        gross_win = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-6
        pf = gross_win / gross_loss
        
        std = np.std(sample)
        sharpe = (np.mean(sample) / std * np.sqrt(24 * 365)) if std > 1e-8 else 0.0
        
        bootstrap_pnls.append(tot_pnl)
        bootstrap_pfs.append(pf)
        bootstrap_sharpes.append(sharpe)
        
    bootstrap_pnls = np.array(bootstrap_pnls)
    bootstrap_pfs = np.array(bootstrap_pfs)
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    
    loss_prob = np.mean(bootstrap_pnls < 0) * 100
    ci_lower_pnl_pct = np.percentile(bootstrap_pnls, 2.5) * 100
    ci_upper_pnl_pct = np.percentile(bootstrap_pnls, 97.5) * 100
    median_pf = np.median(bootstrap_pfs)
    median_sharpe = np.median(bootstrap_sharpes)
    
    return {
        "N": sample_size,
        "loss_prob_pct": loss_prob,
        "ci_lower_pnl_pct": ci_lower_pnl_pct,
        "ci_upper_pnl_pct": ci_upper_pnl_pct,
        "median_pf": median_pf,
        "median_sharpe": median_sharpe,
    }

def main():
    engine = get_engine()
    
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT realized_pnl_pct 
            FROM trade_log 
            WHERE realized_pnl_pct IS NOT NULL AND action = 'CLOSE'
            ORDER BY ts ASC
        """)).fetchall()
        
    historical_returns = np.array([r[0] for r in rows])
    
    if len(historical_returns) < 5:
        raise ValueError(
            f"Insufficient real trade log entries in database (found {len(historical_returns)}). "
            "Refusing to execute bootstrap on synthetic fallback data. Real data required! "
            "Note: To audit historical Phase 9B subgroup CSVs, use execute_phase9b_audit.py."
        )

    print(f"Loaded {len(historical_returns)} trade return records.")
    print("=" * 90)
    print("PROSPECTIVE PHASE 13 BOOTSTRAP ANALYSIS (10,000 RESAMPLES)")
    print("=" * 90)
    print(f"{'Sample Size (N)':<16} | {'Loss Prob (%)':<15} | {'95% CI PnL Range (%)':<25} | {'Median PF':<10} | {'Median Sharpe':<13}")
    print("-" * 90)
    
    for n_sub in [22, 44, 50, 100]:
        res = run_bootstrap_for_sample(historical_returns, n_sub)
        print(f"N = {res['N']:<12} | {res['loss_prob_pct']:>13.2f}% | [{res['ci_lower_pnl_pct']:>+6.2f}%, {res['ci_upper_pnl_pct']:>+6.2f}%] | {res['median_pf']:>9.2f} | {res['median_sharpe']:>12.2f}")

    print("=" * 90)

if __name__ == "__main__":
    main()
