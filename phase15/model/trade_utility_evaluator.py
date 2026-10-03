"""
phase15/model/trade_utility_evaluator.py
-----------------------------------------
Standardized Evaluation Module for Phase 15.

Computes both clipped-probability classification metrics and realistic trade-level utility metrics:
1. Classification Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, Brier Score, Log Loss (with EPSILON clipping).
2. Trade Utility Metrics: Signal Count, TP Count, SL Count, Win Rate (%), Net P&L (in R-units), Profit Factor, Sharpe Ratio, Max Drawdown (%).
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

EPSILON = 1e-15


def evaluate_predictions_with_trade_utility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_raw: Optional[np.ndarray] = None,
    r_tp: float = 2.0,
    r_sl: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates predictions with probability clipping and trade-level utility.
    
    Args:
        y_true: Binary array (1 for ACCEPT, 0 for REJECT).
        y_pred: Binary array of model predictions (1 for ACCEPT, 0 for REJECT).
        y_proba: Continuous probability estimates in [0, 1].
        label_raw: Raw triple barrier outcome array (1 for PT, -1 for SL, 0 for Timeout).
        r_tp: Reward in R-units for PT hit (default: +2.0R).
        r_sl: Loss in R-units for SL hit (default: -1.0R).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    
    # 1. Probability Clipping for Clean Metric Audit
    y_proba_clipped = np.clip(y_proba, EPSILON, 1.0 - EPSILON)
    
    # 2. Classification Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_proba_clipped))
    loss_val = float(log_loss(y_true, y_proba_clipped, labels=[0, 1]))
    
    # 3. Trade Utility Metrics
    accepted_mask = (y_pred == 1)
    signal_count = int(np.sum(accepted_mask))
    
    if signal_count == 0:
        trade_metrics = {
            "signal_count": 0,
            "tp_count": 0,
            "sl_count": 0,
            "timeout_count": 0,
            "win_rate_pct": 0.0,
            "net_pnl_r": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_r": 0.0
        }
    else:
        if label_raw is not None:
            trade_outcomes = np.asarray(label_raw)[accepted_mask]
            tp_count = int(np.sum(trade_outcomes == 1))
            sl_count = int(np.sum(trade_outcomes == -1))
            timeout_count = int(np.sum(trade_outcomes == 0))
            
            # Returns in R-units
            returns_r = np.where(trade_outcomes == 1, r_tp, np.where(trade_outcomes == -1, -r_sl, 0.0))
        else:
            actual_accepted = y_true[accepted_mask]
            tp_count = int(np.sum(actual_accepted == 1))
            sl_count = int(np.sum(actual_accepted == 0))
            timeout_count = 0
            returns_r = np.where(actual_accepted == 1, r_tp, -r_sl)
            
        win_rate_pct = float((tp_count / signal_count) * 100.0)
        gross_gains = float(np.sum(returns_r[returns_r > 0]))
        gross_losses = float(np.abs(np.sum(returns_r[returns_r < 0])))
        
        if gross_losses > 0:
            profit_factor = float(gross_gains / gross_losses)
        elif gross_gains > 0:
            profit_factor = 99.0
        else:
            profit_factor = 1.0
            
        net_pnl_r = float(np.sum(returns_r))
        
        # Sharpe ratio calculation
        std_ret = float(np.std(returns_r))
        mean_ret = float(np.mean(returns_r))
        sharpe_ratio = float((mean_ret / (std_ret + 1e-8)) * np.sqrt(252)) if std_ret > 1e-8 else 0.0
        
        # Maximum Drawdown calculation (in R-units)
        cum_pnl = np.cumsum(returns_r)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_drawdown_r = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        trade_metrics = {
            "signal_count": signal_count,
            "tp_count": tp_count,
            "sl_count": sl_count,
            "timeout_count": timeout_count,
            "win_rate_pct": round(win_rate_pct, 2),
            "net_pnl_r": round(net_pnl_r, 2),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown_r": round(max_drawdown_r, 2)
        }
        
    return {
        "classification": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(loss_val, 4)
        },
        "trade_utility": trade_metrics
    }
