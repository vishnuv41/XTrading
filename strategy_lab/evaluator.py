from __future__ import annotations
"""
strategy_lab/evaluator.py
--------------------------
Standardized offline backtest and metrics evaluator for Strategy Lab.
Enforces identical cost model (10 BPS fees, 5 BPS slippage per side = 30 BPS roundtrip).
"""
import math
import numpy as np
import pandas as pd
from typing import Type
from strategy_lab.base_strategy import BaseStrategy

DEFAULT_TAKER_FEE_PCT = 0.0010   # 10 BPS per side
DEFAULT_SLIPPAGE_PCT = 0.0005    # 5 BPS per side

def run_strategy_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    warmup_bars: int = 100,
    risk_pct_per_trade: float = 0.01,
    taker_fee_pct: float = None,
    slippage_pct: float = None,
) -> dict:
    """
    Runs a deterministic bar-by-bar backtest of a strategy over an OHLCV DataFrame.
    """
    fee_pct = DEFAULT_TAKER_FEE_PCT if taker_fee_pct is None else taker_fee_pct
    slip_pct = DEFAULT_SLIPPAGE_PCT if slippage_pct is None else slippage_pct
    if len(df) <= warmup_bars:
        raise ValueError(f"DataFrame length ({len(df)}) must be > warmup_bars ({warmup_bars})")

    cash = initial_capital
    position = None  # None or dict
    completed_trades = []

    for i in range(warmup_bars, len(df)):
        window = df.iloc[max(0, i - 250): i + 1]
        current_bar = window.iloc[-1]
        c_ts = current_bar["timestamp"]
        c_high = float(current_bar["high"])
        c_low = float(current_bar["low"])
        c_close = float(current_bar["close"])

        # 1. Check exit if position is open
        if position is not None:
            side = position["side"]
            sl = position["stop_loss"]
            tp = position["take_profit"]
            entry_p = position["entry_price"]
            size = position["size"]
            entry_ts = position["entry_ts"]
            bars_held = i - position["entry_bar_idx"]

            exit_triggered = False
            exit_price = 0.0
            exit_reason = None

            hit_sl = False
            hit_tp = False
            same_bar_collision = False

            if side == "LONG":
                hit_sl = (c_low <= sl)
                hit_tp = (c_high >= tp)
                if hit_sl and hit_tp:
                    same_bar_collision = True
                    exit_triggered = True
                    exit_price = sl * (1 - slip_pct)
                    exit_reason = "stop_loss_collision"
                elif hit_sl:
                    exit_triggered = True
                    exit_price = sl * (1 - slip_pct)
                    exit_reason = "stop_loss"
                elif hit_tp:
                    exit_triggered = True
                    exit_price = tp * (1 - slip_pct)
                    exit_reason = "take_profit"
                elif bars_held >= 48:
                    exit_triggered = True
                    exit_price = c_close * (1 - slip_pct)
                    exit_reason = "timeout"
            elif side == "SHORT":
                hit_sl = (c_high >= sl)
                hit_tp = (c_low <= tp)
                if hit_sl and hit_tp:
                    same_bar_collision = True
                    exit_triggered = True
                    exit_price = sl * (1 + slip_pct)
                    exit_reason = "stop_loss_collision"
                elif hit_sl:
                    exit_triggered = True
                    exit_price = sl * (1 + slip_pct)
                    exit_reason = "stop_loss"
                elif hit_tp:
                    exit_triggered = True
                    exit_price = tp * (1 + slip_pct)
                    exit_reason = "take_profit"
                elif bars_held >= 48:
                    exit_triggered = True
                    exit_price = c_close * (1 + slip_pct)
                    exit_reason = "timeout"

            if exit_triggered:
                # Compute realized PnL
                if side == "LONG":
                    gross_pnl = (exit_price - entry_p) * size
                else:
                    gross_pnl = (entry_p - exit_price) * size

                exit_notional = exit_price * size
                exit_fee = exit_notional * fee_pct
                entry_fee = position["entry_fee"]
                total_fees = entry_fee + exit_fee
                net_pnl = gross_pnl - total_fees

                # CORRECTED ACCOUNTING: cash balance changes strictly by net_pnl
                cash += net_pnl
                completed_trades.append({
                    "symbol": strategy.symbol,
                    "side": side,
                    "entry_ts": entry_ts,
                    "exit_ts": c_ts,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "size": size,
                    "notional": position["notional"],
                    "risk_usd": position["risk_usd"],
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "same_bar_collision": same_bar_collision,
                    "gross_pnl": gross_pnl,
                    "total_fees": total_fees,
                    "net_pnl": net_pnl,
                    "pnl_pct": (net_pnl / position["notional"]) * 100.0 if position["notional"] > 0 else 0.0,
                    "equity_after": cash
                })
                position = None

        # 2. Check new signal if position is flat
        if position is None:
            res = strategy.on_bar(window)
            sig = res.get("signal", "HOLD")
            if sig in ["LONG", "SHORT"]:
                raw_entry = float(res.get("entry_price", c_close))
                sl = float(res.get("stop_loss"))
                tp = float(res.get("take_profit"))
                
                entry_price = raw_entry * (1 + slip_pct) if sig == "LONG" else raw_entry * (1 - slip_pct)
                risk_dist = abs(entry_price - sl)
                if risk_dist > 0:
                    target_risk_cash = cash * risk_pct_per_trade
                    size = target_risk_cash / risk_dist
                    notional = size * entry_price
                    entry_fee = notional * fee_pct
                    
                    if cash >= entry_fee:
                        position = {
                            "side": sig,
                            "entry_price": entry_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "size": size,
                            "notional": notional,
                            "risk_usd": target_risk_cash,
                            "entry_fee": entry_fee,
                            "entry_ts": c_ts,
                            "entry_bar_idx": i
                        }

    # Summary Metrics Calculation
    total_trades = len(completed_trades)
    if total_trades == 0:
        return {
            "strategy": strategy.name,
            "symbol": strategy.symbol,
            "timeframe": strategy.timeframe,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl_usd": 0.0,
            "net_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "completed_trades": []
        }

    wins = [t for t in completed_trades if t["net_pnl"] > 0]
    losses = [t for t in completed_trades if t["net_pnl"] <= 0]
    gross_profit = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    net_pnl = sum(t["net_pnl"] for t in completed_trades)

    # Max Drawdown
    cum_pnl = 0.0
    peak = initial_capital
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for t in completed_trades:
        cum_pnl += t["net_pnl"]
        eq = initial_capital + cum_pnl
        if eq > peak:
            peak = eq
        dd = peak - eq
        dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd_usd:
            max_dd_usd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Sharpe Ratio
    pnls = [t["net_pnl_pct"] if "net_pnl_pct" in t else t["pnl_pct"] for t in completed_trades]
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
    mean_pnl = np.mean(pnls) if len(pnls) > 0 else 0.0
    sharpe = (mean_pnl / std_pnl * math.sqrt(total_trades)) if std_pnl > 0 else 0.0

    return {
        "strategy": strategy.name,
        "symbol": strategy.symbol,
        "timeframe": strategy.timeframe,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (len(wins) / total_trades) * 100.0,
        "profit_factor": round(profit_factor, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "net_pnl_pct": round((net_pnl / initial_capital) * 100.0, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "completed_trades": completed_trades
    }
