"""
paper_trading/metrics.py
----------------------------
Turns a VirtualPortfolio's equity curve + closed trades into the
standard performance stats. Deliberately mirrors ml/backtest.py's
_compute_metrics formulas (same Sharpe/drawdown/win-rate/profit-factor
definitions) so paper-trading results and offline-backtest results are
directly comparable — a strategy shouldn't look different just because
it's being measured by a different metrics function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paper_trading.portfolio import VirtualPortfolio


def compute_equity_curve_series(portfolio: VirtualPortfolio) -> pd.Series:
    if not portfolio.equity_curve:
        return pd.Series(dtype=float)
    ts = [s.ts for s in portfolio.equity_curve]
    equity = [s.equity for s in portfolio.equity_curve]
    return pd.Series(equity, index=pd.DatetimeIndex(ts), name="equity")


def compute_metrics(portfolio: VirtualPortfolio, periods_per_year: int = 8760) -> dict:
    """
    Returns a dict with:
      total_return, sharpe_ratio, max_drawdown, num_trades, win_rate,
      profit_factor, avg_trade_return, realized_pnl, unrealized_pnl,
      cash, equity, num_open_positions
    """
    equity_series = compute_equity_curve_series(portfolio)

    if len(equity_series) >= 2:
        bar_returns = equity_series.pct_change().fillna(0.0)
        total_return = equity_series.iloc[-1] / portfolio.starting_cash - 1
        ret_std = bar_returns.std()
        sharpe = (
            (bar_returns.mean() / ret_std) * np.sqrt(periods_per_year)
            if ret_std and ret_std > 0
            else np.nan
        )
        running_max = equity_series.cummax()
        drawdown = equity_series / running_max - 1
        max_drawdown = drawdown.min()
    else:
        total_return = np.nan
        sharpe = np.nan
        max_drawdown = np.nan

    trades = portfolio.closed_trades
    if trades:
        wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
        losses = [t.realized_pnl for t in trades if t.realized_pnl <= 0]
        win_rate = len(wins) / len(trades)
        gross_loss = abs(sum(losses))
        profit_factor = (sum(wins) / gross_loss) if gross_loss > 0 else np.inf
        avg_trade_return = sum(t.realized_pnl_pct for t in trades) / len(trades)
    else:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_return = np.nan

    last_equity = equity_series.iloc[-1] if len(equity_series) else portfolio.cash

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
        "realized_pnl": portfolio.realized_pnl_total(),
        "cash": portfolio.cash,
        "equity": last_equity,
        "num_open_positions": len(portfolio.open_positions),
    }


def print_summary(portfolio: VirtualPortfolio, periods_per_year: int = 8760) -> None:
    m = compute_metrics(portfolio, periods_per_year=periods_per_year)
    print("=" * 50)
    print("Paper Trading Performance Summary")
    print("=" * 50)
    print(f"Equity:            {m['equity']:.2f}")
    print(f"Cash:              {m['cash']:.2f}")
    print(f"Open Positions:    {m['num_open_positions']}")
    total_return = m["total_return"]
    print(f"Total Return:      {total_return:.2%}" if pd.notna(total_return) else "Total Return:      n/a")
    sharpe = m["sharpe_ratio"]
    print(f"Sharpe Ratio:      {sharpe:.2f}" if pd.notna(sharpe) else "Sharpe Ratio:      n/a")
    max_dd = m["max_drawdown"]
    print(f"Max Drawdown:      {max_dd:.2%}" if pd.notna(max_dd) else "Max Drawdown:      n/a")
    print(f"Num Trades:        {m['num_trades']}")
    win_rate = m["win_rate"]
    print(f"Win Rate:          {win_rate:.2%}" if pd.notna(win_rate) else "Win Rate:          n/a")
    print(f"Profit Factor:     {m['profit_factor']:.2f}" if pd.notna(m["profit_factor"]) else "Profit Factor:     n/a")
    print(f"Realized PnL:      {m['realized_pnl']:.2f}")
    print("=" * 50)