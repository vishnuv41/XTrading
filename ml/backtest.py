"""
ml/backtest.py
-----------------
Simple event-driven backtest for evaluating ml.predict's output against
realized price action. Deliberately independent of strategy/ and
risk_engine/ (which handle the live/production trading logic) — this is
purely for offline model evaluation: does acting on the model's signal
actually make money net of costs, and what does the equity curve/risk
profile look like.

One position at a time (no pyramiding), fixed max holding period, exits
early are NOT modeled here (that's risk_engine's stoploss/takeprofit
job) — this measures the raw signal quality under a simple, consistent
exit rule so results are comparable across model versions.
"""

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    price_col: str = "close",
    max_holding: int = 20,
    confidence_threshold: float = 0.4,
    transaction_cost_bps: float = 5.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Simulate trading `predictions['pred_label']` against `df[price_col]`.

    Parameters
    ----------
    df : OHLCV DataFrame (must share the same index/length as `predictions`).
    predictions : output of ml.predict.predict — needs 'pred_label'
        ({-1,0,1}) and 'confidence' (0-1) columns.
    max_holding : bars to hold a position before forcing exit.
    confidence_threshold : minimum confidence to act on a signal.
    transaction_cost_bps : round-trip-half cost in basis points, charged
        on both entry and exit.
    periods_per_year : bars per year, for Sharpe annualization (e.g.
        252 for daily, 252*24 for hourly crypto/FX — pass what matches
        your bar frequency).

    Returns
    -------
    dict with:
        - 'equity_curve' : pd.Series, cumulative equity starting at 1.0
        - 'bar_returns'  : pd.Series, per-bar strategy returns
        - 'trades'       : DataFrame, one row per closed trade
        - 'metrics'      : dict of summary stats
    """
    if len(df) != len(predictions):
        raise ValueError("df and predictions must be the same length/aligned.")

    close = df[price_col].values
    conf = predictions["confidence"].values
    label = predictions["pred_label"].values
    n = len(df)
    cost = transaction_cost_bps / 10000

    bar_returns = np.zeros(n)
    position = 0
    entry_price = None
    entry_idx = None
    bars_in_trade = 0
    trades = []

    def _close_trade(exit_idx, exit_price):
        trade_ret = (exit_price - entry_price) / entry_price * position - 2 * cost
        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "direction": position,
                "bars_held": exit_idx - entry_idx,
                "return": trade_ret,
            }
        )

    for i in range(1, n):
        if position != 0:
            bar_returns[i] = (close[i] - close[i - 1]) / close[i - 1] * position
            bars_in_trade += 1
            if bars_in_trade >= max_holding:
                _close_trade(i, close[i])
                bar_returns[i] -= cost
                position = 0
                bars_in_trade = 0
        else:
            if not np.isnan(conf[i]) and conf[i] >= confidence_threshold and not np.isnan(label[i]) and label[i] != 0:
                position = int(np.sign(label[i]))
                entry_price = close[i]
                entry_idx = i
                bars_in_trade = 0
                bar_returns[i] -= cost

    if position != 0:
        _close_trade(n - 1, close[-1])
        bar_returns[-1] -= cost

    bar_returns_s = pd.Series(bar_returns, index=df.index)
    equity_curve = (1 + bar_returns_s).cumprod()
    trades_df = pd.DataFrame(trades)

    metrics = _compute_metrics(bar_returns_s, equity_curve, trades_df, periods_per_year)

    return {
        "equity_curve": equity_curve,
        "bar_returns": bar_returns_s,
        "trades": trades_df,
        "metrics": metrics,
    }


def _compute_metrics(bar_returns: pd.Series, equity_curve: pd.Series, trades: pd.DataFrame, periods_per_year: int) -> dict:
    total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) else np.nan

    ret_std = bar_returns.std()
    sharpe = (
        (bar_returns.mean() / ret_std) * np.sqrt(periods_per_year)
        if ret_std and ret_std > 0
        else np.nan
    )

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min() if len(drawdown) else np.nan

    if len(trades):
        wins = trades[trades["return"] > 0]["return"]
        losses = trades[trades["return"] <= 0]["return"]
        win_rate = len(wins) / len(trades)
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        avg_trade_return = trades["return"].mean()
    else:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_return = np.nan

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
    }