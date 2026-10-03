"""
experiments/monte_carlo.py
------------------------------
Two related but distinct Monte Carlo techniques, adapted from jesse.trade's
"trade-order shuffling" idea, combined here because each answers a
different question and neither alone is sufficient:

1. ORDER-SHUFFLE (permutation, no replacement): reorders the SAME set of
   realized trades. IMPORTANT MATH FACT this module respects rather than
   hides: summing or multiplying the same set of numbers gives the same
   total regardless of order — so the FINAL return is mathematically
   IDENTICAL across every permutation. Shuffling only changes the PATH
   taken to get there. This makes it the right (and only) tool for
   path-dependent risk questions: how much worse could the max drawdown
   have been, how long could the losing streak have run, purely from a
   different sequencing of the exact trades you already have — with
   zero assumption changes. It is NOT a tool for asking "was my return
   lucky" (that question is about which trades happened, not their
   order — see technique 2).

2. BOOTSTRAP (resampling WITH replacement): draws a new set of len(trades)
   trades from the same pool, some repeated, some omitted. This DOES
   vary the final return and gives a genuine confidence interval /
   probability-of-loss estimate — the standard technique for asking
   "how confident should I be that this edge is real and not one lucky
   sample of trades."

Neither technique re-simulates the strategy or re-runs the model — both
are pure resampling over trade outcomes you already have. Cheap.
"""

from __future__ import annotations

import numpy as np


def _equity_curve(returns: np.ndarray, additive: bool) -> np.ndarray:
    if additive:
        return np.cumsum(returns)
    return np.cumprod(1 + returns)


def _max_drawdown(equity: np.ndarray, additive: bool) -> float:
    running_max = np.maximum.accumulate(equity)
    if additive:
        drawdown = equity - running_max
    else:
        safe_max = np.where(running_max == 0, 1e-12, running_max)
        drawdown = equity / safe_max - 1
    return float(drawdown.min()) if len(drawdown) else 0.0


def _longest_losing_streak(returns: np.ndarray) -> int:
    longest = current = 0
    for r in returns:
        if r < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def order_shuffle_path_risk(
    trade_returns: list,
    n_simulations: int = 1000,
    additive: bool = False,
    seed: int = 42,
) -> dict:
    """
    Permute (no replacement) the same trades n_simulations times and
    characterize how much worse the PATH could have looked — the actual
    final return is unchanged by construction and reported once for
    reference, not as a percentile (there's nothing to rank it against).

    Parameters
    ----------
    trade_returns : per-trade returns in original chronological order.
        Fractional/pct returns with additive=False (compounding — the
        realistic default), or raw $ P&L with additive=True.
    n_simulations : number of random reorderings (1000-5000 is plenty).
    additive, seed : see monte_carlo module docstring / above.

    Returns
    -------
    dict with:
        - 'final_return'                  : identical for every ordering (see module docstring)
        - 'actual_max_drawdown'           : from the REAL chronological order
        - 'actual_longest_losing_streak'
        - 'shuffled_max_drawdown_mean/std/p5/p50/p95'
        - 'shuffled_longest_losing_streak_mean/p95'
        - 'drawdown_percentile'           : where actual_max_drawdown ranks
          (0-100; low = actual path was UNUSUALLY SHALLOW/lucky on
          drawdown specifically, i.e. most reorderings were worse)
        - 'interpretation'
    """
    if len(trade_returns) < 5:
        raise ValueError("Need at least 5 trades for a meaningful shuffle test.")

    returns = np.asarray(trade_returns, dtype=float)
    rng = np.random.default_rng(seed)

    actual_equity = _equity_curve(returns, additive)
    actual_dd = _max_drawdown(actual_equity, additive)
    actual_streak = _longest_losing_streak(returns)
    final_return = float(actual_equity[-1])  # same for any permutation — see docstring

    shuffled_dds = np.empty(n_simulations)
    shuffled_streaks = np.empty(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(returns)
        eq = _equity_curve(shuffled, additive)
        shuffled_dds[i] = _max_drawdown(eq, additive)
        shuffled_streaks[i] = _longest_losing_streak(shuffled)

    drawdown_percentile = float((shuffled_dds < actual_dd).mean() * 100)

    if drawdown_percentile < 25:
        dd_note = (
            f"Actual max drawdown ranks in the bottom {drawdown_percentile:.0f}% (shallower "
            "than most reorderings) — this run's loss clustering was unusually favorable. "
            "A materially deeper drawdown is plausible with the exact same edge, just a "
            "different sequence — size positions for the shuffled p95, not the actual result."
        )
    elif drawdown_percentile > 75:
        dd_note = (
            f"Actual max drawdown ranks in the top {drawdown_percentile:.0f}% (deeper than "
            "most reorderings) — this run happened to cluster losses unusually badly."
        )
    else:
        dd_note = "Actual max drawdown is unremarkable relative to how these trades could have sequenced."

    note = (
        f"Final return ({final_return:.4f}) is identical for every possible ordering of these "
        "trades — order-shuffling cannot tell you whether the return itself was lucky, only "
        "whether the PATH to it was. " + dd_note
    )

    return {
        "final_return": final_return,
        "actual_max_drawdown": actual_dd,
        "actual_longest_losing_streak": actual_streak,
        "shuffled_max_drawdown_mean": float(shuffled_dds.mean()),
        "shuffled_max_drawdown_std": float(shuffled_dds.std()),
        "shuffled_max_drawdown_p5": float(np.percentile(shuffled_dds, 5)),
        "shuffled_max_drawdown_p50": float(np.percentile(shuffled_dds, 50)),
        "shuffled_max_drawdown_p95": float(np.percentile(shuffled_dds, 95)),
        "shuffled_longest_losing_streak_mean": float(shuffled_streaks.mean()),
        "shuffled_longest_losing_streak_p95": float(np.percentile(shuffled_streaks, 95)),
        "drawdown_percentile": drawdown_percentile,
        "n_trades": len(returns),
        "n_simulations": n_simulations,
        "interpretation": note,
    }


def bootstrap_return_confidence(
    trade_returns: list,
    n_simulations: int = 2000,
    additive: bool = False,
    seed: int = 42,
) -> dict:
    """
    Resample WITH replacement from the same trade pool (some trades
    repeated, some omitted each draw) to build a genuine distribution of
    plausible final returns from "the kind of trades this strategy
    produces." Unlike order-shuffling, this DOES vary the final return.

    Returns
    -------
    dict with:
        - 'actual_final_return'
        - 'bootstrap_final_return_mean/std/p5/p50/p95'
        - 'probability_of_loss'   : fraction of bootstrap samples with final_return <= break-even
          (a sample-based estimate of "how often does a similar set of
          trades end up net negative" — NOT a formal statistical test,
          but a useful gut-check; treat with more caution the fewer
          trades you have)
        - 'interpretation'
    """
    if len(trade_returns) < 5:
        raise ValueError("Need at least 5 trades for a meaningful bootstrap test.")

    returns = np.asarray(trade_returns, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(returns)

    actual_final = float(_equity_curve(returns, additive)[-1])

    bootstrap_finals = np.empty(n_simulations)
    for i in range(n_simulations):
        sample = rng.choice(returns, size=n, replace=True)
        bootstrap_finals[i] = _equity_curve(sample, additive)[-1]

    baseline = 0.0 if additive else 1.0  # "break-even" final value in each mode
    probability_of_loss = float((bootstrap_finals <= baseline).mean())

    if probability_of_loss > 0.35:
        note = (
            f"{probability_of_loss:.0%} of bootstrap resamples of these trades ended up at or "
            f"below break-even. With only {n} trades, this edge is not well-established yet — "
            "treat any positive headline return as provisional until you have materially more trades."
        )
    elif probability_of_loss > 0.15:
        note = (
            f"{probability_of_loss:.0%} of bootstrap resamples ended at or below break-even — "
            "a plausible edge, but not yet a strong one. Keep collecting trades before sizing up."
        )
    else:
        note = (
            f"Only {probability_of_loss:.0%} of bootstrap resamples ended at or below "
            "break-even — the edge looks reasonably robust to which specific trades happened "
            "to occur, for whatever it's worth with this sample size."
        )

    return {
        "actual_final_return": actual_final,
        "bootstrap_final_return_mean": float(bootstrap_finals.mean()),
        "bootstrap_final_return_std": float(bootstrap_finals.std()),
        "bootstrap_final_return_p5": float(np.percentile(bootstrap_finals, 5)),
        "bootstrap_final_return_p50": float(np.percentile(bootstrap_finals, 50)),
        "bootstrap_final_return_p95": float(np.percentile(bootstrap_finals, 95)),
        "probability_of_loss": probability_of_loss,
        "n_trades": n,
        "n_simulations": n_simulations,
        "interpretation": note,
    }


def monte_carlo_from_closed_trades(closed_trades: list, use_pct: bool = True, **kwargs) -> dict:
    """
    Convenience wrapper: extract returns from paper_trading.models.ClosedTrade
    objects in chronological order, then run BOTH techniques.

    Parameters
    ----------
    closed_trades : list of ClosedTrade.
    use_pct : True -> realized_pnl_pct, additive=False (compounding,
        recommended). False -> raw realized_pnl, additive=True.
    **kwargs : passed through to both underlying functions (n_simulations, seed).
    """
    ordered = sorted(closed_trades, key=lambda t: t.exit_ts)
    additive = not use_pct
    returns = [t.realized_pnl_pct for t in ordered] if use_pct else [t.realized_pnl for t in ordered]

    return {
        "path_risk": order_shuffle_path_risk(returns, additive=additive, **kwargs),
        "return_confidence": bootstrap_return_confidence(returns, additive=additive, **kwargs),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    losses = list(-rng.uniform(0.005, 0.02, 40))
    big_early_win = [0.35]
    trades_ordered = big_early_win + losses  # win first, then a long losing tail

    print("=== Order-shuffle path risk ===")
    path_result = order_shuffle_path_risk(trades_ordered, n_simulations=2000, additive=False)
    print(f"Final return (order-invariant): {path_result['final_return']:.4f}")
    print(f"Actual max drawdown:            {path_result['actual_max_drawdown']:.4f}")
    print(f"Drawdown percentile:            {path_result['drawdown_percentile']:.1f}")
    print(f"\n{path_result['interpretation']}")
    assert path_result["drawdown_percentile"] < 25, "expected win-first ordering to show a favorably shallow drawdown"

    print("\n=== Bootstrap return confidence ===")
    boot_result = bootstrap_return_confidence(trades_ordered, n_simulations=2000, additive=False)
    print(f"Actual final return:   {boot_result['actual_final_return']:.4f}")
    print(f"Probability of loss:   {boot_result['probability_of_loss']:.2%}")
    print(f"\n{boot_result['interpretation']}")
    assert boot_result["probability_of_loss"] > 0.15, "expected high loss-probability when one trade carries the whole return"

    print("\nOK - both techniques behave as expected on a deliberately fragile (one-big-win) trade set.")
