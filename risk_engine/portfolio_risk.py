"""
Portfolio Risk

Everything in stoploss/takeprofit/position_size.py sizes ONE trade in
isolation. This module looks across ALL currently open positions, which
matters for two reasons that per-trade risk management can't catch:

1. Correlation risk: if a strategy is long BTC, ETH, and SOL
   simultaneously, that isn't really three independent 1%-risk bets —
   during a market-wide selloff these move together, so the effective
   risk is closer to one large correlated bet. Sizing each individually
   at "acceptable" risk can still blow up the portfolio if all three
   are opened at once and correlation spikes (which it typically does
   exactly during the drawdowns you're trying to protect against).

2. Portfolio heat: the sum of risk % across all currently open
   positions. Even fully uncorrelated positions can compound into an
   unacceptable total drawdown if too many are open with no aggregate
   cap — 8 positions each risking 1% is a very different risk profile
   from 1 position risking 1%, even before correlation is considered.

Neither of these checks is optional in a system running multiple
symbols concurrently; skipping them is one of the most common ways a
strategy that looks fine in single-symbol backtests fails live once
scaled to a real, multi-asset portfolio.
"""

import pandas as pd
import numpy as np


def calculate_return_correlation(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate the pairwise correlation matrix of log returns across
    multiple assets.

    Args:
        price_dict: Dict mapping symbol -> close price series (must be
            aligned/same-length or share a common index; misaligned
            timestamps across symbols will produce NaN-heavy, unreliable
            correlations — align on a common timestamp index before
            calling this).

    Returns:
        DataFrame correlation matrix (symbols x symbols), values -1 to 1.
    """
    if len(price_dict) < 2:
        raise ValueError("Need at least 2 symbols to calculate correlation")

    returns = pd.DataFrame({
        symbol: np.log(prices / prices.shift(1)) for symbol, prices in price_dict.items()
    })
    return returns.corr()


def get_correlated_symbols(correlation_matrix: pd.DataFrame, symbol: str,
                            threshold: float = 0.7) -> list[str]:
    """
    Return which symbols are highly correlated (positively) with the
    given symbol, above the threshold.

    Args:
        correlation_matrix: Output of calculate_return_correlation().
        symbol: Symbol to check correlations against.
        threshold: Correlation above this counts as "highly correlated"
            (default 0.7 — a common convention; crypto majors are often
            correlated well above this during risk-off moves, which is
            exactly the scenario this check exists for).

    Returns:
        List of symbol names correlated with `symbol` above threshold
        (excludes the symbol itself).
    """
    if symbol not in correlation_matrix.columns:
        raise ValueError(f"'{symbol}' not found in correlation matrix")

    correlations = correlation_matrix[symbol].drop(symbol)
    return correlations[correlations >= threshold].index.tolist()


def calculate_portfolio_heat(open_positions: list[dict]) -> float:
    """
    Sum the risk % across all currently open positions — "how much of
    the account is at risk right now if every stop-loss got hit at once."

    Args:
        open_positions: List of position dicts, each expected to have a
            'risk_pct' key (as produced by
            position_size.calculate_adjusted_position_size's output).

    Returns:
        Total portfolio heat as a fraction (e.g. 0.05 = 5% of equity at
        risk across all open positions combined).
    """
    return sum(pos.get("risk_pct", 0.0) for pos in open_positions)


def check_new_trade_allowed(open_positions: list[dict], new_symbol: str,
                             new_risk_pct: float,
                             correlation_matrix: pd.DataFrame | None = None,
                             max_portfolio_heat: float = 0.06,
                             max_correlation: float = 0.7,
                             max_correlated_positions: int = 2) -> dict:
    """
    The main gate: decide whether a new trade can be opened given
    current portfolio exposure. Combines the portfolio-heat cap and the
    correlation cap into one decision, since either alone misses risk
    the other catches (heat alone ignores that 3 correlated positions
    are riskier together than the sum of their individual risk_pct
    suggests; correlation alone ignores that even uncorrelated positions
    can pile up past a sane total).

    Args:
        open_positions: List of dicts, each with at least 'symbol' and
            'risk_pct' keys (matching position_size.py's output plus a
            symbol tag added by the caller).
        new_symbol: Symbol of the prospective new trade.
        new_risk_pct: Risk % the new trade would add (from
            position_size.calculate_adjusted_position_size).
        correlation_matrix: Optional. If provided, the correlation check
            is applied; if None, only the portfolio-heat check runs
            (useful early on, before enough multi-symbol history exists
            to compute reliable correlations).
        max_portfolio_heat: Maximum total risk % allowed across all open
            positions including the new one (default 0.06 = 6%, a
            common conservative ceiling — e.g. 6 positions at 1% each,
            or fewer at higher individual risk).
        max_correlation: Correlation threshold for "highly correlated"
            (default 0.7, matching get_correlated_symbols' default).
        max_correlated_positions: Max number of ALREADY-OPEN positions
            allowed to be highly correlated with the new symbol before
            blocking it (default 2 — i.e. a 3rd correlated position in
            the same direction gets blocked).

    Returns:
        dict with keys:
          'allowed'          - bool, final decision
          'reason'           - str explaining the decision
          'current_heat'     - portfolio heat before this trade
          'heat_after_trade' - portfolio heat if this trade is added
          'correlated_open_positions' - list of open symbols correlated
                                         with new_symbol (empty if no
                                         correlation_matrix provided)
    """
    current_heat = calculate_portfolio_heat(open_positions)
    heat_after = current_heat + new_risk_pct

    if heat_after > max_portfolio_heat:
        return {
            "allowed": False,
            "reason": f"Portfolio heat would reach {heat_after:.2%}, "
                      f"exceeding max {max_portfolio_heat:.2%}",
            "current_heat": current_heat,
            "heat_after_trade": heat_after,
            "correlated_open_positions": [],
        }

    correlated_open = []
    if correlation_matrix is not None and new_symbol in correlation_matrix.columns:
        correlated_symbols = set(get_correlated_symbols(correlation_matrix, new_symbol, max_correlation))
        correlated_open = [
            pos["symbol"] for pos in open_positions
            if pos.get("symbol") in correlated_symbols
        ]

        if len(correlated_open) >= max_correlated_positions:
            return {
                "allowed": False,
                "reason": (
                    f"'{new_symbol}' is highly correlated (>= {max_correlation}) with "
                    f"{len(correlated_open)} already-open position(s) "
                    f"({', '.join(correlated_open)}), exceeding max "
                    f"{max_correlated_positions}"
                ),
                "current_heat": current_heat,
                "heat_after_trade": heat_after,
                "correlated_open_positions": correlated_open,
            }

    return {
        "allowed": True,
        "reason": "Within portfolio heat and correlation limits",
        "current_heat": current_heat,
        "heat_after_trade": heat_after,
        "correlated_open_positions": correlated_open,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300

    # Simulate 3 correlated crypto-like assets + 1 independent one
    common_factor = np.cumsum(rng.normal(0, 1, n))
    btc = 100 + common_factor + np.cumsum(rng.normal(0, 0.3, n))
    eth = 50 + common_factor * 0.9 + np.cumsum(rng.normal(0, 0.3, n))
    sol = 20 + common_factor * 0.8 + np.cumsum(rng.normal(0, 0.4, n))
    independent = 200 + np.cumsum(rng.normal(0, 1, n))

    price_dict = {
        "BTC": pd.Series(btc), "ETH": pd.Series(eth),
        "SOL": pd.Series(sol), "XYZ": pd.Series(independent),
    }
    corr = calculate_return_correlation(price_dict)
    print("Correlation matrix:")
    print(corr.round(2))

    print("\nSymbols correlated with BTC (>= 0.7):", get_correlated_symbols(corr, "BTC", 0.7))

    open_positions = [
        {"symbol": "BTC", "risk_pct": 0.015},
        {"symbol": "ETH", "risk_pct": 0.015},
    ]
    print(f"\nCurrent portfolio heat: {calculate_portfolio_heat(open_positions):.2%}")

    decision = check_new_trade_allowed(
        open_positions, new_symbol="SOL", new_risk_pct=0.015,
        correlation_matrix=corr, max_portfolio_heat=0.06,
        max_correlated_positions=2,
    )
    print("\nRequest to open SOL (correlated with BTC/ETH):")
    for k, v in decision.items():
        print(f"  {k}: {v}")

    decision2 = check_new_trade_allowed(
        open_positions, new_symbol="XYZ", new_risk_pct=0.015,
        correlation_matrix=corr, max_portfolio_heat=0.06,
        max_correlated_positions=2,
    )
    print("\nRequest to open XYZ (independent):")
    for k, v in decision2.items():
        print(f"  {k}: {v}")