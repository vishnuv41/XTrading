"""
Simple end-to-end paper trading test.

Run:

python -m paper_trading.test_paper_trading
"""

from datetime import datetime

from paper_trading.execution import ExecutionSimulator
from paper_trading.metrics import print_summary
from paper_trading.portfolio import VirtualPortfolio
from paper_trading.config import settings


def main():
    portfolio = VirtualPortfolio(settings.starting_cash)
    simulator = ExecutionSimulator()

    ts = datetime.utcnow()

    # ----------------------------------------------------------
    # Open a BUY
    # ----------------------------------------------------------

    risk = {
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "position_size": 10.0,
        "risk_pct": 0.01,
    }

    position = simulator.open_from_prediction(
        portfolio=portfolio,
        symbol="BTC/USDT",
        timeframe="1h",
        ts=ts,
        prediction="BUY",
        risk=risk,
    )

    assert position is not None

    portfolio.mark_to_market(ts, {"BTC/USDT": 101.0})

    # ----------------------------------------------------------
    # Trigger TP
    # ----------------------------------------------------------

    trade = simulator.check_and_close(
        portfolio=portfolio,
        symbol="BTC/USDT",
        ts=ts,
        bar_high=112.0,
        bar_low=99.0,
        bar_close=111.0,
    )

    assert trade is not None

    portfolio.mark_to_market(ts, {})

    print_summary(portfolio)


if __name__ == "__main__":
    main()