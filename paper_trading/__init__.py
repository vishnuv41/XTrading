"""
Paper Trading Engine

Modules
-------
config.py
models.py
portfolio.py
execution.py
exit_manager.py
metrics.py
db_logger.py
run_paper_trading.py
"""

from paper_trading.config import settings
from paper_trading.execution import ExecutionSimulator
from paper_trading.exit_manager import check_exit
from paper_trading.metrics import (
    compute_metrics,
    compute_equity_curve_series,
    print_summary,
)
from paper_trading.models import (
    ClosedTrade,
    EquitySnapshot,
    OpenPosition,
    PredictionRecord,
)
from paper_trading.portfolio import VirtualPortfolio

__all__ = [
    "settings",
    "ExecutionSimulator",
    "check_exit",
    "compute_metrics",
    "compute_equity_curve_series",
    "print_summary",
    "OpenPosition",
    "ClosedTrade",
    "PredictionRecord",
    "EquitySnapshot",
    "VirtualPortfolio",
]