"""
experiments/tracker.py
--------------------------
Lightweight experiment record (Part K of the architecture review).
Purpose: every backtest/paper-trading run that matters gets ONE row
recording what was run and what happened, so repeated tweaking doesn't
turn into "did that help? I don't actually know anymore." Deliberately
NOT a database table or a run-tracking service (MLflow-style) — this
project already has PostgreSQL for trade/prediction logs
(paper_trading/db_logger.py); duplicating that here would be exactly
the kind of unnecessary abstraction the review brief asked to avoid.
Append-only JSONL is enough for "compare experiment N to experiment M."

Usage:
    from experiments.tracker import ExperimentRecord, log_experiment, load_experiments

    log_experiment(ExperimentRecord(
        experiment_id="btc_1h_v3_wider_barriers",
        strategy_version="regime_adaptive+fusion_v1",
        feature_version="ml/features@2024-xx-xx",
        model_version="models_artifacts/BTCUSDT_1h",
        symbol="BTC/USDT", timeframe="1h",
        train_period="2022-01-01:2023-06-30", validation_period="2023-07-01:2023-10-31",
        test_period="2023-11-01:2023-12-31",
        fees_bps=10.0, slippage_bps=5.0, leverage=3.0, risk_pct=0.01,
        num_trades=57, win_rate=0.579, profit_factor=1.18, expectancy=0.021,
        max_drawdown=-0.084, total_return=0.0437, sharpe=1.22,
    ))

    df = load_experiments()  # -> pd.DataFrame, one row per experiment, for comparison
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

DEFAULT_LOG_PATH = os.getenv("EXPERIMENT_LOG_PATH", "experiments/experiment_log.jsonl")


@dataclass
class ExperimentRecord:
    experiment_id: str
    strategy_version: str
    feature_version: str
    model_version: str
    symbol: str
    timeframe: str

    train_period: Optional[str] = None
    validation_period: Optional[str] = None
    test_period: Optional[str] = None

    fees_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    leverage: Optional[float] = None
    risk_pct: Optional[float] = None

    num_trades: Optional[int] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    expectancy: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    sharpe: Optional[float] = None

    notes: Optional[str] = None
    logged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def log_experiment(record: ExperimentRecord, path: str = DEFAULT_LOG_PATH) -> None:
    """Append one experiment record as a JSON line. Creates the file/dir if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_experiments(path: str = DEFAULT_LOG_PATH):
    """Load all logged experiments as a pandas DataFrame, sorted newest-first, for comparison."""
    import pandas as pd

    if not os.path.exists(path):
        return pd.DataFrame(columns=[f.name for f in ExperimentRecord.__dataclass_fields__.values()])

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if "logged_at" in df.columns and len(df):
        df = df.sort_values("logged_at", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    tmp_path = "/tmp/experiment_log_smoke_test.jsonl"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    log_experiment(
        ExperimentRecord(
            experiment_id="smoke_test_1", strategy_version="v1", feature_version="v1",
            model_version="v1", symbol="BTC/USDT", timeframe="1h",
            num_trades=10, win_rate=0.5, profit_factor=1.1, total_return=0.02, sharpe=0.9,
        ),
        path=tmp_path,
    )
    df = load_experiments(tmp_path)
    print(df.to_string())
    assert len(df) == 1
    print("\nOK.")
