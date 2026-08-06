"""
database/timescaledb.py
------------------------
Idempotent schema setup + write helpers, running against plain
PostgreSQL (no PostgreSQL extension required). The raw DDL lives in
database/schema.sql -- this module applies it and gives ingestion code
a couple of small, safe write helpers, rather than every caller
writing its own INSERT.

Kept as `database/timescaledb.py` (not renamed) so every existing
`from database.timescaledb import ...` call site across the project
keeps working unchanged. If PostgreSQL is installed later, this is
the only file that would need new hypertable-specific helpers added --
callers importing from here wouldn't need to change.
"""

import logging
from pathlib import Path

from sqlalchemy import text

from database.connection import get_engine

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def apply_schema():
    """
    Run schema.sql against the configured database. Safe to call
    repeatedly — every statement in schema.sql uses IF NOT EXISTS /
    if_not_exists, so re-running it is a no-op on an already-initialized
    database rather than an error.
    """
    sql = SCHEMA_PATH.read_text()
    engine = get_engine()
    with engine.begin() as conn:
        # Statements are ;-separated; PostgreSQL functions like
        # create_hypertable() must run in their own statement, not
        # batched with CREATE TABLE, so split naively on ';'. This is
        # fine here because schema.sql is hand-written and doesn't
        # contain semicolons inside string literals/functions.
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    logger.info("Schema applied from %s", SCHEMA_PATH)


def upsert_candles(rows: list[dict]):
    """
    Bulk-insert OHLCV rows, skipping any that already exist (same
    exchange/symbol/timeframe/ts). Safe to call with overlapping data —
    this is what makes historical_loader.py and gap_filler.py safe to
    retry/re-run over the same time range.

    Each row must have keys: exchange, symbol, timeframe, ts, open,
    high, low, close, volume, and optionally is_synthetic.
    """
    if not rows:
        return 0

    engine = get_engine()
    stmt = text(
        """
        INSERT INTO ohlcv (exchange, symbol, timeframe, ts, open, high, low, close, volume, is_synthetic)
        VALUES (:exchange, :symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :is_synthetic)
        ON CONFLICT (exchange, symbol, timeframe, ts) DO NOTHING
        """
    )
    with engine.begin() as conn:
        for row in rows:
            row.setdefault("is_synthetic", False)
        result = conn.execute(stmt, rows)
    return result.rowcount


def update_ingestion_state(exchange: str, symbol: str, timeframe: str, last_synced_ts, status: str = "ok", error: str = None):
    """Record ingestion progress so historical_loader/gap_filler can resume without a full table scan."""
    engine = get_engine()
    stmt = text(
        """
        INSERT INTO ingestion_state (exchange, symbol, timeframe, last_synced_ts, last_synced_at, status, last_error)
        VALUES (:exchange, :symbol, :timeframe, :last_synced_ts, now(), :status, :error)
        ON CONFLICT (exchange, symbol, timeframe)
        DO UPDATE SET last_synced_ts = :last_synced_ts, last_synced_at = now(), status = :status, last_error = :error
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, {
            "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
            "last_synced_ts": last_synced_ts, "status": status, "error": error,
        })
