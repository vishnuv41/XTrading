"""
pipeline/data_loader.py
--------------------------
The actual seam between Person 1 and Person 2: reads OHLCV out of
Person 1's PostgreSQL (database/) and hands back a DataFrame in
exactly the shape Person 2's ML pipeline expects (see
ml/utils/preprocessing.py and ml/train._make_synthetic_ohlcv, which
this is a drop-in real-data replacement for):

    columns: timestamp, open, high, low, close, volume
    sorted ascending by timestamp

This intentionally does NOT go through api/market_api.py's HTTP route
function — that endpoint is for external callers (dashboards, Person 3)
and raises HTTPException on no-data, which isn't the right failure mode
for a training script. This talks to the DB directly via
database.connection, the same way market_api.py does internally, so
there's one query implementation but two consumption paths (HTTP vs.
in-process DataFrame).
"""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine

logger = logging.getLogger(__name__)


def load_ohlcv(
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    start: datetime = None,
    end: datetime = None,
    limit: int = None,
) -> pd.DataFrame:
    """
    Load OHLCV history for symbol/timeframe as a DataFrame ready for
    ml.train.run_training_pipeline / ml.predict.predict.

    Raises ValueError (not HTTPException) if no rows are found, since
    this is meant for scripts/notebooks, not an HTTP handler.
    """
    engine = get_engine()
    query = """
        SELECT ts, open, high, low, close, volume FROM ohlcv
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
    """
    params = {"exchange": exchange, "symbol": symbol, "timeframe": timeframe}

    # Strict Closed-Candle Rule: Exclude forming candles (ts + 1h > NOW()) unless explicit end is supplied
    if not end:
        # Default 1h duration guard
        dur_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 3600)
        from datetime import datetime, timezone, timedelta
        max_ts = datetime.now(timezone.utc) - timedelta(seconds=dur_seconds)
        query += " AND ts <= :max_closed_ts"
        params["max_closed_ts"] = max_ts

    if start:
        query += " AND ts >= :start"
        params["start"] = start
    if end:
        query += " AND ts <= :end"
        params["end"] = end
    if limit and not start:
        query += " ORDER BY ts DESC LIMIT :limit"
        params["limit"] = limit
    else:
        query += " ORDER BY ts ASC"
        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    if not rows:
        raise ValueError(
            f"No OHLCV data for {symbol} {timeframe} on {exchange}. "
            "Has ingestion/historical_loader.backfill_symbol() been run for this symbol/timeframe yet?"
        )

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if limit and not start:
        df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows for %s %s (%s) from %s to %s",
                len(df), symbol, timeframe, exchange, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return df


def load_ohlcv_multi(symbols: list, timeframe: str, exchange: str = "binance", **kwargs) -> dict:
    """Convenience: load several symbols at once (e.g. a base symbol plus BTC for ml.features.cross_asset_features). Returns {symbol: DataFrame}."""
    return {symbol: load_ohlcv(symbol, timeframe, exchange=exchange, **kwargs) for symbol in symbols}
