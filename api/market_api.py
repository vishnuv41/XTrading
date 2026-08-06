"""
api/market_api.py
--------------------
REST endpoints Person 2 (training/backtesting) and Person 3 (dashboard)
consume market data through. This is the one supported way to read
OHLCV data from outside person1/ — callers shouldn't query PostgreSQL
directly, so the schema can change without breaking downstream code.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from database.connection import get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"])


@router.get("/symbols")
def get_symbols(exchange: str = Query(default=None)):
    """List distinct symbols currently stored, optionally filtered by exchange."""
    engine = get_engine()
    query = "SELECT DISTINCT symbol, exchange FROM ohlcv"
    params = {}
    if exchange:
        query += " WHERE exchange = :exchange"
        params["exchange"] = exchange

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [{"symbol": r[0], "exchange": r[1]} for r in rows]


@router.get("/ohlcv")
def get_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime = Query(default=None),
    end: datetime = Query(default=None),
    exchange: str = Query(default="binance"),
    limit: int = Query(default=1000, le=10000),
):
    """
    Return OHLCV rows for symbol/timeframe, ascending by time. This is
    the endpoint Person 2's data-loading step points at instead of
    _make_synthetic_ohlcv() once real data is available.
    """
    engine = get_engine()
    query = """
        SELECT ts, open, high, low, close, volume FROM ohlcv
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
    """
    params = {"exchange": exchange, "symbol": symbol, "timeframe": timeframe}
    if start:
        query += " AND ts >= :start"
        params["start"] = start
    if end:
        query += " AND ts <= :end"
        params["end"] = end
    query += " ORDER BY ts ASC LIMIT :limit"
    params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol} {timeframe} on {exchange}")

    return [
        {"timestamp": r[0].isoformat(), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in rows
    ]


@router.get("/latest")
def get_latest(symbol: str, timeframe: str, exchange: str = Query(default="binance")):
    """Return the single most recent candle for symbol/timeframe (ORDER BY ts DESC, not get_ohlcv's ascending order, so LIMIT 1 is actually the latest)."""
    engine = get_engine()
    query = text("""
        SELECT ts, open, high, low, close, volume FROM ohlcv
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        ORDER BY ts DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"exchange": exchange, "symbol": symbol, "timeframe": timeframe}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No data for {symbol} {timeframe} on {exchange}")
    return {"timestamp": row[0].isoformat(), "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5]}
