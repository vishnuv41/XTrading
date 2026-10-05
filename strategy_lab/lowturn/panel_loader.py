"""
strategy_lab/lowturn/panel_loader.py
-----------------------------------
Loads multi-asset panels for the low-turnover benchmark suite.
Universe: BTC, ETH, BNB, XRP, ADA, LTC, SOL, DOGE, LINK.
Timeframes: 1D (primary), 4H (intermediate).

Ensures strict chronological partitioning:
- Development Window: First 75% of timeline (2020-08-11 to 2025-03-24)
- Reserved Holdout Window: Last 25% of timeline (2025-03-25 to 2026-10-05) + post 2026-09-11
"""

from datetime import datetime, timezone
from typing import Dict, Tuple, List, Optional
import pandas as pd
from sqlalchemy import text

from database.connection import get_engine

UNIVERSE_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LTC/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
]


def load_symbol_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load clean, sorted OHLCV dataframe for a single symbol from the database."""
    engine = get_engine()
    query = """
        SELECT ts, open, high, low, close, volume
        FROM ohlcv
        WHERE exchange = 'binance' AND symbol = :symbol AND timeframe = :timeframe
    """
    params = {"symbol": symbol, "timeframe": timeframe}
    if start_ts is not None:
        query += " AND ts >= :start_ts"
        params["start_ts"] = start_ts
    if end_ts is not None:
        query += " AND ts <= :end_ts"
        params["end_ts"] = end_ts

    query += " ORDER BY ts ASC"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        raise ValueError(f"No OHLCV rows found for {symbol} {timeframe}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def load_panel_universe(
    timeframe: str = "1d",
    symbols: Optional[List[str]] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV dataframes for all symbols in universe."""
    target_symbols = symbols or UNIVERSE_SYMBOLS
    panel = {}
    for sym in target_symbols:
        df = load_symbol_ohlcv(sym, timeframe=timeframe, start_ts=start_ts, end_ts=end_ts)
        panel[sym] = df
    return panel


def get_timeline_split_dates(common_timeframe: str = "1d") -> Tuple[datetime, datetime, datetime, datetime]:
    """
    Compute exact timeline dates across common universe (starts at SOL listing 2020-08-11).
    Returns (universe_start, dev_end, holdout_start, holdout_end).
    """
    sol_df = load_symbol_ohlcv("SOL/USDT", timeframe=common_timeframe)
    universe_start = sol_df.index[0].to_pydatetime()
    universe_end = sol_df.index[-1].to_pydatetime()

    total_duration = universe_end - universe_start
    dev_duration = total_duration * 0.75
    dev_end = universe_start + dev_duration
    holdout_start = dev_end

    return universe_start, dev_end, holdout_start, universe_end


def load_dev_and_holdout_panels(
    timeframe: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], dict]:
    """
    Loads separate development and holdout panels for the universe.
    Returns:
        (dev_panel, holdout_panel, split_metadata)
    """
    start_ts, dev_end, holdout_start, end_ts = get_timeline_split_dates(timeframe)

    dev_panel = load_panel_universe(timeframe=timeframe, start_ts=start_ts, end_ts=dev_end)
    holdout_panel = load_panel_universe(timeframe=timeframe, start_ts=holdout_start, end_ts=end_ts)

    meta = {
        "start_ts": start_ts.isoformat(),
        "dev_end": dev_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "end_ts": end_ts.isoformat(),
        "timeframe": timeframe,
        "n_symbols": len(UNIVERSE_SYMBOLS),
    }

    return dev_panel, holdout_panel, meta
