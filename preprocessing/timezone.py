"""
preprocessing/timezone.py
----------------------------
Project-wide rule: every timestamp stored or passed downstream is a
timezone-AWARE UTC datetime. Never naive, never local. This module is
the one place that rule is enforced/converted, so nothing downstream
(database writes, Person 2's ML pipeline) has to think about timezones
again.
"""

from datetime import datetime, timezone

import pandas as pd


def to_utc(ts) -> datetime:
    """
    Normalize a timestamp (datetime, pandas Timestamp, ISO string, or
    ms-since-epoch int) to a timezone-aware UTC datetime.
    """
    if isinstance(ts, (int, float)):
        # Heuristic: treat as milliseconds if it's large enough to be a
        # ms-epoch value (exchange APIs return ms), else seconds.
        seconds = ts / 1000 if ts > 10**12 else ts
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    if isinstance(ts, str):
        ts = pd.Timestamp(ts)

    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()

    if not isinstance(ts, datetime):
        raise TypeError(f"Cannot convert {type(ts)} to a UTC datetime")

    if ts.tzinfo is None:
        # Naive datetimes are assumed to already be UTC (not local) —
        # every writer in this codebase (exchange responses, DB reads)
        # should be producing UTC already; this just attaches the tzinfo
        # rather than silently reinterpreting the wall-clock value.
        return ts.replace(tzinfo=timezone.utc)

    return ts.astimezone(timezone.utc)


def ensure_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Ensure a DataFrame's timestamp column is tz-aware UTC and sorted ascending."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df
