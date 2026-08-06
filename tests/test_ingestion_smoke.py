"""
tests/test_ingestion_smoke.py
--------------------------------
End-to-end smoke test: fake exchange -> clean/validate -> DB write,
using a stubbed exchange and a monkeypatched upsert_candles instead of
a real Binance connection + live PostgreSQL, so this test runs in CI
without external infra. It still exercises the real code path in
ingestion/historical_loader.py, including pagination and the
clean/validate steps.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from exchange.base_exchange import BaseExchange
from ingestion import historical_loader


class FakeExchange(BaseExchange):
    """Returns two pages of candles then an empty page, simulating reaching the end of available history."""
    name = "fake"

    def __init__(self):
        self._pages = [
            [[1704067200000 + i * 60_000, 100 + i, 101 + i, 99 + i, 100.5 + i, 10] for i in range(3)],
            [[1704067380000 + i * 60_000, 103 + i, 104 + i, 102 + i, 103.5 + i, 10] for i in range(3)],
            [],
        ]
        self._call = 0

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        page = self._pages[min(self._call, len(self._pages) - 1)]
        self._call += 1
        return page

    async def fetch_symbols(self):
        return ["BTC/USDT"]

    async def watch_ohlcv(self, symbol, timeframe):
        if False:
            yield None  # pragma: no cover — not exercised by this smoke test

    async def close(self):
        pass


def test_backfill_smoke(monkeypatch):
    written_rows = []

    def fake_upsert(rows):
        written_rows.extend(rows)
        return len(rows)

    def fake_update_state(*args, **kwargs):
        pass

    monkeypatch.setattr(historical_loader, "upsert_candles", fake_upsert)
    monkeypatch.setattr(historical_loader, "update_ingestion_state", fake_update_state)

    exchange = FakeExchange()
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)

    total = asyncio.run(historical_loader.backfill_symbol(exchange, "BTC/USDT", "1m", since, until))

    assert total == 6
    assert len(written_rows) == 6
    # rows must be internally consistent OHLC (validator ran) and ascending by ts
    timestamps = [r["ts"] for r in written_rows]
    assert timestamps == sorted(timestamps)
    for row in written_rows:
        assert row["high"] >= max(row["open"], row["close"], row["low"])
        assert row["low"] <= min(row["open"], row["close"], row["high"])
