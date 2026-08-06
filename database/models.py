"""
database/models.py
--------------------
SQLAlchemy ORM models mirroring database/schema.sql. These are
optional-use — ingestion's hot path uses timescaledb.upsert_candles()
(raw SQL, faster for bulk writes), but these give api/market_api.py and
any ad-hoc scripts a typed, query-friendly interface via
database/connection.get_session_factory().

Keep this in sync with schema.sql manually — schema.sql is the source
of truth for what's actually applied to the DB (via migrations), this
is a read-oriented mirror of it.
"""

from sqlalchemy import Boolean, Column, DateTime, Float, PrimaryKeyConstraint, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class OHLCV(Base):
    __tablename__ = "ohlcv"

    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    is_synthetic = Column(Boolean, nullable=False, default=False)
    inserted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        PrimaryKeyConstraint("exchange", "symbol", "timeframe", "ts"),
    )

    def __repr__(self):
        return f"<OHLCV {self.exchange}:{self.symbol}:{self.timeframe} @ {self.ts}>"


class IngestionState(Base):
    __tablename__ = "ingestion_state"

    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    last_synced_ts = Column(DateTime(timezone=True))
    last_synced_at = Column(DateTime(timezone=True))
    status = Column(String, nullable=False, default="ok")
    last_error = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("exchange", "symbol", "timeframe"),
    )
