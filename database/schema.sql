-- database/schema.sql
-- ---------------------------------------------------------------------
-- Plain PostgreSQL schema for OHLCV storage (no TimescaleDB extension
-- required). Run once against a fresh database (or via
-- database/migrations/ for incremental changes on an existing one).
--
-- Design notes:
--   * One table for all symbols/timeframes/exchanges, distinguished by
--     columns rather than one-table-per-symbol — keeps ingestion and
--     the ML pipeline's queries simple ("give me BTC/USDT 1h between
--     X and Y") without needing to discover table names dynamically.
--   * Primary key is (exchange, symbol, timeframe, ts) so the same
--     candle can never be inserted twice — ingestion can safely retry
--     on failure using ON CONFLICT DO NOTHING / DO UPDATE.
--   * ts is stored as TIMESTAMPTZ (UTC) — see preprocessing/timezone.py
--     for the rule that nothing is ever written here in a non-UTC zone.
--   * No TimescaleDB hypertable/compression here. A BRIN index on ts
--     is used instead — BRIN is built into plain PostgreSQL, costs
--     almost nothing to maintain on append-mostly, time-ordered data
--     like OHLCV candles, and is what gives most of TimescaleDB's
--     time-range-query speed benefit without the extension. If you
--     later install TimescaleDB, converting this table to a hypertable
--     is a one-line addition (SELECT create_hypertable(...)) — nothing
--     else in this schema needs to change.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ohlcv (
    exchange    TEXT        NOT NULL,
    symbol      TEXT        NOT NULL,       -- e.g. 'BTC/USDT'
    timeframe   TEXT        NOT NULL,       -- e.g. '1m', '5m', '1h', '1d'
    ts          TIMESTAMPTZ NOT NULL,       -- candle open time, UTC
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    -- Set true for a candle written by the gap_filler/backfill path
    -- after detecting a hole, vs. a normal live/historical write —
    -- lets Person 2 or a dashboard flag/exclude reconstructed bars
    -- if they ever want to.
    is_synthetic BOOLEAN    NOT NULL DEFAULT FALSE,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exchange, symbol, timeframe, ts)
);

-- Most common query pattern: "latest N candles for symbol X / timeframe Y".
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
    ON ohlcv (symbol, timeframe, ts DESC);

-- Cheap, low-maintenance range-scan index for "candles between date A
-- and date B" queries across the whole table (backtesting, ML training
-- set extraction). BRIN indexes are tiny and nearly free to update on
-- data that arrives roughly in time order, which OHLCV ingestion does.
CREATE INDEX IF NOT EXISTS idx_ohlcv_ts_brin
    ON ohlcv USING BRIN (ts);

-- ---------------------------------------------------------------------
-- Ingestion bookkeeping: tracks the last successfully written candle
-- per (exchange, symbol, timeframe), so historical_loader.py and
-- gap_filler.py know where to resume without re-scanning the whole
-- ohlcv table on every run.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ingestion_state (
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    last_synced_ts  TIMESTAMPTZ,            -- most recent candle confirmed written
    last_synced_at  TIMESTAMPTZ NOT NULL DEFAULT now(),  -- wall-clock time of that sync
    status          TEXT        NOT NULL DEFAULT 'ok',   -- 'ok' | 'error' | 'backfilling'
    last_error      TEXT,
    PRIMARY KEY (exchange, symbol, timeframe)
);

-- ============================================================
-- Paper Trading Tables
-- ============================================================
CREATE TABLE IF NOT EXISTS prediction_log (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,      -- bar timestamp the prediction was made on
    prediction      TEXT        NOT NULL,      -- 'BUY' | 'SELL' | 'HOLD'
    confidence      DOUBLE PRECISION,
    prob_down       DOUBLE PRECISION,
    prob_flat       DOUBLE PRECISION,
    prob_up         DOUBLE PRECISION,
    market_state    TEXT,
    trend           TEXT,
    volatility      TEXT,
    entry_price     DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    position_size   DOUBLE PRECISION,
    risk_pct        DOUBLE PRECISION,
    notional_value  DOUBLE PRECISION,
    risk_block_reason TEXT,                    -- non-null when risk engine downgraded to HOLD
    executed        BOOLEAN     NOT NULL DEFAULT FALSE,  -- did this prediction actually result in a trade_log OPEN row
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exchange, symbol, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_symbol_tf_ts
    ON prediction_log (symbol, timeframe, ts DESC);

CREATE TABLE IF NOT EXISTS trade_log (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        TEXT        NOT NULL,      -- uuid4, shared by the OPEN/CLOSE pair
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    side            TEXT        NOT NULL,      -- 'long' | 'short'
    action          TEXT        NOT NULL,      -- 'OPEN' | 'CLOSE'
    ts              TIMESTAMPTZ NOT NULL,      -- bar timestamp of this event
    price           DOUBLE PRECISION NOT NULL, -- fill price, after simulated slippage
    size            DOUBLE PRECISION NOT NULL, -- position size in asset units
    fee             DOUBLE PRECISION NOT NULL, -- fee paid on this fill (quote currency)
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    exit_reason     TEXT,                      -- NULL on OPEN; 'stop_loss'|'take_profit'|'timeout'|'signal_flip' on CLOSE
    realized_pnl    DOUBLE PRECISION,           -- NULL on OPEN
    realized_pnl_pct DOUBLE PRECISION,          -- NULL on OPEN
    bars_held       INTEGER,                    -- NULL on OPEN
    cash_after      DOUBLE PRECISION NOT NULL,
    equity_after    DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (trade_id, action)
);

CREATE INDEX IF NOT EXISTS idx_trade_log_symbol_ts
    ON trade_log (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_trade_log_trade_id
    ON trade_log (trade_id);