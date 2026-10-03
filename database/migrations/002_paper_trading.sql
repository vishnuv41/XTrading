-- database/migrations/002_paper_trading.sql
-- ---------------------------------------------------------------------
-- Adds the two tables paper_trading/db_logger.py writes to.
--
-- Design notes:
--   * prediction_log gets ONE row per bar the pipeline evaluates,
--     regardless of whether a trade was executed — this is what lets
--     you audit "the model said X but risk engine blocked it" after
--     the fact. UNIQUE(exchange, symbol, timeframe, ts) makes re-running
--     the engine over the same candle idempotent (ON CONFLICT DO NOTHING).
--   * trade_log is an append-only ledger: one row per OPEN event, one
--     row per CLOSE event, linked by trade_id (uuid4, shared by the
--     pair). Append-only rather than update-in-place because it's a
--     simpler, more auditable pattern for a trading ledger — nothing
--     ever mutates after being written, so there's no risk of a crash
--     mid-update leaving a half-written trade record.
--   * Both tables are safe to run against the same plain-Postgres setup
--     as schema.sql (BRIN on ts for range scans, no TimescaleDB
--     extension required).
-- ---------------------------------------------------------------------

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
