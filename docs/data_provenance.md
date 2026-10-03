# Data Provenance & Schema Specification

## Overview
This document specifies the market data ingestion sources, verification rules, PostgreSQL database schemas, and data flow pipelines in XTrading.

---

## Strategy Baseline Pipeline Flow

```
Binance REST / WS
       │
       ▼
PostgreSQL ohlcv (1m & 1h)
       │
       ▼
66 MTF Causal Feature Engine
       │
       ▼
50/50 XGBoost + CatBoost Continuous-Return Regression
       │
       ▼
250-Prediction Rolling Window Rank Gate (Top-1% / 0.9900)
       │
       ▼
1.5 ATR SL / 3.0 ATR TP / 48-Bar Timeout / 4-Bar Cooldown
       │
       ▼
PostgreSQL trade_log (Completed Closed Trades ONLY)
```

---

## Data Sources & Ingestion Streams

1. **Primary Exchange API**: Binance Public REST & WebSocket (`ccxt`, `ccxtpro`).
2. **Symbols**: `BTC/USDT`, `ETH/USDT`.
3. **Timeframes**: `1m` (high-frequency stream) and `1h` (canonical strategy timeframe).
4. **Resilience**: Automatic retry with exponential backoff on `DDoSProtection` and `RateLimitExceeded` rate-limit errors (`exchange/ccxt_client.py`). Automatic WebSocket reconnect loop (`database/redis_cache.py`).

---

## Quality Verification & Data Cleaning

Every incoming OHLCV candle must pass through two quality control stages prior to database insertion:

1. **Validator (`preprocessing/validator.py`)**:
   - Drops rows containing `None`, `NaN`, or infinite (`inf`) values.
   - Enforces $P_{\text{open}}, P_{\text{high}}, P_{\text{low}}, P_{\text{close}} > 0$ and $V_{\text{volume}} \ge 0$.
   - Verifies intrabar price boundaries: $P_{\text{high}} \ge \max(P_{\text{open}}, P_{\text{close}}, P_{\text{low}})$ and $P_{\text{low}} \le \min(P_{\text{open}}, P_{\text{close}}, P_{\text{high}})$.
2. **Cleaner (`preprocessing/cleaner.py`)**:
   - Deduplicates identical timestamp records (retaining latest).
   - Filters out zero-volume zero-range illiquid ticks.
   - Sorts records chronologically ascending.

---

## Database Schemas (TimescaleDB / PostgreSQL)

### 1. `ohlcv` Table
```sql
CREATE TABLE ohlcv (
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open NUMERIC(18, 8) NOT NULL,
    high NUMERIC(18, 8) NOT NULL,
    low NUMERIC(18, 8) NOT NULL,
    close NUMERIC(18, 8) NOT NULL,
    volume NUMERIC(24, 8) NOT NULL,
    PRIMARY KEY (ts, symbol, timeframe)
);
```

### 2. `prediction_log` Table
```sql
CREATE TABLE prediction_log (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    prediction VARCHAR(10) NOT NULL,
    confidence NUMERIC(10, 6) NOT NULL,
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 3. `trade_log` Table
```sql
CREATE TABLE trade_log (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL, -- 'OPEN' or 'CLOSE'
    ts TIMESTAMPTZ NOT NULL,
    price NUMERIC(18, 8) NOT NULL,
    size NUMERIC(18, 8) NOT NULL,
    fee NUMERIC(18, 8) NOT NULL,
    stop_loss NUMERIC(18, 8),
    take_profit NUMERIC(18, 8),
    exit_reason VARCHAR(20), -- 'stop_loss', 'take_profit', 'timeout'
    realized_pnl NUMERIC(18, 8),
    realized_pnl_pct NUMERIC(10, 6),
    bars_held INTEGER
);
```
