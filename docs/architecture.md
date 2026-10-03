# System Architecture Specification

## Overview
XTrading is a modular, high-reliability automated quantitative trading and paper-trading platform built for multi-timeframe cryptocurrency markets. It decouples market data ingestion, feature engineering, probabilistic signal generation, risk-managed order execution, state persistence, and observability into distinct, robust components.

```
                         ┌───────────────────────────┐
                         │   Binance REST / WS API   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Ingestion & Gap Cleaner  │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   TimescaleDB / Postgres  │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
 ┌───────────────────┐   ┌───────────────────────────┐   ┌───────────────────┐
 │ Risk Engine & ATR ├──►│    Paper Trading Engine   │◄──┤  XGBoost + CatB   │
 │   Barrier Sizer   │   │     (Portfolio State)     │   │ Probability Model │
 └───────────────────┘   └─────────────┬─────────────┘   └───────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Observability / Watchdog│
                         │    FastAPI Dashboard UI   │
                         └───────────────────────────┘
```

---

## Core Components

### 1. Ingestion Engine (`ingestion/`)
- **Modules**: `historical_loader.py`, `gap_filler.py`, `run_live_ingestion.py`
- **Function**: Continuously pulls 1m and 1h OHLCV kline updates from Binance.
- **Validation**: Enforces strict bar integrity check (`preprocessing/validator.py`) and zero-volume deduplication (`preprocessing/cleaner.py`).

### 2. Feature Engineering & Pipelines (`enhanced_ml/`, `inference/`)
- **Modules**: `realtime_pipeline.py`, `feature_engineering.py`
- **Function**: Computes 66 multi-timeframe features (RSI, MACD, ATR, Supertrend, Volatility Ratios, EMA Spreads, Multi-period Return Momentum) across 1m and 1h windows.
- **Resilience**: Strips invalid NaNs/inf values and falls back gracefully to `HOLD` decision on insufficient history ($N < 10$).

### 3. Machine Learning Inference (`paper_trading/engine.py`)
- **Model**: Ensembled XGBoost + CatBoost classifier (50/50 probability blending).
- **Signal Logic**: Converts raw predicted probabilities to percentile scores using historical rolling windows (`0.9900` percentile rank gate).

### 4. Risk Engine & Exit Management (`paper_trading/execution.py`, `paper_trading/exit_manager.py`)
- **Position Sizing**: Risk-heat capital allocation (default 1.0% account risk heat).
- **ATR Barriers**: Dynamically sets Stop Loss (1.5x ATR) and Take Profit (3.0x ATR).
- **Intrabar Barrier Priority**: Resolves single-candle SL/TP touch ambiguity by prioritizing Stop Loss defensively.
- **Timeout Exit**: Enforces 48-bar maximum trade duration limit.

### 5. Virtual Portfolio & Persistence (`paper_trading/portfolio.py`, `database/`)
- **Portfolio Model**: Maintains virtual cash, open positions, closed trade logs, and equity curve snapshots.
- **Database Pooling**: Uses SQLAlchemy sync connection pool (`pool_recycle=3600`, `with_db_retry`) and `asyncpg` connection pool.

### 6. Process Watchdog & Observability (`experiments/watchdog.py`, `experiments/dashboard.py`)
- **Watchdog**: Periodically verifies ingestion heartbeats, candle freshness, prediction timestamps, and engine health.
- **Analytics Dashboard**: FastAPI web app (`/`) and CLI utility (`--cli`) reporting equity curve, win/loss stats, drawdown, asset breakdown, and trade logs.
