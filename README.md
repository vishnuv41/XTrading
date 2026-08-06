# XTrading — Combined Project (Person 1 + Person 2)

## Layout

```
config/, database/, exchange/, ingestion/, preprocessing/, api/   <- Person 1 (data)
indicators/, ml/, regime/, strategy/, risk_engine/, inference/    <- Person 2 (ML)
pipeline/                                                          <- the bridge between them
tests/                                                              <- merged test suite (both sides)
run_training_from_db.py                                            <- real end-to-end entrypoint
```

No package names collide between the two sides (`ml`, `indicators`, `strategy`, etc. vs. `config`,
`database`, `exchange`, etc.), so they sit side by side at the project root rather than nested —
every existing import in both codebases (`from ml.train import ...`, `from config.settings import
settings`, ...) works unchanged.

## The actual workflow (new — this is what "combine" means concretely)

`pipeline/data_loader.py` queries Person 1's PostgreSQL `ohlcv` table and returns a DataFrame in
exactly the shape Person 2's ML pipeline expects (`timestamp, open, high, low, close, volume`,
UTC, ascending) — the same shape `ml.train._make_synthetic_ohlcv()` produced, so it's a drop-in
real-data replacement.

`run_training_from_db.py` is the new entrypoint that chains them:

```bash
# 1. one-time: get real history into the DB
python -m ingestion.historical_loader     # or run scheduler.py for backfill + live stream

# 2. train on it
python run_training_from_db.py --symbol BTC/USDT --timeframe 1h
```

This was verified end-to-end in a sandboxed test (mocked DB layer + stand-in ML libraries, since
neither Postgres nor xgboost/lightgbm/catboost are available in that environment) — argument
parsing, the data contract, `ml.train.run_training_pipeline`, and full artifact persistence
(`ensemble.pkl`, `calibrator.pkl`, `feature_columns.pkl`, `label_map.json`, `metadata.json`) all
ran without error. **You should re-verify on your machine with real data and real libraries**,
the same way you did for each side individually.

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # fill in DB_PASSWORD, Binance keys, etc.
```

Needs a running Postgres+PostgreSQL and Redis instance (not included — see `database/schema.sql`
for the DDL to apply, or `docker-compose.yml` if you asked for one).

## Tests

```powershell
pytest tests/ -v
```

`test_cleaner.py`, `test_validator.py`, `test_resampler.py` need only pandas — no infra.
`test_ingestion_smoke.py` and `test_ml_pipeline.py`/`test_indicators.py` need the full
`requirements.txt` installed (ccxt/sqlalchemy for the former, xgboost/lightgbm/catboost/shap for
the latter) but no live services — both mock or synthesize their own data.

## Known open item — not fixed in this combination

`inference/realtime_pipeline.py` calls `strategy.generate_signal` and several `risk_engine.*`
functions with signatures that don't match those modules' real APIs (flagged earlier, unrelated
to this merge). Live prediction → signal → risk-sizing isn't wired yet; batch training from real
data (above) is. That's the next piece of actual integration work, once you're ready for it.
