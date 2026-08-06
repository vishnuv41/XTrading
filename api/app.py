"""
api/app.py
------------
FastAPI app entrypoint tying market_api.py and websocket_api.py
together. Run with: `uvicorn api.app:app --host 0.0.0.0 --port 8001`.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.market_api import router as market_router
from api.websocket_api import router as ws_router
from database.connection import close_async_pool
from database.timescaledb import apply_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_schema()  # idempotent — safe to run on every startup
    yield
    await close_async_pool()


app = FastAPI(title="XTrading Market Data API", lifespan=lifespan)
app.include_router(market_router)
app.include_router(ws_router)


@app.get("/health")
def health():
    return {"status": "ok"}
