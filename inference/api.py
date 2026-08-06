"""
Trading AI REST API

Person 2 - Quantitative Trading & AI

This API exposes the complete inference pipeline.

Pipeline

Incoming OHLCV
        ↓
Feature Engineering
        ↓
Indicators
        ↓
Market Regime
        ↓
Strategy
        ↓
Risk Engine
        ↓
ML Prediction
        ↓
Probability Calibration
        ↓
Final Response

Run:

uvicorn inference.api:app --reload

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.realtime_pipeline import run_realtime_pipeline

# ----------------------------------------------------
# Logging
# ----------------------------------------------------

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("TradingAPI")

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------

app = FastAPI(

    title="Trading Intelligence API",

    description="AI Powered Crypto Trading Engine",

    version="1.0.0"

)

# ----------------------------------------------------
# Globals
# ----------------------------------------------------

MODEL = None

CALIBRATOR = None

FEATURE_COLUMNS = None

MODEL_DIRECTORY = Path("models")

# ----------------------------------------------------
# Request Models
# ----------------------------------------------------

class Candle(BaseModel):

    timestamp: int

    open: float

    high: float

    low: float

    close: float

    volume: float


class PredictionRequest(BaseModel):

    symbol: str = Field(

        ...,

        example="BTCUSDT"

    )

    timeframe: str = Field(

        ...,

        example="15m"

    )

    candles: List[Candle]


# ----------------------------------------------------
# Response Model
# ----------------------------------------------------

class PredictionResponse(BaseModel):

    symbol: str

    timeframe: str

    prediction: str

    confidence: float

    market_state: str

    trend: str

    volatility: str

    entry: Optional[float]

    stop_loss: Optional[float]

    take_profit: Optional[float]

    explanation: List[str]


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def candles_to_dataframe(

    request: PredictionRequest

) -> pd.DataFrame:

    rows = []

    for candle in request.candles:

        rows.append({

            "timestamp": candle.timestamp,

            "open": candle.open,

            "high": candle.high,

            "low": candle.low,

            "close": candle.close,

            "volume": candle.volume

        })

    df = pd.DataFrame(rows)

    return df


# ----------------------------------------------------
# Startup
# ----------------------------------------------------

@app.on_event("startup")

def startup_event():

    global MODEL

    global CALIBRATOR

    global FEATURE_COLUMNS

    logger.info("Loading models...")

    model_file = MODEL_DIRECTORY / "ensemble.pkl"

    calibrator_file = MODEL_DIRECTORY / "calibrator.pkl"

    features_file = MODEL_DIRECTORY / "feature_columns.pkl"

    if not model_file.exists():

        logger.warning(

            "No trained model found."

        )

        return

    MODEL = joblib.load(model_file)

    logger.info("Model Loaded.")

    if calibrator_file.exists():

        CALIBRATOR = joblib.load(calibrator_file)

        logger.info("Calibrator Loaded.")

    if features_file.exists():

        FEATURE_COLUMNS = joblib.load(features_file)

        logger.info(

            f"{len(FEATURE_COLUMNS)} Features Loaded."

        )


# ----------------------------------------------------
# Root
# ----------------------------------------------------

@app.get("/")

def root():

    return {

        "application": "Trading Intelligence API",

        "status": "running",

        "version": "1.0.0"

    }


# ----------------------------------------------------
# Health
# ----------------------------------------------------

@app.get("/health")

def health():

    return {

        "status": "healthy",

        "model_loaded": MODEL is not None,

        "calibrator_loaded": CALIBRATOR is not None,

        "feature_count":

        len(FEATURE_COLUMNS)

        if FEATURE_COLUMNS is not None

        else 0

    }
# ----------------------------------------------------
# Prediction Endpoint
# ----------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    global MODEL
    global CALIBRATOR
    global FEATURE_COLUMNS

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model has been loaded."
        )

    try:

        logger.info(
            f"Prediction request received for "
            f"{request.symbol} ({request.timeframe})"
        )

        # -----------------------------
        # Convert candles to DataFrame
        # -----------------------------

        df = candles_to_dataframe(request)

        # -----------------------------
        # Run realtime pipeline
        # -----------------------------

        result = run_realtime_pipeline(

            df=df,

            model=MODEL,

            calibrator=CALIBRATOR,

            feature_columns=FEATURE_COLUMNS

        )

        logger.info(

            "Prediction completed successfully."

        )

        return PredictionResponse(

            symbol=request.symbol,

            timeframe=request.timeframe,

            prediction=result["prediction"],

            confidence=float(result["confidence"]),

            market_state=result["market_state"],

            trend=result["trend"],

            volatility=result["volatility"],

            entry=result.get("entry"),

            stop_loss=result.get("stop_loss"),

            take_profit=result.get("take_profit"),

            explanation=result.get("explanation", [])

        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ----------------------------------------------------
# Market State Endpoint
# ----------------------------------------------------

@app.post("/market-state")
def market_state(request: PredictionRequest):

    if MODEL is None:

        raise HTTPException(

            status_code=503,

            detail="Model not loaded"

        )

    df = candles_to_dataframe(request)

    result = run_realtime_pipeline(

        df=df,

        model=MODEL,

        calibrator=CALIBRATOR,

        feature_columns=FEATURE_COLUMNS

    )

    return {

        "symbol": request.symbol,

        "timeframe": request.timeframe,

        "market_state": result["market_state"],

        "trend": result["trend"],

        "volatility": result["volatility"]

    }


# ----------------------------------------------------
# Features Endpoint
# ----------------------------------------------------

@app.post("/features")
def features(request: PredictionRequest):

    if MODEL is None:

        raise HTTPException(

            status_code=503,

            detail="Model not loaded"

        )

    df = candles_to_dataframe(request)

    result = run_realtime_pipeline(

        df=df,

        model=MODEL,

        calibrator=CALIBRATOR,

        feature_columns=FEATURE_COLUMNS,

        return_features=True

    )

    return {

        "rows": len(result["features"]),

        "columns": list(result["features"].columns),

        "features": result["features"].tail(5).to_dict(
            orient="records"
        )

    }


# ----------------------------------------------------
# Model Information
# ----------------------------------------------------

@app.get("/model-info")
def model_info():

    if MODEL is None:

        return {

            "loaded": False

        }

    return {

        "loaded": True,

        "calibrator": CALIBRATOR is not None,

        "feature_count":

            len(FEATURE_COLUMNS)

            if FEATURE_COLUMNS is not None

            else 0

    }


# ----------------------------------------------------
# Supported Symbols
# ----------------------------------------------------

@app.get("/symbols")
def supported_symbols():

    return {

        "symbols": [

            "BTCUSDT",

            "ETHUSDT",

            "BNBUSDT",

            "SOLUSDT",

            "XRPUSDT",

            "ADAUSDT"

        ]

    }


# ----------------------------------------------------
# Supported Timeframes
# ----------------------------------------------------

@app.get("/timeframes")
def supported_timeframes():

    return {

        "timeframes": [

            "1m",

            "5m",

            "15m",

            "1h",

            "4h",

            "1d"

        ]

    }
# ----------------------------------------------------
# Middleware
# ----------------------------------------------------

from fastapi import Request
from fastapi.responses import JSONResponse
import time


@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()

    response = await call_next(request)

    duration = round(time.time() - start_time, 4)

    logger.info(
        f"{request.method} "
        f"{request.url.path} "
        f"{response.status_code} "
        f"{duration}s"
    )

    response.headers["X-Process-Time"] = str(duration)

    return response


# ----------------------------------------------------
# Global Exception Handler
# ----------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(exc)

    return JSONResponse(

        status_code=500,

        content={

            "success": False,

            "error": str(exc)

        }

    )


# ----------------------------------------------------
# Custom Error Responses
# ----------------------------------------------------

@app.get("/ping")
def ping():

    return {

        "success": True,

        "message": "pong"

    }


@app.get("/version")
def version():

    return {

        "api": "Trading Intelligence API",

        "version": "1.0.0"

    }


@app.get("/status")
def status():

    return {

        "model_loaded": MODEL is not None,

        "calibrator_loaded": CALIBRATOR is not None,

        "feature_columns":

            len(FEATURE_COLUMNS)

            if FEATURE_COLUMNS is not None

            else 0

    }


# ----------------------------------------------------
# Reload Models
# ----------------------------------------------------

@app.post("/reload")
def reload_models():

    global MODEL
    global CALIBRATOR
    global FEATURE_COLUMNS

    logger.info("Reloading models...")

    try:

        model_file = MODEL_DIRECTORY / "ensemble.pkl"

        calibrator_file = MODEL_DIRECTORY / "calibrator.pkl"

        features_file = MODEL_DIRECTORY / "feature_columns.pkl"

        MODEL = joblib.load(model_file)

        if calibrator_file.exists():

            CALIBRATOR = joblib.load(calibrator_file)

        if features_file.exists():

            FEATURE_COLUMNS = joblib.load(features_file)

        return {

            "success": True,

            "message": "Models reloaded"

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ----------------------------------------------------
# Shutdown Event
# ----------------------------------------------------

@app.on_event("shutdown")
def shutdown_event():

    logger.info(

        "Trading API shutting down."

    )


# ----------------------------------------------------
# Startup Banner
# ----------------------------------------------------

@app.on_event("startup")
def startup_banner():

    logger.info("=" * 60)

    logger.info("Trading Intelligence API Started")

    logger.info("=" * 60)

    # ----------------------------------------------------
# OpenAPI Tags
# ----------------------------------------------------

app.openapi_tags = [

    {
        "name": "Health",
        "description": "Health and status endpoints"
    },

    {
        "name": "Prediction",
        "description": "AI Trading Predictions"
    },

    {
        "name": "Models",
        "description": "Model information"
    }

]

# ----------------------------------------------------
# Documentation Endpoint
# ----------------------------------------------------

@app.get("/info")
def info():

    return {

        "application": "Trading Intelligence API",

        "author": "Person 2 - Quantitative Trading & AI",

        "version": "1.0.0",

        "description":

            "AI-powered trading prediction engine.",

        "pipeline": [

            "OHLCV",

            "Indicators",

            "Regime",

            "Strategy",

            "Risk Engine",

            "Machine Learning",

            "Calibration",

            "Prediction"

        ]

    }


# ----------------------------------------------------
# Ready Endpoint
# ----------------------------------------------------

@app.get("/ready")
def ready():

    ready = (

        MODEL is not None

        and FEATURE_COLUMNS is not None

    )

    return {

        "ready": ready

    }


# ----------------------------------------------------
# Startup Summary
# ----------------------------------------------------

@app.get("/summary")
def summary():

    return {

        "model_loaded":

            MODEL is not None,

        "calibrator_loaded":

            CALIBRATOR is not None,

        "features":

            len(FEATURE_COLUMNS)

            if FEATURE_COLUMNS is not None

            else 0,

        "supported_symbols": [

            "BTCUSDT",

            "ETHUSDT",

            "BNBUSDT",

            "SOLUSDT",

            "XRPUSDT",

            "ADAUSDT"

        ],

        "supported_timeframes": [

            "1m",

            "5m",

            "15m",

            "1h",

            "4h",

            "1d"

        ]

    }


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "inference.api:app",

        host="0.0.0.0",

        port=8000,

        reload=True,

        log_level="info"

    )