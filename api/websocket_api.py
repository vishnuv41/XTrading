"""
api/websocket_api.py
------------------------
Push live closed candles to internal subscribers (Person 2's live
inference loop, Person 3's websocket_gateway) by bridging Redis pub/sub
(published from ingestion/realtime_stream.py) to a FastAPI WebSocket
endpoint. This is Person 1's boundary of responsibility — Person 3's
own websocket_gateway is what fans this out to browser clients.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database.redis_cache import subscribe_candles

logger = logging.getLogger(__name__)
router = APIRouter(tags=["market-ws"])


@router.websocket("/ws/ohlcv/{symbol}/{timeframe}")
async def ws_ohlcv(websocket: WebSocket, symbol: str, timeframe: str):
    """
    Usage: connect to /ws/ohlcv/BTC%2FUSDT/1m (URL-encode the slash in
    the symbol), receive one JSON message per newly-closed candle.
    """
    await websocket.accept()
    logger.info("WS client connected: %s %s", symbol, timeframe)
    try:
        async for candle in subscribe_candles(symbol, timeframe):
            await websocket.send_json(candle)
    except WebSocketDisconnect:
        logger.info("WS client disconnected: %s %s", symbol, timeframe)
