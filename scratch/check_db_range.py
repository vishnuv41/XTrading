import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    btc_res = conn.execute(text("SELECT MIN(ts), MAX(ts), COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT'")).fetchall()
    eth_res = conn.execute(text("SELECT MIN(ts), MAX(ts), COUNT(*) FROM ohlcv WHERE symbol='ETH/USDT'")).fetchall()
    print("BTC DB Range:", btc_res)
    print("ETH DB Range:", eth_res)
