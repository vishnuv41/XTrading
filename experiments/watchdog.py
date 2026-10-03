"""
experiments/watchdog.py
-----------------------
Process Watchdog & Health Check Utility for Phase 13 Live Daemons.
Monitors PostgreSQL for continuous candle ingestion and model prediction heartbeat.
"""
import sys
import time
import logging
from datetime import datetime, timezone
from sqlalchemy import text

sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from database.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("watchdog")

def check_system_health(max_allowed_gap_seconds: int = 7200):
    engine = get_engine()
    now_utc = datetime.now(timezone.utc)
    
    with engine.connect() as conn:
        # 1. Check OHLCV max timestamps
        ohlcv_rows = conn.execute(text("""
            SELECT symbol, MAX(ts) FROM ohlcv WHERE timeframe = '1h' GROUP BY symbol
        """)).fetchall()
        
        logger.info("=== OHLCV HEARTBEAT ===")
        for r in ohlcv_rows:
            sym, max_ts = r[0], r[1]
            if max_ts:
                gap = (now_utc - max_ts).total_seconds()
                status = "OK" if gap <= max_allowed_gap_seconds else f"STALE ({gap/3600:.1f} hours lag)"
                logger.info("  %-10s | Last OHLCV: %s | Status: %s", sym, max_ts, status)
            else:
                logger.warning("  %-10s | NO OHLCV RECORDS", sym)
                
        # 2. Check Prediction Log max timestamps
        pred_rows = conn.execute(text("""
            SELECT symbol, MAX(ts), COUNT(*) FROM prediction_log WHERE ts >= '2026-09-06 14:00:00+00' GROUP BY symbol
        """)).fetchall()
        
        logger.info("=== PREDICTION LOG HEARTBEAT ===")
        for r in pred_rows:
            sym, max_ts, count = r[0], r[1], r[2]
            if max_ts:
                gap = (now_utc - max_ts).total_seconds()
                status = "OK" if gap <= max_allowed_gap_seconds else f"STALE ({gap/3600:.1f} hours lag)"
                logger.info("  %-10s | Last Pred: %s | Total: %d | Status: %s", sym, max_ts, count, status)
            else:
                logger.warning("  %-10s | NO PREDICTION RECORDS", sym)

        # 3. Check Active Position status
        open_pos = conn.execute(text("""
            SELECT trade_id, symbol, side, price, ts, bars_held FROM trade_log WHERE ts >= '2026-09-06 14:00:00+00' AND action = 'OPEN' AND trade_id NOT IN (SELECT trade_id FROM trade_log WHERE action = 'CLOSE')
        """)).fetchall()
        
        logger.info("=== ACTIVE POSITIONS ===")
        if open_pos:
            for p in open_pos:
                logger.info("  ACTIVE POSITION: %s %s @ $%.2f (Entered: %s)", p[1], p[2].upper(), p[3], p[4])
        else:
            logger.info("  No open positions (Portfolio Flat).")

def main():
    logger.info("Starting Phase 13 System Watchdog Check...")
    check_system_health()

if __name__ == "__main__":
    main()
