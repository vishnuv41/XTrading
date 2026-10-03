"""
experiments/sync_and_check_progress.py
----------------------------------------
Operational utility to sync missing 1H candles from Binance REST API,
evaluate any un-evaluated closed candles sequentially for BTC & ETH,
and report the current Phase 13 trade log and active position status.
"""
import sys
import asyncio
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from database.connection import get_engine
from exchange.binance import BinanceExchange
from ingestion.historical_loader import backfill_symbol
from ml.predict import load_training_artifacts
from paper_trading.engine import PaperTradingEngine

def model_dir_for(symbol: str, timeframe: str) -> str:
    return f"models_artifacts/{symbol.replace('/', '')}_{timeframe}"

async def run_backfill():
    exchange = BinanceExchange()
    since = datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc)
    await backfill_symbol(exchange, "BTC/USDT", "1h", since=since)
    await backfill_symbol(exchange, "ETH/USDT", "1h", since=since)
    await exchange.close()

def main():
    engine = get_engine()
    print("=== STEP 1: FETCHING LATEST CLOSED 1H CANDLES FROM BINANCE ===")
    asyncio.run(run_backfill())
    
    print("\n=== STEP 2: SEQUENTIAL EVALUATION FOR UN-EVALUATED CLOSED CANDLES ===")
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        artifacts = load_training_artifacts(model_dir_for(symbol, "1h"))
        pt_engine = PaperTradingEngine(
            symbol=symbol, timeframe="1h", exchange="binance",
            model=artifacts["ensemble"], feature_columns=artifacts["feature_columns"],
            calibrator=artifacts["calibrator"], db_engine=engine, persist_to_db=True,
        )
        
        with engine.connect() as conn:
            max_p_ts = conn.execute(text("""
                SELECT MAX(ts) FROM prediction_log WHERE symbol = :sym AND ts >= '2026-09-06 14:00:00+00'
            """), {"sym": symbol}).scalar()
            
            new_candles = conn.execute(text("""
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :sym AND timeframe = '1h' AND ts > :max_ts
                AND ts + INTERVAL '1 hour' <= NOW()
                ORDER BY ts ASC
            """), {"sym": symbol, "max_ts": max_p_ts}).fetchall()
            
            if new_candles:
                print(f"[{symbol}] Found {len(new_candles)} new closed candle(s) to evaluate.")
                df_full = pd.read_sql(
                    text("SELECT ts as timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = :sym AND timeframe = '1h' ORDER BY ts ASC"),
                    con=conn, params={"sym": symbol}
                )
                for c in new_candles:
                    c_ts = c[0]
                    idx = df_full[df_full["timestamp"] <= c_ts].index
                    if len(idx) > 0:
                        window = df_full.iloc[max(0, idx[-1] - 250): idx[-1] + 1].reset_index(drop=True)
                        res = pt_engine.on_bar(window)
                        print(f"  --> [{symbol}] Evaluated Bar {c_ts}: pred={res.get('prediction')} gate_pass={res.get('gate_pass')} executed={res.get('executed')}")
            else:
                print(f"[{symbol}] Up to date. No new closed candles since {max_p_ts}.")

    print("\n=== STEP 3: PHASE 13 FACTUAL STATUS QUERY ===")
    print("Protocol Status: ONGOING PROSPECTIVE VALIDATION (N < 30)")
    print("Rule of Non-Interference: Zero statistical decision gate evaluation until N >= 30.\n")
    
    with engine.connect() as conn:
        for sym in ["BTC/USDT", "ETH/USDT"]:
            preds = conn.execute(text("""
                SELECT ts, prediction, confidence, executed, created_at
                FROM prediction_log
                WHERE symbol = :sym AND ts >= '2026-09-06 14:00:00+00'
                ORDER BY ts DESC LIMIT 3
            """), {"sym": sym}).fetchall()
            print(f"Latest 3 Predictions for {sym}:")
            for p in preds:
                print(f"  ts={p[0]} | pred={p[1]} | conf={p[2]:.4f} | executed={p[3]}")
                
        print("\n=== PHASE 13 TRADES IN DATABASE ===")
        trades = conn.execute(text("""
            SELECT id, trade_id, symbol, side, action, ts, price, size, fee, stop_loss, take_profit, exit_reason, realized_pnl, realized_pnl_pct, bars_held
            FROM trade_log
            WHERE ts >= '2026-09-06 14:00:00+00'
            ORDER BY ts ASC, id ASC
        """)).fetchall()
        
        trades_dict = {}
        for r in trades:
            t_id = r[1]
            if t_id not in trades_dict:
                trades_dict[t_id] = {"open": None, "close": None}
            if r[4] == "OPEN":
                trades_dict[t_id]["open"] = r
            elif r[4] == "CLOSE":
                trades_dict[t_id]["close"] = r
                
        total_realized_usd = 0.0
        completed_count = 0
        wins = 0
        losses = 0
        
        for t_id, data in trades_dict.items():
            op = data["open"]
            cl = data["close"]
            if op and cl:
                completed_count += 1
                pnl_usd = cl[12]
                pnl_pct = cl[13] * 100 if cl[13] is not None else 0.0
                total_realized_usd += pnl_usd
                if pnl_usd > 0:
                    wins += 1
                else:
                    losses += 1
                print(f"[CLOSED] {op[2]} {op[3].upper()} (ID: {t_id[:8]}) | Entry: {op[5]} @ ${op[6]:.2f} | Exit: {cl[5]} @ ${cl[6]:.2f} ({cl[11]}) | Bars: {cl[14]} | Net PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
            elif op and not cl:
                print(f"[ACTIVE] {op[2]} {op[3].upper()} (ID: {t_id[:8]}) | Entry: {op[5]} @ ${op[6]:.2f} | SL: ${op[9]:.2f} | TP: ${op[10]:.2f} | Size: {op[7]:.4f}")

        print("\n=== LATEST OHLCV CANDLE & ACTIVE MTM ===")
        eth_candle = conn.execute(text("""
            SELECT ts, open, high, low, close FROM ohlcv WHERE symbol = 'ETH/USDT' AND timeframe = '1h' ORDER BY ts DESC LIMIT 1
        """)).fetchone()
        if eth_candle:
            print(f"Latest Closed 1H Candle: {eth_candle[0]} | Open: ${eth_candle[1]:.2f} | High: ${eth_candle[2]:.2f} | Low: ${eth_candle[3]:.2f} | Close: ${eth_candle[4]:.2f}")
            entry_price = 2490.594675
            size = 3.32796265
            latest_close = eth_candle[4]
            floating_pnl_usd = (latest_close - entry_price) * size
            floating_pnl_pct = (latest_close - entry_price) / entry_price * 100
            print(f"Active ETH Position Floating MTM PnL: ${floating_pnl_usd:+.2f} ({floating_pnl_pct:+.2f}%)")

if __name__ == "__main__":
    main()
