"""
experiments/check_live_progress.py
-----------------------------------
Factual Progress & State Query for Phase 13 Prospective Trading.
Queries PostgreSQL for prediction logs, trade logs, and active position MTM.
"""
import sys
from datetime import datetime, timezone
from sqlalchemy import text
sys.path.insert(0, r"d:\all\XTrading_combined (1)\XTrading")
from database.connection import get_engine

def main():
    engine = get_engine()
    with engine.connect() as conn:
        print("=== PHASE 13 FACTUAL STATUS QUERY ===")
        print("Protocol Status: ONGOING PROSPECTIVE VALIDATION (N < 30)")
        print("Rule of Non-Interference: Zero statistical gate evaluation until N >= 30.\n")
        
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
            SELECT ts, close FROM ohlcv WHERE symbol = 'ETH/USDT' AND timeframe = '1h' ORDER BY ts DESC LIMIT 1
        """)).fetchone()
        if eth_candle:
            print(f"Latest Closed 1H Candle: {eth_candle[0]} | Close: ${eth_candle[1]:.2f}")
            entry_price = 2490.594675
            size = 3.32796265
            latest_close = eth_candle[1]
            floating_pnl_usd = (latest_close - entry_price) * size
            floating_pnl_pct = (latest_close - entry_price) / entry_price * 100
            print(f"Active ETH Position Floating MTM PnL: ${floating_pnl_usd:+.2f} ({floating_pnl_pct:+.2f}%)")

if __name__ == "__main__":
    main()
