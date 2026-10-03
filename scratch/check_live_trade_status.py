import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_engine
from paper_trading.db_logger import load_global_portfolio_state
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    # 1. Closed trades
    closed_trades = conn.execute(text("""
        SELECT trade_id, symbol, side, ts, price, size, fee, realized_pnl, realized_pnl_pct, exit_reason, cash_after, equity_after
        FROM trade_log
        WHERE action = 'CLOSE'
        ORDER BY id ASC
    """)).mappings().all()

state = load_global_portfolio_state()

print("=========================================================")
print("          PHASE 13 LIVE PAPER TRADING STATUS             ")
print("=========================================================")
print(f"Target Trial Completed Trades N: {len(closed_trades)} / 30")
print("---------------------------------------------------------")

wins = 0
losses = 0
for i, t in enumerate(closed_trades, 1):
    pnl = float(t["realized_pnl"] or 0.0)
    pnl_pct = float(t["realized_pnl_pct"] or 0.0)
    if pnl > 0: wins += 1
    elif pnl < 0: losses += 1
    print(f"Trade #{i} [{t['symbol']}] {t['side']} | Exit TS: {t['ts']} | Exit Price: ${t['price']:,.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Reason: {t['exit_reason']}")

print("---------------------------------------------------------")
print(f"Canonical Starting Equity: ${state['starting_cash']:,.2f}")
print(f"Total Realized P&L:        ${state['total_realized_pnl']:+,.2f} ({(state['total_realized_pnl']/state['starting_cash'])*100:+.2f}%)")
print(f"Current Canonical Equity:   ${state['global_equity']:,.2f}")
print(f"Available Portfolio Cash:   ${state['available_cash']:,.2f}")
print(f"Completed Win / Loss:       {wins} Wins / {losses} Losses (Win Rate: {(wins/len(closed_trades))*100 if len(closed_trades)>0 else 0:.1f}%)")
print(f"Active Open Positions:      {len(state['active_positions'])}")

if state['active_positions']:
    for pos in state['active_positions']:
        print(f"  -> ACTIVE: [{pos['symbol']}] {pos['side']} | Entry: ${pos['entry_price']:,.2f} | Size: {pos['size']} | Entry TS: {pos['entry_ts']}")
else:
    print("  -> Currently NO active open positions. Listening for Top 1% signal...")
print("=========================================================")
