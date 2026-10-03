import sys
import os
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.data_loader import load_ohlcv

end_dt = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
df_btc = load_ohlcv("BTC/USDT", "1h", end=end_dt, limit=None)
df_eth = load_ohlcv("ETH/USDT", "1h", end=end_dt, limit=None)

print(f"BTC 2017-2025 rows: {len(df_btc)} | Min TS: {df_btc['timestamp'].min()} | Max TS: {df_btc['timestamp'].max()}")
print(f"ETH 2017-2025 rows: {len(df_eth)} | Min TS: {df_eth['timestamp'].min()} | Max TS: {df_eth['timestamp'].max()}")
