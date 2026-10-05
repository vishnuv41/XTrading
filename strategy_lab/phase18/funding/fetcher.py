"""
strategy_lab/phase18/funding/fetcher.py
----------------------------------------
Fast historical funding rate fetcher for Binance USD-M Futures using direct REST endpoint.
Fetches 8-hour funding rate history for the 9 universe assets up to 2026-09-30 23:59:59 UTC.
Stores raw series in strategy_lab/phase18/data/funding_rates/ with full provenance.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from strategy_lab.phase18.manifest_validator import load_manifest, check_dataset_boundaries

FUNDING_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "funding_rates"
FUNDING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mapping standard spot symbols to Binance USD-M perpetual raw symbols
BINANCE_RAW_FUTURES_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USDT": "ADAUSDT",
    "LTC/USDT": "LTCUSDT",
    "SOL/USDT": "SOLUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "LINK/USDT": "LINKUSDT",
}


def fetch_historical_funding_rates(
    symbol: str,
    start_ts_ms: int = 1577836800000,  # 2020-01-01 00:00:00 UTC
    end_ts_ms: int = 1790812799000,    # 2026-09-30 23:59:59 UTC
) -> pd.DataFrame:
    """
    Fetches paginated funding rate history from Binance USD-M Futures REST endpoint.
    """
    raw_sym = BINANCE_RAW_FUTURES_MAP[symbol]
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    
    all_rows = []
    current_start = start_ts_ms
    limit = 1000

    print(f"Fetching funding rates for {symbol} ({raw_sym})...")

    while current_start < end_ts_ms:
        try:
            params = {
                "symbol": raw_sym,
                "startTime": current_start,
                "limit": limit,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} for {raw_sym}: {resp.text}")
                time.sleep(1.0)
                continue

            records = resp.json()
            if not records or not isinstance(records, list):
                break

            for r in records:
                ts = int(r["fundingTime"])
                if ts <= end_ts_ms:
                    mp_val = r.get("markPrice")
                    mark_p = float(mp_val) if (mp_val is not None and mp_val != "") else 0.0
                    all_rows.append({
                        "timestamp": ts,
                        "datetime": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                        "funding_rate": float(r["fundingRate"]),
                        "mark_price": mark_p,
                    })

            last_ts = int(records[-1]["fundingTime"])
            if last_ts <= current_start or len(records) < 2:
                break
            current_start = last_ts + 1
            time.sleep(0.05)  # Fast respectful pacing
        except Exception as exc:
            print(f"Error fetching {raw_sym} at {current_start}: {exc}. Retrying...")
            time.sleep(1.0)
            continue

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "datetime", "funding_rate", "mark_price"])

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # Filter strictly to boundary
    df = df[df.index <= "2026-09-30 23:59:59+00:00"]
    return df


def download_all_funding_data():
    """Fetch and persist funding rate datasets for all 9 assets."""
    manifest = load_manifest()
    universe = manifest["universe"]

    for sym in universe:
        clean_name = sym.replace("/", "_")
        target_csv = FUNDING_DATA_DIR / f"{clean_name}_funding.csv"
        
        df = fetch_historical_funding_rates(sym)
        df.to_csv(target_csv)
        print(f"[OK] Saved {len(df)} funding rows to {target_csv.name} ({df.index[0]} to {df.index[-1]})")


if __name__ == "__main__":
    download_all_funding_data()
