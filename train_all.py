"""
train_all.py
------------
Sequentially trains every symbol/timeframe combo, smallest first, each
into its own models_artifacts/<SYMBOL>_<tf>/ dir. Skips combos that
already have a metadata.json (already trained) unless --force is passed
— re-running this shouldn't silently retrain BTC/USDT 1h and invalidate
the confidence_threshold=0.70 operating point that's already been
validated and locked into config/settings.py.

Run from the XTrading/ root: python train_all.py
"""
import argparse
import os
import time
import subprocess
import sys

# smallest -> largest so you see accuracy fast and can Ctrl+C before
# the expensive ones if something looks off. 1d is excluded by default
# below (140 samples is too small to trust — see metadata.json warning).
RUNS = [
    ("BTC/USDT", "4h"), ("ETH/USDT", "4h"), ("SOL/USDT", "4h"),
    ("BTC/USDT", "1h"), ("ETH/USDT", "1h"), ("SOL/USDT", "1h"),
    ("BTC/USDT", "15m"), ("ETH/USDT", "15m"), ("SOL/USDT", "15m"),
    ("BTC/USDT", "5m"), ("ETH/USDT", "5m"), ("SOL/USDT", "5m"),
    ("BTC/USDT", "1m"), ("ETH/USDT", "1m"), ("SOL/USDT", "1m"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="retrain even combos that already have models_artifacts/<SYMBOL>_<tf>/metadata.json")
    ap.add_argument("--include-1d", action="store_true",
                     help="also train 1d — off by default, only 140 samples, too small to trust (see evaluate_model.py warning)")
    args = ap.parse_args()

    runs = RUNS.copy()
    if args.include_1d:
        runs = [("BTC/USDT", "1d"), ("ETH/USDT", "1d"), ("SOL/USDT", "1d")] + runs

    results = []
    for symbol, tf in runs:
        out_dir = f"models_artifacts/{symbol.replace('/', '')}_{tf}"
        meta_path = os.path.join(out_dir, "metadata.json")

        if os.path.exists(meta_path) and not args.force:
            print(f"\nSkipping {symbol} {tf} — {meta_path} already exists (use --force to retrain).")
            results.append((symbol, tf, "SKIPPED (already trained)", "-"))
            continue

        n_splits = 3 if tf in ("1m", "5m") else 5
        print(f"\n{'='*60}\nTraining {symbol} {tf} -> {out_dir} (n_splits={n_splits})\n{'='*60}")
        start = time.time()
        try:
            subprocess.run(
                [sys.executable, "run_training_from_db.py",
                 "--symbol", symbol, "--timeframe", tf,
                 "--output-dir", out_dir, "--n-splits", str(n_splits)],
                check=True,
            )
            elapsed = time.time() - start
            results.append((symbol, tf, "OK", f"{elapsed/60:.1f} min"))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            results.append((symbol, tf, f"FAILED: {e}", f"{elapsed/60:.1f} min"))
            print(f"!! {symbol} {tf} failed, continuing to next combo...")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for symbol, tf, status, elapsed in results:
        print(f"{symbol:10s} {tf:5s} {status:40s} {elapsed}")

    print(f"\nNext: for each OK combo, run evaluate_model.py and sweep_confidence.py "
          f"before trusting any accuracy number, then update config/settings.py's "
          f"per-timeframe confidence thresholds for whichever ones show a real, "
          f"cost-surviving edge (see sweep_confidence.py output).")


if __name__ == "__main__":
    main()