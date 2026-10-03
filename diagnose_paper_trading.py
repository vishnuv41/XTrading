"""
diagnose_paper_trading.py
----------------------------
Answers the "where do the trades actually come from" question directly
from PaperTradingEngine.run_replay's own return value — no separate
tally logic that could drift from what the engine actually does.

Defaults to the same holdout_frac as run_paper_trading.py / 
run_backtest_from_db.py, so this is directly comparable to both.

Run:
    python diagnose_paper_trading.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h
"""

import argparse
import logging

import pandas as pd

from config.settings import settings
from ml.predict import load_training_artifacts
from paper_trading.engine import PaperTradingEngine
from paper_trading.metrics import print_summary
from pipeline.data_loader import load_ohlcv

logging.basicConfig(level=logging.WARNING)  # quiet the per-bar OPEN/CLOSE logger.info noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--exchange", default=settings.exchange.default_exchange)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--warmup-bars", type=int, default=250)
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                     help="Default matches run_backtest_from_db.py, so this run is comparable "
                          "to that backtest number, not to a full-dataset replay.")
    args = ap.parse_args()

    artifacts = load_training_artifacts(args.model_dir)
    engine = PaperTradingEngine(
        symbol=args.symbol, timeframe=args.timeframe, exchange=args.exchange,
        model=artifacts["ensemble"], feature_columns=artifacts["feature_columns"],
        calibrator=artifacts["calibrator"], persist_to_db=False,  # read-only diagnostic, don't write rows
    )

    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
    n = len(df)
    holdout_start = int(n * (1 - args.holdout_frac))
    slice_start = max(0, holdout_start - args.warmup_bars)
    df_replay = df.iloc[slice_start:].reset_index(drop=True)

    print(f"Replaying {len(df_replay) - args.warmup_bars} bars "
          f"(holdout_frac={args.holdout_frac}, {len(df_replay)} of {n} total candles, "
          f"{args.warmup_bars} reserved for warmup)...\n")
    results = engine.run_replay(df_replay, warmup_bars=args.warmup_bars)

    # --- raw prediction distribution (every bar, whether traded or not) ---
    preds = pd.Series([r["prediction"] for r in results])
    print("=" * 70)
    print("RAW PREDICTIONS (every bar, engine's actual output)")
    print("=" * 70)
    print(preds.value_counts().reindex(["SELL", "HOLD", "BUY"], fill_value=0).to_string())

    # --- executed vs skipped, split by what the prediction was ---
    exec_by_pred = pd.DataFrame({"prediction": preds, "executed": [r["executed"] for r in results]})
    print(f"\n{'='*70}\nEXECUTED vs SKIPPED, by predicted direction\n{'='*70}")
    print(exec_by_pred.groupby("prediction")["executed"].agg(["sum", "count"]).rename(
        columns={"sum": "executed", "count": "total"}))
    print("(SELL/BUY rows with executed=False were skipped by open_from_prediction — duplicate "
          "position, invalid/blocked risk sizing, or no cash; HOLD is never executed by definition.)")

    # --- exits: reason x direction, straight from the portfolio's closed trades ---
    closed = engine.portfolio.closed_trades
    if closed:
        trades_df = pd.DataFrame([{
            "side": t.side, "reason": t.exit_reason, "pnl": t.realized_pnl,
            "pnl_pct": t.realized_pnl_pct,
        } for t in closed])
        print(f"\n{'='*70}\nEXITS: reason x direction\n{'='*70}")
        pivot = trades_df.pivot_table(index="reason", columns="side", values="pnl",
                                       aggfunc=["count", "mean"])
        print(pivot.to_string())

        print(f"\n{'='*70}\nBUY vs SELL expectancy (closed trades only)\n{'='*70}")
        for side in ("long", "short"):
            sub = trades_df[trades_df["side"] == side]
            if len(sub) == 0:
                print(f"  {side}: no closed trades")
                continue
            wins = sub[sub["pnl"] > 0]
            losses = sub[sub["pnl"] <= 0]
            win_rate = len(wins) / len(sub)
            pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if losses["pnl"].sum() != 0 else float("inf")
            print(f"  {side:6s}  n={len(sub):4d}  win_rate={win_rate:.2%}  "
                  f"avg_pnl={sub['pnl'].mean():+.2f}  profit_factor={pf:.3f}")
    else:
        print("\nNo trades closed in this window.")

    print(f"\n{'='*70}\nPORTFOLIO SUMMARY (this window only)\n{'='*70}")
    print_summary(engine.portfolio, periods_per_year={
        "1m": 60 * 24 * 365, "5m": 12 * 24 * 365, "15m": 4 * 24 * 365,
        "1h": 24 * 365, "4h": 6 * 365, "1d": 365,
    }.get(args.timeframe, 8760))


if __name__ == "__main__":
    main()