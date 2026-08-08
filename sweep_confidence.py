"""
sweep_confidence.py
---------------------
Sweeps confidence_threshold through the barrier-aware backtest and
prints a table: does raising the bar for "high enough confidence to
trade" turn this from a losing to a winning strategy (fewer, better
trades), and at what point do you run out of trades to trust the
result at all?

Usage:
    python sweep_confidence.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h
"""
import argparse

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.predict import load_training_artifacts, predict
from ml.backtest import run_triple_barrier_backtest
from ml.utils.preprocessing import build_feature_matrix

BARS_PER_YEAR = {
    "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
    "1h": 365 * 24, "4h": 365 * 6, "1d": 365,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--exchange", default=settings.exchange.default_exchange)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--max-holding", type=int, default=20)
    ap.add_argument("--pt-mult", type=float, default=2.0)
    ap.add_argument("--sl-mult", type=float, default=2.0)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--transaction-cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
    n = len(df)
    holdout_start = int(n * (1 - args.holdout_frac))
    df_hold = df.iloc[holdout_start:].reset_index(drop=True)

    artifacts = load_training_artifacts(args.model_dir)
    preds = predict(df_hold, ensemble=artifacts["ensemble"],
                     feature_columns=artifacts["feature_columns"],
                     calibrator=artifacts["calibrator"])

    feat_df_hold = build_feature_matrix(df_hold, has_volume=True)
    volatility = (feat_df_hold["ATR14"] / feat_df_hold["close"]).reindex(df_hold.index)
    periods_per_year = BARS_PER_YEAR.get(args.timeframe, 252)

    print(f"{'threshold':>10} {'trades':>8} {'win_rate':>10} {'total_ret':>10} "
          f"{'sharpe':>8} {'profit_factor':>14}")
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        result = run_triple_barrier_backtest(
            df_hold, preds, volatility=volatility, vol_window=args.vol_window,
            pt_mult=args.pt_mult, sl_mult=args.sl_mult, max_holding=args.max_holding,
            confidence_threshold=thresh, transaction_cost_bps=args.transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        m = result["metrics"]
        wr = f"{m['win_rate']*100:.1f}%" if m["win_rate"] == m["win_rate"] else "n/a"
        tr = f"{m['total_return']*100:+.2f}%" if m["total_return"] == m["total_return"] else "n/a"
        sh = f"{m['sharpe_ratio']:.2f}" if m["sharpe_ratio"] == m["sharpe_ratio"] else "n/a"
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] == m["profit_factor"] else "n/a"
        print(f"{thresh:>10.2f} {m['num_trades']:>8} {wr:>10} {tr:>10} {sh:>8} {pf:>14}")

    print("\nLook for the threshold where total_ret turns positive AND trades "
          "stays >=30 (fewer than that and the result isn't trustworthy either "
          "way). If nothing in this range works, the edge doesn't survive "
          "costs at any reasonable trade frequency with symmetric pt_mult=sl_mult "
          "— next lever is retraining with wider/asymmetric barriers, not "
          "threshold tuning.")


if __name__ == "__main__":
    main()