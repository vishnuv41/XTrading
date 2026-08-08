"""
run_backtest_from_db.py
------------------------
Backtests a trained model's signal against real, held-out price action
using ml.backtest.run_backtest — the actual "does this edge survive
costs" check. Uses the trailing `holdout_frac` of candles (same window
evaluate_model.py checks), so results are directly comparable to the
accuracy/confusion-matrix numbers you already have.

Note: this holdout was also what the probability calibrator was fit on
(not the base models — those never saw it). So this is a fair test of
"does the trading logic work," but if you want a fully untouched final
test set, carve out an extra slice beyond calibration_holdout_frac next
time you retrain.

Usage:
    python run_backtest_from_db.py --symbol BTC/USDT --timeframe 1h --model-dir models_artifacts/BTCUSDT_1h
"""
import argparse

from config.settings import settings
from pipeline.data_loader import load_ohlcv
from ml.predict import load_training_artifacts, predict
from ml.backtest import run_backtest, run_triple_barrier_backtest
from ml.utils.preprocessing import build_feature_matrix

# bars/year by timeframe, for Sharpe annualization
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
    ap.add_argument("--pt-mult", type=float, default=2.0, help="must match what the model was trained on")
    ap.add_argument("--sl-mult", type=float, default=2.0, help="must match what the model was trained on")
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--confidence-threshold", type=float, default=0.4)
    # Binance spot taker fee is 10bps/side; 5bps assumes maker orders or a
    # fee discount. Use 10-15bps for a more conservative, realistic check,
    # and add a few bps on top for slippage on top of that if trading size
    # that could move the book.
    ap.add_argument("--transaction-cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    df = load_ohlcv(args.symbol, args.timeframe, exchange=args.exchange)
    n = len(df)
    holdout_start = int(n * (1 - args.holdout_frac))
    df_hold = df.iloc[holdout_start:].reset_index(drop=True)

    artifacts = load_training_artifacts(args.model_dir)
    preds = predict(
        df_hold,
        ensemble=artifacts["ensemble"],
        feature_columns=artifacts["feature_columns"],
        calibrator=artifacts["calibrator"],
    )

    periods_per_year = BARS_PER_YEAR.get(args.timeframe, 252)

    # volatility series must match what triple_barrier_labels used at
    # training time (ATR14/close), or the barriers won't correspond to
    # what the model actually learned
    feat_df_hold = build_feature_matrix(df_hold, has_volume=True)
    volatility = (feat_df_hold["ATR14"] / feat_df_hold["close"]).reindex(df_hold.index)

    result = run_triple_barrier_backtest(
        df_hold,
        preds,
        volatility=volatility,
        vol_window=args.vol_window,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        max_holding=args.max_holding,
        confidence_threshold=args.confidence_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    m = result["metrics"]
    buy_hold_return = df_hold["close"].iloc[-1] / df_hold["close"].iloc[0] - 1

    print(f"\n{args.symbol} {args.timeframe} — backtest over last {len(df_hold)} candles "
          f"({args.holdout_frac:.0%} holdout), cost={args.transaction_cost_bps}bps/side\n")
    print(f"Strategy total return:  {m['total_return']*100:+.2f}%")
    print(f"Buy & hold return:      {buy_hold_return*100:+.2f}%")
    print(f"Sharpe ratio:           {m['sharpe_ratio']:.2f}")
    print(f"Max drawdown:           {m['max_drawdown']*100:.2f}%")
    print(f"Number of trades:       {m['num_trades']}")
    print(f"Win rate:               {m['win_rate']*100:.2f}%" if m['win_rate'] == m['win_rate'] else "Win rate: n/a (0 trades)")
    print(f"Profit factor:          {m['profit_factor']:.2f}")
    print(f"Avg trade return:       {m['avg_trade_return']*100:.3f}%")

    if m["num_trades"] < 30:
        print(f"\n⚠️  Only {m['num_trades']} trades in this window — too few to trust "
              f"Sharpe/win-rate/profit-factor. Use a longer holdout or lower "
              f"confidence_threshold to get more trades before drawing conclusions.")
    if m["total_return"] == m["total_return"] and m["total_return"] <= 0:
        print(f"\n⚠️  Strategy return is non-positive after {args.transaction_cost_bps}bps "
              f"costs — the raw accuracy edge is real (see evaluate_model.py) but too "
              f"thin to survive trading costs at this cost/confidence/holding setting. "
              f"Try: raising confidence_threshold (fewer, higher-quality trades), "
              f"widening the barrier gap (pt_mult/sl_mult) so wins are bigger than costs, "
              f"or reducing max_holding to cut exposure to costly whipsaws.")


if __name__ == "__main__":
    main()