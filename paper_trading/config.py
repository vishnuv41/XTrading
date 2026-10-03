"""
paper_trading/config.py
-------------------------
Env-driven config for the paper trading engine, following the same
pattern as config/settings.py (single frozen dataclass instance,
imported rather than re-read). Kept separate from config.settings.risk
because these knobs (fees, slippage, timeout) are specific to how the
*simulator* fills orders, not to the risk engine's own stop/target/
sizing logic — conflating the two would make it unclear which module
owns which setting.
"""

import os
from dataclasses import dataclass

from config.settings import _env_bool  # reuse the same env-parsing helper


@dataclass(frozen=True)
class PaperTradingConfig:
    # Starting virtual cash balance for a fresh portfolio.
    starting_cash: float = float(os.getenv("PAPER_STARTING_CASH", "10000"))

    # Round-trip trading fee, in basis points, charged on BOTH the open
    # and the close fill (so a full round trip costs 2x this). 10 bps
    # (0.10%) is a reasonable default for a retail crypto taker fee.
    fee_bps: float = float(os.getenv("PAPER_FEE_BPS", "10.0"))

    # Simulated slippage in basis points applied against the trader on
    # every fill (worse price than the signal's reference price) — buys
    # fill higher, sells fill lower, longs close lower, shorts close
    # higher. Kept separate from fee_bps so each can be tuned/disabled
    # independently (e.g. slippage=0 for a pure fee-cost sensitivity run).
    slippage_bps: float = float(os.getenv("PAPER_SLIPPAGE_BPS", "5.0"))

    # Force-close a position after this many bars regardless of SL/TP
    # (mirrors ml/backtest.py's max_holding, kept independently
    # configurable since live paper trading may want a different
    # horizon than what a given model was backtested with).
    timeout_bars: int = int(os.getenv("PAPER_TIMEOUT_BARS", "48"))

    # If True, a new opposite-direction signal (e.g. SELL while long is
    # open) closes the existing position first ("signal_flip" exit)
    # instead of being ignored. If False, an opposite signal while a
    # position is open is ignored entirely until it exits naturally.
    allow_signal_flip_exit: bool = _env_bool("PAPER_ALLOW_SIGNAL_FLIP_EXIT", True)

    # Bars per year used for Sharpe annualization in metrics.py — must
    # match the timeframe being traded (e.g. 1h crypto ~ 24*365).
    periods_per_year: int = int(os.getenv("PAPER_PERIODS_PER_YEAR", "8760"))

    # Margin multiplier: cash required to OPEN a position of a given
    # notional value is notional / leverage, not the full notional.
    # Matters because risk_engine.position_size sizes positions from
    # risk_pct-of-equity and stop distance alone — for a tight stop
    # (e.g. 1% away) at 1% risk, that formula lands on notional roughly
    # equal to equity itself, which is the correct sizing for a
    # margined/futures account but will constantly reject as
    # "insufficient cash" against a plain spot-cash account. Set to 1.0
    # for a true cash/spot account (positions capped at what cash can
    # actually buy outright); set higher (e.g. 5.0, 10.0) to model a
    # margined or futures account matching how the sizing formula
    # already assumes capital is used. Default of 3.0 is a moderate
    # middle ground — override via env for either extreme.
    leverage: float = float(os.getenv("PAPER_LEVERAGE", "3.0"))

    # Trail the stop-loss using the Supertrend line (risk_engine.stoploss.
    # calculate_supertrend_stop) once a position is open, instead of
    # leaving the entry-time ATR-based stop fixed for the trade's whole
    # life. Only ever TIGHTENS the stop (moves it in the trade's favor),
    # never loosens it — see paper_trading.exit_manager.update_trailing_stop.
    # Off by default: this changes exit behavior for anything already
    # tuned against a fixed stop (backtests, the confidence-threshold
    # sweep), so turning it on is a deliberate choice, not a silent
    # default change.
    enable_trailing_stop: bool = _env_bool("PAPER_ENABLE_TRAILING_STOP", False)


settings = PaperTradingConfig()
