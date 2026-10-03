"""
enhanced_ml/labeling/tp_before_sl.py
--------------------------------------
V2 label: for a trade setup opened at each bar (entry/SL/TP computed
the same way risk_engine actually computes them for a live trade), did
price hit TP before SL within max_holding bars?

label = 1   TP touched first
label = 0   SL touched first
label = NaN neither touched within max_holding (drop before training,
            same convention as ml/labeling/triple_barrier.py) — this is
            deliberately NOT folded into 0, since "timed out" is not
            the same outcome as "stopped out" and conflating them would
            bias the model toward calling timeouts losses.

This does not modify or import from ml/labeling/triple_barrier.py — V1's
label definition is untouched. This module instead reuses
risk_engine.stoploss.calculate_stop_loss and
risk_engine.takeprofit.calculate_take_profit so the label's SL/TP match
what a live trade would actually use, rather than an independent barrier
formula that could drift from the real risk engine over time.

Anti-leakage: barrier touches are checked using bars strictly AFTER the
setup bar (high[i+1 : i+1+max_holding], never bar i itself or earlier).
Only future candles (which is expected/required for any label) are
used; no feature computed elsewhere may look into this same window.
"""

import numpy as np
import pandas as pd

from risk_engine.stoploss import calculate_stop_loss
from risk_engine.takeprofit import calculate_take_profit


def label_tp_before_sl(
    df: pd.DataFrame,
    side: str = "long",
    atr_col: str = "ATR14",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    sl_multiplier: float = 2.0,
    risk_reward_ratio: float = 2.0,
    max_holding: int = 48,
) -> pd.DataFrame:
    """
    Label each bar with whether a `side` trade entered at that bar's
    close would hit take-profit before stop-loss.

    Parameters
    ----------
    side : 'long' or 'short'. Labels are side-specific — this answers
        "would a LONG (or SHORT) setup here have worked", not both at
        once. Call twice (once per side) if you need both.
    atr_col : column with the ATR value to size SL/TP from (must
        already exist in df — this module does not compute indicators).
    sl_multiplier, risk_reward_ratio : passed straight through to
        risk_engine.stoploss.calculate_stop_loss /
        risk_engine.takeprofit.calculate_take_profit, so a setup here
        is defined identically to how the live risk engine would size
        the same trade.
    max_holding : vertical/timeout barrier in bars.

    Returns
    -------
    DataFrame indexed like `df`, columns:
        - 'label'        : 1.0 (TP first), 0.0 (SL first), NaN (timeout
                            or insufficient forward data)
        - 'entry_price', 'stop_loss', 'take_profit' : the setup used
        - 'touch_type'    : 'tp' | 'sl' | 'timeout' | None
        - 'holding_bars'  : bars actually held until touch (NaN if timeout)

    The last `max_holding` rows won't have a full forward window and
    will have NaN label/touch fields — drop before training, same as
    ml/labeling/triple_barrier.py's documented convention.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    close = df[close_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    labels = np.full(n, np.nan)
    entry_prices = np.full(n, np.nan)
    stop_losses = np.full(n, np.nan)
    take_profits = np.full(n, np.nan)
    touch_type = np.array([None] * n, dtype=object)
    holding_bars = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if i + max_holding >= n:
            continue  # not enough forward data for a full window

        entry = close[i]
        sl = calculate_stop_loss(entry, atr[i], side, multiplier=sl_multiplier)
        tp = calculate_take_profit(entry, sl, side, risk_reward_ratio=risk_reward_ratio)

        window_high = high[i + 1 : i + 1 + max_holding]
        window_low = low[i + 1 : i + 1 + max_holding]

        if side == "long":
            tp_hit = np.where(window_high >= tp)[0]
            sl_hit = np.where(window_low <= sl)[0]
        else:  # short
            tp_hit = np.where(window_low <= tp)[0]
            sl_hit = np.where(window_high >= sl)[0]

        first_tp = tp_hit[0] if len(tp_hit) else np.inf
        first_sl = sl_hit[0] if len(sl_hit) else np.inf

        entry_prices[i] = entry
        stop_losses[i] = sl
        take_profits[i] = tp

        if first_tp == np.inf and first_sl == np.inf:
            touch_type[i] = "timeout"
            # label stays NaN — timeout is not a win or a loss, see module docstring
            continue
        elif first_tp < first_sl:
            labels[i] = 1.0
            touch_type[i] = "tp"
            holding_bars[i] = first_tp + 1
        elif first_sl < first_tp:
            labels[i] = 0.0
            touch_type[i] = "sl"
            holding_bars[i] = first_sl + 1
        else:
            # same bar hits both — can't tell intrabar ordering from
            # OHLC alone; conservative assumption is SL first (don't
            # credit a win we can't actually confirm).
            labels[i] = 0.0
            touch_type[i] = "sl"
            holding_bars[i] = first_sl + 1

    return pd.DataFrame(
        {
            "label": labels,
            "entry_price": entry_prices,
            "stop_loss": stop_losses,
            "take_profit": take_profits,
            "touch_type": touch_type,
            "holding_bars": holding_bars,
        },
        index=df.index,
    )