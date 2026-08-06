"""
Triple-barrier labeling
--------------------------
Lopez de Prado's triple-barrier method: for each bar, set an upper
(profit-take) and lower (stop-loss) barrier scaled by volatility, plus
a vertical (max holding period) barrier, then label by whichever is
touched first. This is a meaningfully better label than a fixed
n-bar-forward-return threshold because it accounts for the path, not
just the endpoint, and adapts barrier width to prevailing volatility.
"""

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility: pd.Series = None,
    vol_window: int = 20,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels.

    Parameters
    ----------
    volatility : per-bar volatility series used to scale barrier width
        (e.g. rolling std of returns, or ATR{p}/close). If None, uses
        a rolling std of log returns over `vol_window`.
    pt_mult, sl_mult : barrier width as a multiple of volatility.
    max_holding : vertical barrier — max bars to hold before forcing exit.
    min_ret : if the vertical-barrier return's absolute value is below
        this, the label is forced to 0 regardless of sign (filters out
        labeling near-flat vertical exits as directional).

    Returns
    -------
    DataFrame indexed like `df`, columns:
        - 'label'        : +1 (upper hit first), -1 (lower hit first), 0 (vertical / flat)
        - 'touch_type'   : 'upper' | 'lower' | 'vertical'
        - 'touch_idx'    : positional index where the barrier was touched
        - 'ret'          : realized return at the touch point
        - 'barrier_upper', 'barrier_lower' : the price levels used
        - 'holding_bars' : bars actually held until touch

    Note: the last `max_holding` rows won't have a full forward window
    to evaluate and will have NaN label/touch fields — drop before
    training, same as any other warm-up/cool-down region.
    """
    df = df.copy()
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain column: {price_col}")

    close = df[price_col].values
    n = len(df)

    if volatility is None:
        log_ret = np.log(df[price_col] / df[price_col].shift(1))
        volatility = log_ret.rolling(vol_window).std()
    vol = volatility.values

    labels = np.full(n, np.nan)
    touch_type = np.array([None] * n, dtype=object)
    touch_idx = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    barrier_upper = np.full(n, np.nan)
    barrier_lower = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(vol[i]) or vol[i] == 0:
            continue
        if i + max_holding >= n:
            continue  # not enough forward data for a full vertical window

        upper = close[i] * (1 + pt_mult * vol[i])
        lower = close[i] * (1 - sl_mult * vol[i])
        barrier_upper[i] = upper
        barrier_lower[i] = lower

        window = close[i + 1 : i + 1 + max_holding]
        upper_hit = np.where(window >= upper)[0]
        lower_hit = np.where(window <= lower)[0]

        first_upper = upper_hit[0] if len(upper_hit) else np.inf
        first_lower = lower_hit[0] if len(lower_hit) else np.inf

        if first_upper == np.inf and first_lower == np.inf:
            # vertical barrier: hold to the end of the window
            exit_offset = max_holding - 1
            touch_type[i] = "vertical"
        elif first_upper < first_lower:
            exit_offset = int(first_upper)
            touch_type[i] = "upper"
        else:
            exit_offset = int(first_lower)
            touch_type[i] = "lower"

        exit_idx = i + 1 + exit_offset
        exit_price = close[exit_idx]
        ret = (exit_price - close[i]) / close[i]

        if touch_type[i] == "vertical" and abs(ret) < min_ret:
            label = 0
        elif touch_type[i] == "upper":
            label = 1
        elif touch_type[i] == "lower":
            label = -1
        else:
            label = int(np.sign(ret))

        labels[i] = label
        touch_idx[i] = exit_idx
        rets[i] = ret
        holding_bars[i] = exit_offset + 1

    result = pd.DataFrame(
        {
            "label": labels,
            "touch_type": touch_type,
            "touch_idx": touch_idx,
            "ret": rets,
            "barrier_upper": barrier_upper,
            "barrier_lower": barrier_lower,
            "holding_bars": holding_bars,
        },
        index=df.index,
    )
    return result