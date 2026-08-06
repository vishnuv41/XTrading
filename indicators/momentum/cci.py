"""
Commodity Channel Index (CCI)
-------------------------------
Measures deviation of typical price from its moving average, normalized
by mean absolute deviation. Unbounded oscillator; commonly read against
+/-100 thresholds for overbought/oversold in the strategy engine.

Expected input: DataFrame with columns ['high', 'low', 'close'].
"""

import numpy as np
import pandas as pd


def calculate_cci(df: pd.DataFrame, period: int = 20, constant: float = 0.015) -> pd.DataFrame:
    """
    Compute the Commodity Channel Index.

    Parameters
    ----------
    period : lookback window for the moving average and mean deviation (default 20).
    constant : scaling constant, 0.015 by convention so ~70-80% of values
        fall within +/-100.

    Returns
    -------
    DataFrame (copy of input) with added column:
        - 'cci'
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(period).mean()
    mean_dev = typical_price.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    df["cci"] = (typical_price - sma_tp) / (constant * mean_dev)

    return df