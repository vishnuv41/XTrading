"""
On-Balance Volume (OBV)

Cumulative volume flow indicator: adds volume on up-closes, subtracts on
down-closes. Divergence between OBV and price (e.g. price making new highs
while OBV doesn't) is a classic early-warning signal used as an ML feature
for trend exhaustion.
"""

import pandas as pd
import numpy as np


def calculate_obv(df: pd.DataFrame, close_col: str = "close",
                   volume_col: str = "volume") -> pd.DataFrame:
    """
    Add an OBV column.

    Args:
        df: DataFrame with 'close' and 'volume' columns.

    Returns:
        df with a new column 'OBV'.
    """
    for col in (close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    direction = np.sign(df[close_col].diff()).fillna(0)
    df["OBV"] = (direction * df[volume_col]).cumsum()
    return df


if __name__ == "__main__":
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    dummy = pd.DataFrame({
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    dummy = calculate_obv(dummy)
    print(dummy.tail())