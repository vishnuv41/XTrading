"""
Ichimoku Kinko Hyo (Ichimoku Cloud)
------------------------------------
Multi-line trend/support-resistance system. Useful to the regime detector
as a cloud-thickness / price-vs-cloud signal, and to the strategy engine
as a TK-cross entry signal.

Expected input: DataFrame with columns ['high', 'low', 'close'].
"""

import pandas as pd


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    chikou_shift: int = 26,
) -> pd.DataFrame:
    """
    Compute the Ichimoku Cloud components.

    Returns
    -------
    DataFrame (copy of input) with added columns:
        - 'ichimoku_tenkan'   : conversion line (fast)
        - 'ichimoku_kijun'    : base line (slow)
        - 'ichimoku_senkou_a' : leading span A, projected forward
        - 'ichimoku_senkou_b' : leading span B, projected forward
        - 'ichimoku_chikou'   : lagging span (close shifted back)
        - 'ichimoku_cloud_thickness' : |senkou_a - senkou_b|, a rough
          volatility/consolidation proxy useful to the regime detector

    Note: senkou_a/b are shifted `kijun_period` bars into the future
    relative to the bar they're calculated from (standard convention),
    so the last `kijun_period` rows will show forward-projected values
    beyond the current close.
    """
    df = df.copy()
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    high, low, close = df["high"], df["low"], df["close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(kijun_period)

    chikou = close.shift(-chikou_shift)

    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_senkou_a"] = senkou_a
    df["ichimoku_senkou_b"] = senkou_b
    df["ichimoku_chikou"] = chikou
    df["ichimoku_cloud_thickness"] = (senkou_a - senkou_b).abs()

    return df