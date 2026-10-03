"""
strategy/cross_asset.py
----------------------------
Cross-Asset Confirmation Strategy (new — see roadmap in the
architecture review). Hypothesis: a single-asset signal (e.g. BTC BUY)
is more reliable when the broader market — other assets in the pooled
BTC/ETH/SOL universe — is moving the same direction, and less reliable
when it's diverging (BTC signal bullish while ETH/SOL are weak or
falling, suggesting an asset-specific move rather than genuine market-
wide risk-on). This doesn't generate a new entry signal on its own; it
CONFIRMS or VETOES an entry.py signal already generated for the primary
asset, the same way strategy/multi_timeframe.py confirms across
timeframes rather than generating new signals.

Causality note (the review brief was explicit about this): every input
here — RS_{name}/CORR_{name}_{w} from ml.features.cross_asset_features,
and reference_trend_bias — is derived from the SAME bar's simultaneous
close across assets (all trading the same exchange/timeframe), never a
future bar. This is standard cross-sectional confirmation, not
lookahead: at decision time T, "what did BTC/ETH/SOL just do as of T"
is legitimately known. Do NOT feed this module anything derived from a
reference asset's bar that closes AFTER the primary asset's bar T — if
your pooled ingestion has any clock skew between symbols, align via
merge_asof(direction='backward') the same way multi_timeframe.py does,
not a plain positional join.

Required columns:
    entry_signal                          (strategy/entry.py, primary asset)
    RS_{name}                              (ml.features.cross_asset_features,
                                            one or more reference assets already merged in)
Optional:
    CORR_{name}_{w}                        used to down-weight/skip a
                                            reference asset that isn't
                                            actually correlated right now
    reference_trend_bias: dict[str, Series] each reference asset's own
                                            trend_bias (strategy/trend.py),
                                            aligned to df's index — adds
                                            a second, independent
                                            confirmation signal beyond RS
"""

import pandas as pd


def generate_cross_asset_signal(
    df: pd.DataFrame,
    reference_names: list,
    reference_trend_bias: dict = None,
    corr_window: int = 20,
    min_correlation: float = 0.3,
    rs_lookback: int = 10,
    min_agreement_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Confirm/veto df['entry_signal'] against pooled reference-asset state.

    Args:
        reference_names: e.g. ["ETH", "SOL"] — must match the `name` keys
            used when calling ml.features.cross_asset_features.calculate_cross_asset_features
            on this df (so RS_{name}/CORR_{name}_{corr_window} exist).
        reference_trend_bias: optional {name: pd.Series of 'bullish'/'bearish'/'neutral'},
            aligned to df's index. If omitted, confirmation relies on RS trend alone.
        corr_window: which CORR_{name}_{w} column to use for the
            correlation gate (must match a window already computed).
        min_correlation: reference assets with |correlation| below this
            are excluded from voting — an uncorrelated asset agreeing or
            disagreeing is not meaningful confirmation either way.
        rs_lookback: bars over which RS_{name} must be rising (BUY) or
            falling (SELL) to count as that reference "agreeing".
        min_agreement_frac: fraction of ELIGIBLE (sufficiently
            correlated) reference assets that must agree for
            'cross_asset_confirmed' to be True.

    Returns:
        df with new columns:
          'cross_asset_confirmed' - bool
          'cross_asset_reason'     - list[str]
          'final_signal'           - entry_signal where confirmed, else 'HOLD'
    """
    if "entry_signal" not in df.columns:
        raise ValueError("df missing 'entry_signal' — run generate_entry_signal() first")
    if not reference_names:
        raise ValueError("reference_names must contain at least one reference asset")

    reference_trend_bias = reference_trend_bias or {}

    confirmed = []
    reasons_list = []

    for idx in df.index:
        signal = df.at[idx, "entry_signal"]
        if signal == "HOLD":
            confirmed.append(True)  # trivially "confirmed" — nothing to veto
            reasons_list.append([])
            continue

        eligible_votes = []
        reasons = []

        for name in reference_names:
            rs_col = f"RS_{name}"
            corr_col = f"CORR_{name}_{corr_window}"

            if rs_col not in df.columns:
                continue
            if corr_col in df.columns:
                corr_val = df.at[idx, corr_col]
                if pd.isna(corr_val) or abs(corr_val) < min_correlation:
                    continue  # not correlated enough right now to mean anything

            rs_series = df[rs_col]
            pos = df.index.get_loc(idx)
            if pos < rs_lookback:
                continue
            rs_now = rs_series.iloc[pos]
            rs_then = rs_series.iloc[pos - rs_lookback]
            if pd.isna(rs_now) or pd.isna(rs_then):
                continue

            rs_rising = rs_now > rs_then
            agrees_rs = rs_rising if signal == "BUY" else (not rs_rising)

            agrees_trend = None
            if name in reference_trend_bias:
                bias_series = reference_trend_bias[name]
                bias = bias_series.get(idx, None) if hasattr(bias_series, "get") else None
                if bias is not None:
                    agrees_trend = (bias == "bullish") if signal == "BUY" else (bias == "bearish")

            # A reference asset "agrees" if RS confirms, and (if trend
            # bias is available) trend doesn't actively contradict it.
            agrees = agrees_rs and (agrees_trend if agrees_trend is not None else True)
            eligible_votes.append(agrees)
            if agrees:
                reasons.append(f"{name} relative strength + trend confirm {signal.lower()}")
            else:
                reasons.append(f"{name} diverges from {signal.lower()}")

        if not eligible_votes:
            # No reference asset was sufficiently correlated/available —
            # can't confirm OR veto; pass through unconfirmed (caller's
            # min_agreement_frac gate below decides how strict to be).
            confirmed.append(False)
            reasons_list.append(["No sufficiently correlated reference asset available to confirm."])
            continue

        agreement_frac = sum(eligible_votes) / len(eligible_votes)
        is_confirmed = agreement_frac >= min_agreement_frac
        confirmed.append(is_confirmed)
        reasons_list.append(reasons)

    df = df.copy()
    df["cross_asset_confirmed"] = confirmed
    df["cross_asset_reason"] = reasons_list
    df["final_signal"] = df["entry_signal"].where(df["cross_asset_confirmed"], "HOLD")
    return df


if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators import calculate_all_indicators
    from regime import calculate_market_state
    from strategy.entry import generate_entry_signal
    from ml.features.cross_asset_features import calculate_cross_asset_features

    n = 400

    def make_df(seed, drift):
        rng = np.random.default_rng(seed)
        close = np.cumsum(rng.normal(drift, 1, n)) + 100
        return pd.DataFrame({
            "open": close + rng.normal(0, 0.3, n), "high": close + rng.random(n) * 1.5,
            "low": close - rng.random(n) * 1.5, "close": close,
            "volume": rng.integers(100, 5000, n),
        })

    btc = make_df(1, 0.05)
    eth = make_df(2, 0.05)   # correlated drift direction
    sol = make_df(3, -0.05)  # diverging

    btc = calculate_all_indicators(btc)
    btc = calculate_market_state(btc)
    btc = generate_entry_signal(btc)
    btc = calculate_cross_asset_features(btc, {"ETH": eth, "SOL": sol})

    result = generate_cross_asset_signal(btc, reference_names=["ETH", "SOL"])
    print(result["entry_signal"].value_counts())
    print(result["cross_asset_confirmed"].value_counts())
    non_hold = result[result["entry_signal"] != "HOLD"]
    if len(non_hold):
        print(non_hold[["entry_signal", "cross_asset_confirmed", "final_signal", "cross_asset_reason"]].head(8).to_string())
