"""
Meta-labeling
-------------
Lopez de Prado's meta-labeling: given a primary model's predicted
trade *side* (+1 long / -1 short, e.g. from strategy/signal.py or a
simple rule) and the triple-barrier *outcome* labels, produce a
secondary binary label describing whether the primary side would have
been profitable ('meta_label' = 1) or not (0).

This lets a secondary classifier learn "should I act on this signal"
(sizing / filtering) separately from "which direction" (the primary
model), which is generally more robust than a single 3-class model.
"""

import numpy as np
import pandas as pd


def generate_meta_labels(
    side: pd.Series,
    barrier_labels: pd.DataFrame,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Combine a primary-model side prediction with triple-barrier outcome
    labels into a binary meta-label.

    Parameters
    ----------
    side : pd.Series
        Primary model's predicted direction per bar: > 0 for long,
        < 0 for short, 0 for no position. Must share `barrier_labels`'
        index (or be alignable to it).
    barrier_labels : pd.DataFrame
        Output of `triple_barrier_labels` — must contain `label_col`
        with values in {-1, 0, 1} (and ideally 'ret').
    label_col : str
        Column in `barrier_labels` holding the realized outcome
        direction (-1/0/1).

    Returns
    -------
    pd.DataFrame
        Copy of `barrier_labels` with two extra columns:
            side        : the aligned primary-model side, sign-normalized to {-1, 0, 1}
            meta_label  : 1 if side agrees in sign with the realized outcome
                          and outcome != 0, else 0. NaN where either input is NaN.
    """
    out = barrier_labels.copy()

    side_aligned = side.reindex(out.index)
    side_sign = np.sign(side_aligned)
    outcome = out[label_col]

    meta = pd.Series(np.nan, index=out.index, dtype="float64")
    valid = side_sign.notna() & outcome.notna()

    # Correct call: side and outcome agree in sign and outcome is non-flat.
    correct = valid & (outcome != 0) & (np.sign(side_sign) == np.sign(outcome))
    # Any other valid case (wrong direction, no position, or flat outcome) = 0.
    meta.loc[valid] = 0.0
    meta.loc[correct] = 1.0

    out["side"] = side_sign
    out["meta_label"] = meta
    return out
