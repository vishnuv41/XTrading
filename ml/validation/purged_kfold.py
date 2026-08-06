"""
Purged K-Fold with embargo
-------------------------------
Standard K-Fold leaks on financial time series because labels are built
from *overlapping forward windows* (e.g. triple-barrier's max_holding).
A training sample whose label window extends into the test period has
effectively seen test-period information. This splitter:

  1. Purges any training sample whose event window [i, t1[i]] overlaps
     the test fold's time range.
  2. Embargoes a further small window of samples immediately after each
     test fold from training, since the test fold's outcome can causally
     influence prices/labels just after it (autocorrelation, book effects).

Reference: Lopez de Prado, "Advances in Financial Machine Learning", ch. 7.
"""

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01):
        """
        Parameters
        ----------
        n_splits : number of folds.
        t1 : Series aligned to the training data's positional order, where
            t1.iloc[i] = the positional index at which observation i's
            label-defining event ends (e.g. `touch_idx` from
            triple_barrier_labels). Required — this is what distinguishes
            purged K-fold from plain K-fold.
        pct_embargo : fraction of total samples embargoed after each test
            fold (default 1%).
        """
        if t1 is None:
            raise ValueError("PurgedKFold requires `t1` (event end positions).")
        self.n_splits = n_splits
        self.t1 = t1.reset_index(drop=True)
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        embargo_size = int(n * self.pct_embargo)

        fold_bounds = [(f[0], f[-1] + 1) for f in np.array_split(indices, self.n_splits)]

        for start, end in fold_bounds:
            test_idx = indices[start:end]

            # a train sample is purged if its own event window overlaps
            # the test fold's time span at all
            test_span_start = start
            test_span_end = max(self.t1.iloc[start:end].max(), end - 1)

            train_mask = np.ones(n, dtype=bool)
            train_mask[start:end] = False  # exclude the test fold itself

            for i in indices:
                if not train_mask[i]:
                    continue
                event_end = self.t1.iloc[i]
                if np.isnan(event_end):
                    continue
                event_end = int(event_end)
                # overlap check: [i, event_end] intersects [test_span_start, test_span_end]
                if i <= test_span_end and event_end >= test_span_start:
                    train_mask[i] = False

            # embargo: drop the embargo_size samples immediately following the test fold
            embargo_end = min(n, end + embargo_size)
            train_mask[end:embargo_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits