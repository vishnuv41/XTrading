"""
Walk-forward validation
---------------------------
Time-respecting train/test splits: train always precedes test, and the
window advances chronologically. Prevents the classic leakage bug of
random K-fold on time series (training on the future to predict the past).

Two modes:
    expanding : train window grows each fold (train_size ignored after
                the first fold — always starts at index 0)
    rolling   : train window is a fixed size, sliding forward each fold
"""

import numpy as np


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    train_size: int = None,
    test_size: int = None,
    gap: int = 0,
    mode: str = "expanding",
):
    """
    Generate (train_idx, test_idx) index arrays for walk-forward validation.

    Parameters
    ----------
    n_samples : total number of rows in the dataset.
    n_splits : number of folds.
    train_size : initial (or fixed, for 'rolling') training window size.
        If None, computed automatically to fit n_splits folds evenly.
    test_size : size of each test fold. If None, computed automatically.
    gap : number of bars skipped between train end and test start —
        use this to match your label's forward-looking window (e.g. the
        triple-barrier's max_holding) so the test set can't contain any
        bar whose label was computed using information from the train
        set's boundary.
    mode : 'expanding' or 'rolling'.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray positional indices.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")

    if test_size is None:
        test_size = n_samples // (n_splits + 1)
    if train_size is None:
        train_size = test_size

    indices = np.arange(n_samples)

    for i in range(n_splits):
        test_start = train_size + gap + i * test_size
        test_end = test_start + test_size
        if test_end > n_samples:
            break

        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, test_start - gap - train_size)

        train_end = test_start - gap
        if train_end <= train_start:
            continue

        train_idx = indices[train_start:train_end]
        test_idx = indices[test_start:test_end]
        yield train_idx, test_idx