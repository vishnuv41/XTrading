"""
Local label-space utility
--------------------------
Shared by xgboost_model.py, lightgbm_model.py, and catboost_model.py.

Problem it solves:
Triple-barrier labels are typically {-1, 0, 1} (or meta-labels {0, 1}).
Depending on the fold/window, not all classes may be present in y — e.g.
a quiet fold might only contain {0, 1}. Boosting libraries either reject
non-contiguous labels (xgboost) or silently change their inferred number
of classes (all three), which breaks a fixed-width predict_proba output
that ensemble.py depends on to average across models/folds.

LocalLabelSpace fixes this by:
  1. encode(y)   -> remaps whatever labels ARE present in this fold to a
                    contiguous 0..n_present-1 range for fitting.
  2. expand_proba(narrow) -> takes the model's [n, n_present] proba matrix
                    and expands it back to a fixed [n, num_class] matrix,
                    zero-filling any class that was absent from this fold,
                    so shapes are always consistent across folds/wrappers.
"""

import numpy as np


class LocalLabelSpace:
    """
    Parameters
    ----------
    num_class : int
        The full/global number of classes the caller expects
        (e.g. 3 for {-1,0,1} triple-barrier labels, 2 for meta-labels).
        This is the width of the arrays returned by expand_proba,
        regardless of how many classes are actually present in a
        given fit() call.
    """

    def __init__(self, num_class: int):
        if num_class is None or num_class < 2:
            raise ValueError(f"num_class must be >= 2, got {num_class}")
        self.num_class = int(num_class)
        self.present_classes_ = None  # sorted array of raw labels seen in this fold

    def encode(self, y) -> np.ndarray:
        """
        Fit on y's unique values and return y remapped to 0..n_present-1
        (contiguous), preserving sort order of the raw labels.
        """
        y_arr = np.asarray(y)
        self.present_classes_ = np.unique(y_arr)
        # index of each label within the sorted present_classes_ array
        encoded = np.searchsorted(self.present_classes_, y_arr)
        return encoded

    @property
    def n_present(self) -> int:
        if self.present_classes_ is None:
            raise RuntimeError("encode() must be called before n_present is available.")
        return len(self.present_classes_)

    def expand_proba(self, narrow: np.ndarray) -> np.ndarray:
        """
        Expand a [n_samples, n_present] proba matrix (in local/encoded
        class order) into a fixed [n_samples, num_class] matrix.

        Any global class absent from present_classes_ gets a zero-filled
        column. Handles the binary-model edge case where narrow may come
        back as a 1-D array of P(class=1) instead of a 2-column matrix.
        """
        if self.present_classes_ is None:
            raise RuntimeError("encode()/load() must set present_classes_ before expand_proba().")

        narrow = np.asarray(narrow)
        n_samples = narrow.shape[0]

        # Normalize binary 1-D output (P of positive class) to 2 columns
        if narrow.ndim == 1:
            narrow = np.column_stack([1.0 - narrow, narrow])

        wide = np.zeros((n_samples, self.num_class), dtype=float)

        # Map each local column back to its global class index.
        global_idx = self._to_global_index(self.present_classes_)

        for local_col, g in enumerate(global_idx):
            wide[:, g] = narrow[:, local_col]

        return wide

    def _to_global_index(self, raw_labels: np.ndarray) -> np.ndarray:
        """
        Map raw label values to global 0..num_class-1 indices.

        Supports the two conventions used in this project:
          - already-contiguous {0, 1, ..., num_class-1}
          - triple-barrier {-1, 0, 1} for num_class == 3
        """
        raw_labels = np.asarray(raw_labels)

        if raw_labels.min() >= 0 and raw_labels.max() < self.num_class:
            return raw_labels.astype(int)

        if self.num_class == 3 and set(np.unique(raw_labels)).issubset({-1, 0, 1}):
            return (raw_labels + 1).astype(int)

        raise ValueError(
            f"Cannot map raw labels {raw_labels.tolist()} into global range "
            f"[0, {self.num_class}). Remap labels to a supported convention "
            f"before fitting (see ml/train.py)."
        )
