"""
Probability calibration
---------------------------
Tree-ensemble probabilities are usually miscalibrated (overconfident
near 0/1). This matters directly for position sizing (Kelly criterion,
confidence-scaled sizing) which needs P(win) to actually mean P(win).

Two methods:
    'isotonic' : non-parametric, more flexible, needs more data (~1000+
                 samples per class) or it overfits the calibration curve.
    'sigmoid'  : Platt scaling — a 1D logistic regression on the raw
                 score. Fewer samples needed, but assumes a sigmoid-
                 shaped miscalibration.

Fit calibration on a held-out set (never the same data the base model
trained on) to avoid re-leaking training-set overconfidence into the
"calibrated" output.
"""

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class _ConstantCalibrator:
    """Fallback for sigmoid calibration when a class has 0 or 100% of the
    holdout (LogisticRegression can't fit a single-class target)."""
    def __init__(self, value: float):
        self.value = value


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError("method must be 'isotonic' or 'sigmoid'")
        self.method = method
        self.calibrators_ = {}  # one per class for multiclass, or single for binary
        self.n_classes_ = None

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray):
        """
        Parameters
        ----------
        raw_proba : ndarray [n_samples, n_classes] of uncalibrated
            probabilities from the base/ensemble model, on a held-out set.
        y_true : true labels for that same held-out set.
        """
        raw_proba = np.asarray(raw_proba)
        n_classes = raw_proba.shape[1]
        self.n_classes_ = n_classes

        # IMPORTANT: column index == class index, always — this is a
        # contract with EnsembleModel/base models (they return
        # fixed-width, LABEL_MAP-ordered columns even when a class is
        # absent from the data). We must therefore calibrate ALL
        # n_classes columns, not just np.unique(y_true): if a class is
        # rare/absent from the holdout set, np.unique(y_true) silently
        # returns fewer classes than columns, and zipping them together
        # positionally (enumerate(classes)) fits each calibrator against
        # the WRONG probability column and leaves the true last column(s)
        # permanently zero after transform — making that class
        # impossible to predict and corrupting the rest. Iterating
        # class indices 0..n_classes-1 directly avoids this entirely.
        self.classes_ = np.arange(n_classes)

        for cls in self.classes_:
            binary_y = (y_true == cls).astype(int)
            scores = raw_proba[:, cls]

            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(scores, binary_y)
            else:
                calibrator = LogisticRegression()
                # A class with zero positive examples in the holdout
                # (binary_y all 0) can't fit a real sigmoid — fall back
                # to a constant "always low" calibrator rather than
                # letting LogisticRegression raise.
                if binary_y.sum() == 0 or binary_y.sum() == len(binary_y):
                    calibrator = _ConstantCalibrator(float(binary_y.mean()))
                else:
                    calibrator.fit(scores.reshape(-1, 1), binary_y)

            self.calibrators_[cls] = calibrator

        return self

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities; output is renormalized to sum to 1 per row."""
        raw_proba = np.asarray(raw_proba)
        calibrated = np.zeros_like(raw_proba, dtype=float)

        for cls in self.classes_:
            scores = raw_proba[:, cls]
            calibrator = self.calibrators_[cls]
            if isinstance(calibrator, _ConstantCalibrator):
                calibrated[:, cls] = calibrator.value
            elif self.method == "isotonic":
                calibrated[:, cls] = calibrator.predict(scores)
            else:
                calibrated[:, cls] = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

    def fit_transform(self, raw_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(raw_proba, y_true)
        return self.transform(raw_proba)

    def save(self, path: str):
        """
        Persist this fitted calibrator (the object itself, via joblib —
        the per-class IsotonicRegression/LogisticRegression sub-models
        pickle cleanly). Inference code can then do
        `calibrator = ProbabilityCalibrator.load(path)` and call
        `.transform(...)` directly, with no separate reconstruction step.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ProbabilityCalibrator":
        """Reload a calibrator saved with .save(). Returns a ready-to-use instance."""
        return joblib.load(path)