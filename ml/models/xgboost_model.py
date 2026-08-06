"""
XGBoost model wrapper
-------------------------
Thin, consistent wrapper (fit/predict/predict_proba/save/load/
feature_importance) so ensemble.py can treat this interchangeably with
the LightGBM and CatBoost wrappers.

Labels are locally re-encoded to a contiguous range before fitting (see
_label_utils.LocalLabelSpace) — xgboost's sklearn wrapper silently
overrides an explicit multiclass objective when a fold happens to
contain only 2 distinct label values, then rejects non-contiguous
labels. predict_proba always returns a fixed-width [n, num_class]
matrix (zero-filled for any class absent from this fold) so ensemble.py
can rely on a consistent shape across folds and across the three model
wrappers.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from ._label_utils import LocalLabelSpace


class XGBoostModel:
    """
    Classification wrapper around xgboost.XGBClassifier.

    Works for binary meta-labels (0/1) or multiclass triple-barrier
    labels (typically remapped from {-1,0,1} to {0,1,2} before fitting —
    see ml/train.py for the remap step).
    """

    def __init__(self, params: dict = None, num_class: int = None, class_weight: str = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.num_class = num_class
        self.class_weight = class_weight  # None or "balanced"
        self.model = None
        self.feature_names_ = None
        self.label_space_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds: int = None):
        self.feature_names_ = list(X.columns)
        params = dict(self.params)

        self.label_space_ = LocalLabelSpace(self.num_class or 2)
        y_encoded = self.label_space_.encode(y)
        n_present = self.label_space_.n_present

        if n_present > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_present
            params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"

        self.model = xgb.XGBClassifier(**params)
        fit_kwargs = {}
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_encoded)
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            y_val_encoded = np.array(
                [np.searchsorted(self.label_space_.present_classes_, v) for v in np.asarray(y_val)]
            )
            fit_kwargs["eval_set"] = [(X_val, y_val_encoded)]
            fit_kwargs["verbose"] = False
        self.model.fit(X, y_encoded, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        narrow = self.model.predict_proba(X[self.feature_names_])
        return self.label_space_.expand_proba(narrow)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)

    def save(self, path: str):
        self._check_fitted()
        self.model.save_model(path)
        with open(path + ".meta.json", "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names_,
                    "num_class": self.num_class,
                    "present_classes": self.label_space_.present_classes_.tolist(),
                },
                f,
            )

    def load(self, path: str):
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        self.feature_names_ = meta["feature_names"]
        self.num_class = meta["num_class"]
        self.label_space_ = LocalLabelSpace(self.num_class)
        self.label_space_.present_classes_ = np.array(meta["present_classes"])

        params = dict(self.params)
        n_present = len(self.label_space_.present_classes_)
        if n_present > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_present
            params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"

        self.model = xgb.XGBClassifier(**params)
        self.model.load_model(path)
        return self

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")