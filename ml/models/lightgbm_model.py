"""
LightGBM model wrapper
--------------------------
Same interface and same local-label-space handling as XGBoostModel
(see _label_utils.py) so ensemble.py can treat all three wrappers
interchangeably regardless of which classes a given fold contains.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from ._label_utils import LocalLabelSpace


class LightGBMModel:
    """Classification wrapper around lightgbm.LGBMClassifier."""

    def __init__(self, params: dict = None, num_class: int = None, class_weight: str = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
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
            params["objective"] = "multiclass"
            params["num_class"] = n_present
        else:
            params["objective"] = "binary"

        self.model = lgb.LGBMClassifier(**params)
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
            callbacks = []
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            fit_kwargs["callbacks"] = callbacks
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
        self.model.booster_.save_model(path)
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

        booster = lgb.Booster(model_file=path)
        self.model = _BoosterClassifierAdapter(booster)
        return self

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")


class _BoosterClassifierAdapter:
    """Minimal adapter so a raw loaded lgb.Booster exposes predict_proba/feature_importances_."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw

    @property
    def feature_importances_(self):
        return self.booster.feature_importance()