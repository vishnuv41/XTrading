"""
CatBoost model wrapper
--------------------------
Same interface and same local-label-space handling as XGBoostModel /
LightGBMModel (see _label_utils.py). CatBoost's native handling of
categorical features (e.g. session flags, day-of-week as category) is
the main reason to keep it in the mix rather than relying on GBMs alone.
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ._label_utils import LocalLabelSpace


class CatBoostModel:
    """Classification wrapper around catboost.CatBoostClassifier."""

    def __init__(self, params: dict = None, num_class: int = None, cat_features: list = None, class_weight: str = None):
        default_params = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 50,  # print progress every 50 iterations instead of
                            # nothing — a silent multi-minute fit looks
                            # identical to a hang and invites a premature
                            # Ctrl+C that kills the whole training run.
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.class_weight = class_weight  # None or "balanced"
        self.num_class = num_class
        self.cat_features = cat_features or []
        self.model = None
        self.feature_names_ = None
        self.label_space_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds: int = None):
        self.feature_names_ = list(X.columns)
        params = dict(self.params)

        self.label_space_ = LocalLabelSpace(self.num_class or 2)
        y_encoded = self.label_space_.encode(y)
        n_present = self.label_space_.n_present

        params["loss_function"] = "MultiClass" if n_present > 2 else "Logloss"
        if early_stopping_rounds:
            params["early_stopping_rounds"] = early_stopping_rounds

        self.model = CatBoostClassifier(**params)
        sample_weight = None
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y_encoded)
        train_pool = Pool(X, y_encoded, cat_features=self.cat_features, weight=sample_weight)

        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            y_val_encoded = np.array(
                [np.searchsorted(self.label_space_.present_classes_, v) for v in np.asarray(y_val)]
            )
            eval_pool = Pool(X_val, y_val_encoded, cat_features=self.cat_features)

        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
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
            self.model.get_feature_importance(), index=self.feature_names_
        ).sort_values(ascending=False)

    def save(self, path: str):
        self._check_fitted()
        self.model.save_model(path)
        with open(path + ".meta.json", "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names_,
                    "num_class": self.num_class,
                    "cat_features": self.cat_features,
                    "present_classes": self.label_space_.present_classes_.tolist(),
                },
                f,
            )

    def load(self, path: str):
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        self.feature_names_ = meta["feature_names"]
        self.num_class = meta["num_class"]
        self.cat_features = meta["cat_features"]
        self.label_space_ = LocalLabelSpace(self.num_class)
        self.label_space_.present_classes_ = np.array(meta["present_classes"])

        self.model = CatBoostClassifier()
        self.model.load_model(path)
        return self

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")