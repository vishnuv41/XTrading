"""
Ensemble model (stacking / blending)
----------------------------------------
Combines XGBoost, LightGBM, CatBoost (or any wrapper with the same
fit/predict_proba interface) via simple averaging or stacking with a
logistic-regression meta-learner trained on out-of-fold predictions.

Stacking mode requires a `cv` argument: an iterable of (train_idx,
test_idx) index-array pairs, e.g. from ml.validation.walk_forward or
ml.validation.purged_kfold. Passing a leakage-safe (time-respecting)
splitter here matters — a random KFold on time series data will leak
information across the train/test boundary through the meta-learner.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class EnsembleModel:
    def __init__(self, base_models: dict, blend_method: str = "stacking", meta_learner=None):
        """
        Parameters
        ----------
        base_models : dict of {name: model_instance}, each exposing
            .fit(X, y) and .predict_proba(X) -> ndarray[n, n_classes].
        blend_method : 'average' (simple mean of predict_proba across
            base models) or 'stacking' (meta-learner on OOF predictions).
        meta_learner : sklearn-compatible classifier for stacking mode.
            Defaults to LogisticRegression if None.
        """
        self.base_models = base_models
        self.blend_method = blend_method
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        self.fitted_base_models_ = {}
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, cv=None):
        # Fixed class-space width, taken from the base models' own
        # num_class rather than np.unique(y): a class can legitimately
        # be absent from this particular dataset/fold, and inferring
        # width from what's present would silently shrink the output
        # (see ml/models/_label_utils.py for the same issue at the base-
        # model level).
        first_model = next(iter(self.base_models.values()))
        n_classes = getattr(first_model, "num_class", None) or len(np.unique(y))
        self.classes_ = np.arange(n_classes)

        if self.blend_method == "stacking":
            if cv is None:
                raise ValueError("Stacking requires a `cv` splitter (list of (train_idx, test_idx)).")

            oof_preds = np.zeros((len(X), n_classes * len(self.base_models)))
            oof_filled = np.zeros(len(X), dtype=bool)

            n_folds = len(cv)
            for fold_i, (fold_train_idx, fold_test_idx) in enumerate(cv, start=1):
                X_train, y_train = X.iloc[fold_train_idx], y.iloc[fold_train_idx]
                X_test = X.iloc[fold_test_idx]

                for j, (name, model) in enumerate(self.base_models.items()):
                    print(f"    fold {fold_i}/{n_folds} — fitting {name} "
                          f"({len(fold_train_idx)} train / {len(fold_test_idx)} test rows)...")
                    fold_model = _clone_model(model)
                    fold_model.fit(X_train, y_train)
                    proba = fold_model.predict_proba(X_test)  # already fixed-width, num_class columns
                    oof_preds[fold_test_idx, j * n_classes : (j + 1) * n_classes] = proba
                oof_filled[fold_test_idx] = True

            # train meta-learner only on rows that were actually held out at least once
            print("    fitting meta-learner on out-of-fold predictions...")
            self.meta_learner.fit(oof_preds[oof_filled], y.values[oof_filled])
            self._meta_n_classes = n_classes

        # always refit base models on the full dataset for inference
        for name, model in self.base_models.items():
            print(f"    refitting {name} on the full training set...")
            model.fit(X, y)
            self.fitted_base_models_[name] = model

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_probas = [model.predict_proba(X) for model in self.fitted_base_models_.values()]

        if self.blend_method == "average":
            return np.mean(base_probas, axis=0)

        stacked = np.hstack(base_probas)
        narrow = self.meta_learner.predict_proba(stacked)

        # meta_learner.classes_ can be narrower than the full class space
        # if some class never appeared in the OOF training rows; expand
        # back to fixed width (zero-filled for absent classes) so callers
        # can always index by the fixed LABEL_MAP positions.
        meta_classes = self.meta_learner.classes_
        if len(meta_classes) == len(self.classes_):
            return narrow

        full = np.zeros((narrow.shape[0], len(self.classes_)))
        for i, cls in enumerate(meta_classes):
            full[:, int(cls)] = narrow[:, i]
        return full

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def feature_importance(self) -> pd.DataFrame:
        """Returns per-model feature importances side by side for comparison."""
        importances = {}
        for name, model in self.fitted_base_models_.items():
            if hasattr(model, "feature_importance"):
                importances[name] = model.feature_importance()
        return pd.DataFrame(importances)


def _clone_model(model):
    """Shallow re-instantiation for fold-local training (avoids mutating the shared instance)."""
    cls = model.__class__
    init_kwargs = {}
    if hasattr(model, "params"):
        init_kwargs["params"] = dict(model.params)
    if hasattr(model, "num_class"):
        init_kwargs["num_class"] = model.num_class
    if hasattr(model, "class_weight"):
        init_kwargs["class_weight"] = model.class_weight
    if hasattr(model, "cat_features"):
        init_kwargs["cat_features"] = model.cat_features
    return cls(**init_kwargs)