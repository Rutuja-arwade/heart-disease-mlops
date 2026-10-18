"""Feature engineering and the reusable preprocessing pipeline.

Everything that transforms raw clinical attributes into model inputs lives in
a single scikit-learn Pipeline, which is pickled together with the estimator.
Training, batch scoring and the API therefore share one code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds two domain-derived cardiology features.

    * ``hr_achieved_ratio``  - max heart rate achieved / age-predicted max
      (220 - age). Low values indicate chronotropic incompetence, a known
      marker of coronary disease.
    * ``rate_pressure_index`` - thalach x resting systolic BP / 1000, a proxy
      for myocardial oxygen demand (rate-pressure product).

    ``enabled`` is exposed as a hyperparameter so the tuning search can decide
    whether the engineered features help.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        if self.enabled:
            predicted_max_hr = (220 - X["age"]).clip(lower=1)
            X["hr_achieved_ratio"] = X["thalach"] / predicted_max_hr
            X["rate_pressure_index"] = X["thalach"] * X["trestbps"] / 1000.0
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            names += config.ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)


def select_numeric(X: pd.DataFrame) -> list[str]:
    """Continuous columns plus whichever engineered columns are present."""
    return [c for c in config.CONTINUOUS_FEATURES + config.ENGINEERED_FEATURES if c in X.columns]


def select_categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in config.CATEGORICAL_FEATURES if c in X.columns]


def select_binary(X: pd.DataFrame) -> list[str]:
    return [c for c in config.BINARY_FEATURES if c in X.columns]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        [
            ("num", numeric, select_numeric),
            ("cat", categorical, select_categorical),
            ("bin", binary, select_binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, engineered: bool = True) -> Pipeline:
    """Full model: feature builder -> preprocessing -> classifier."""
    return Pipeline(
        [
            ("features", ClinicalFeatureBuilder(enabled=engineered)),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
