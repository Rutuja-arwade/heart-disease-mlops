"""Feature engineering and preprocessing pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import ClinicalFeatureBuilder, build_preprocessor


def test_feature_builder_adds_engineered_columns(xy):
    X, _ = xy
    out = ClinicalFeatureBuilder(enabled=True).fit_transform(X.head(5))
    for col in config.ENGINEERED_FEATURES:
        assert col in out.columns
    row = X.iloc[0]
    assert out["hr_achieved_ratio"].iloc[0] == pytest.approx(row["thalach"] / (220 - row["age"]))
    assert out["rate_pressure_index"].iloc[0] == pytest.approx(row["thalach"] * row["trestbps"] / 1000)


def test_feature_builder_disabled_is_identity(xy):
    X, _ = xy
    out = ClinicalFeatureBuilder(enabled=False).fit_transform(X.head(5))
    pd.testing.assert_frame_equal(out, X.head(5))


def test_feature_builder_does_not_mutate_input(xy):
    X, _ = xy
    before = X.copy()
    ClinicalFeatureBuilder().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_preprocessor_output_has_no_missing_values(xy):
    X, _ = xy
    assert X.isna().any().any(), "fixture should contain the real missing ca/thal values"
    features = ClinicalFeatureBuilder().fit_transform(X)
    out = build_preprocessor().fit_transform(features)
    assert not np.isnan(out).any()


def test_preprocessor_scales_and_one_hot_encodes(xy):
    X, _ = xy
    features = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(features)
    out = pre.transform(features)
    names = list(pre.get_feature_names_out())
    # 6 continuous + 2 engineered scaled, cp(4)+restecg(3)+slope(3)+thal(3) one-hot, 3 binary
    assert out.shape == (len(X), 8 + 13 + 3)
    assert "cp_4.0" in names and "hr_achieved_ratio" in names
    age = out[:, names.index("age")]
    assert age.mean() == pytest.approx(0, abs=1e-6)
    assert age.std() == pytest.approx(1, abs=1e-6)


def test_preprocessor_ignores_unseen_category(xy):
    X, _ = xy
    features = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(features)
    odd = features.head(1).copy()
    odd["thal"] = 5.0  # never seen during fit
    assert pre.transform(odd).shape[0] == 1


def test_preprocessor_without_engineered_features(xy):
    X, _ = xy
    features = ClinicalFeatureBuilder(enabled=False).fit_transform(X)
    out = build_preprocessor().fit_transform(features)
    assert out.shape[1] == 6 + 13 + 3
