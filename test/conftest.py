"""Shared fixtures. Tests never touch the network or the committed model."""

from __future__ import annotations

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import clean_data, load_raw_data, split_features_target
from src.features import build_pipeline

RAW_SNIPPET = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4
"""


@pytest.fixture
def raw_frame(tmp_path) -> pd.DataFrame:
    path = tmp_path / "raw.data"
    path.write_text(RAW_SNIPPET)
    return load_raw_data(path)


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """The real cleaned dataset when available, otherwise rebuilt from the raw file."""
    if config.CLEAN_DATA_FILE.exists():
        return pd.read_csv(config.CLEAN_DATA_FILE)
    df, _ = clean_data(load_raw_data(config.RAW_DATA_FILE))
    return df


@pytest.fixture(scope="session")
def xy(clean_df):
    return split_features_target(clean_df)


@pytest.fixture(scope="session")
def fitted_pipeline(xy):
    X, y = xy
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"), engineered=True)
    return pipeline.fit(X, y)


@pytest.fixture
def patient() -> dict:
    return {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 150,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": 7,
    }


@pytest.fixture
def healthy_patient() -> dict:
    return {
        "age": 37,
        "sex": 0,
        "cp": 3,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 185,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }


@pytest.fixture
def model_files(tmp_path, fitted_pipeline, monkeypatch):
    """Write a freshly trained model to a temp dir and point the API at it."""
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "model_metadata.json"
    joblib.dump(fitted_pipeline, model_path)
    meta_path.write_text(
        json.dumps({"model_version": "test-1", "model_family": "logistic_regression", "decision_threshold": 0.5})
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(meta_path))
    return model_path, meta_path


@pytest.fixture
def client(model_files):
    from fastapi.testclient import TestClient

    from src.api.app import app

    with TestClient(app) as test_client:
        yield test_client


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


np.random.seed(config.RANDOM_STATE)
