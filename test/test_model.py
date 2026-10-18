"""Model building, evaluation and packaging tests."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from src.evaluate import compute_metrics, feature_importance
from src.features import build_pipeline
from src.predict import predict_records, risk_level, to_frame
from src.train import model_specs


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(max_iter=2000, solver="liblinear"), RandomForestClassifier(n_estimators=50, random_state=0)],
)
def test_pipeline_fits_and_outputs_probabilities(xy, estimator):
    X, y = xy
    pipeline = build_pipeline(estimator).fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_majority_baseline_in_cv(xy):
    X, y = xy
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"))
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()
    assert auc > 0.85


def test_compute_metrics_perfect_and_keys():
    y = np.array([0, 1, 1, 0])
    metrics = compute_metrics(y, y, np.array([0.1, 0.9, 0.8, 0.2]))
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert all(v == pytest.approx(1.0) for v in metrics.values())


def test_compute_metrics_handles_no_positive_predictions():
    metrics = compute_metrics(np.array([0, 1]), np.array([0, 0]), np.array([0.2, 0.4]))
    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0


def test_pickle_round_trip_is_identical(tmp_path, fitted_pipeline, xy):
    X, _ = xy
    path = tmp_path / "m.joblib"
    joblib.dump(fitted_pipeline, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict_proba(X), fitted_pipeline.predict_proba(X))


def test_feature_importance_for_linear_model(fitted_pipeline):
    importance = feature_importance(fitted_pipeline)
    assert importance is not None and len(importance) == 24
    assert importance.abs().is_monotonic_decreasing


def test_quick_specs_have_single_candidate():
    for spec in model_specs(quick=True):
        assert all(len(v) == 1 for v in spec.param_space.values())
    names = [s.name for s in model_specs()]
    assert {"logistic_regression", "random_forest"} <= set(names)


def test_predict_records_contract(fitted_pipeline, patient, healthy_patient):
    sick, healthy = predict_records(fitted_pipeline, [patient, healthy_patient])
    assert sick["prediction"] == 1 and healthy["prediction"] == 0
    for result in (sick, healthy):
        assert 0.5 <= result["confidence"] <= 1.0
        assert result["label"] in {"heart_disease", "no_heart_disease"}


def test_predict_handles_missing_optional_fields(fitted_pipeline, patient):
    patient.pop("ca")
    patient.pop("thal")
    frame = to_frame([patient])
    assert list(frame.columns) == config.FEATURE_ORDER
    assert frame[["ca", "thal"]].isna().all().all()
    assert predict_records(fitted_pipeline, [patient])[0]["prediction"] in (0, 1)


@pytest.mark.parametrize("p,expected", [(0.05, "low"), (0.3, "moderate"), (0.59, "moderate"), (0.6, "high")])
def test_risk_level_bands(p, expected):
    assert risk_level(p) == expected


def test_exported_model_artifacts_are_consistent():
    """The committed champion model must load and match its metadata."""
    if not config.MODEL_FILE.exists():
        pytest.skip("model not trained yet")
    import json

    model = joblib.load(config.MODEL_FILE)
    meta = json.loads(config.MODEL_METADATA_FILE.read_text())
    assert meta["features"] == config.FEATURE_ORDER
    assert meta["test_metrics"]["test_roc_auc"] > 0.8
    assert hasattr(model, "predict_proba")


def test_training_entrypoint_quick_mode(tmp_path, monkeypatch):
    """End-to-end smoke test of src.train with an isolated MLflow store."""
    import mlflow

    from src import train

    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT", "pytest-smoke")
    model_mtime = config.MODEL_FILE.stat().st_mtime if config.MODEL_FILE.exists() else None

    result = train.run(quick=True, register=False)

    assert result["model_family"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert result["test_metrics"]["test_roc_auc"] > 0.75
    runs = mlflow.search_runs(experiment_names=["pytest-smoke"])
    # 4 parent runs + 4 trial runs + 1 selection run
    assert len(runs) == 9
    assert {"metrics.cv_roc_auc_mean", "metrics.test_recall"} <= set(runs.columns)
    # quick mode must never overwrite the exported production model
    if model_mtime is not None:
        assert config.MODEL_FILE.stat().st_mtime == model_mtime
