"""Model development with hyperparameter search and MLflow experiment tracking.

For every model family a parent MLflow run is created; each hyperparameter
candidate evaluated by the search is logged as a nested child run (params +
cross-validated metrics). The parent run stores the best configuration,
hold-out test metrics, diagnostic plots, the CV results table and the fitted
pipeline. The best family (by mean CV ROC-AUC) is registered as the champion
and exported to ``models/`` for serving.

    python -m src.train              # full search (used locally and in CI)
    python -m src.train --quick      # tiny search for smoke tests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, roc_curve  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src import __version__, config  # noqa: E402
from src.data import load_clean_data, sha256_of, split_features_target  # noqa: E402
from src.evaluate import (  # noqa: E402
    CV_SCORING,
    compute_metrics,
    feature_importance,
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_pr,
    plot_roc,
)
from src.features import build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_space: dict
    search: str = "grid"  # "grid" or "random"
    n_iter: int = 20
    description: str = ""


@dataclass
class ModelResult:
    name: str
    pipeline: object
    best_params: dict
    cv_metrics: dict
    test_metrics: dict
    run_id: str
    y_proba: object = field(repr=False, default=None)


def model_specs(quick: bool = False) -> list[ModelSpec]:
    rs = config.RANDOM_STATE
    specs = [
        ModelSpec(
            "baseline_majority",
            DummyClassifier(strategy="most_frequent"),
            {"features__enabled": [False]},
            description="Predicts the majority class; the floor every real model must beat.",
        ),
        ModelSpec(
            "logistic_regression",
            LogisticRegression(solver="liblinear", max_iter=5000, random_state=rs),
            {
                "features__enabled": [True, False],
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
            search="grid",
            description="Interpretable linear baseline; L1/L2 regularisation strength tuned by grid search.",
        ),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(random_state=rs, n_jobs=1),
            {
                "features__enabled": [True, False],
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": [None, "balanced"],
            },
            search="random",
            n_iter=30,
            description="Bagged trees capture non-linear interactions; randomised search over depth/leaf size.",
        ),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=rs),
            {
                "features__enabled": [True, False],
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.7, 1.0],
                "model__min_samples_leaf": [1, 5, 10],
            },
            search="random",
            n_iter=25,
            description="Boosted shallow trees; randomised search over learning rate and tree size.",
        ),
    ]
    if quick:
        for spec in specs:
            spec.param_space = {k: v[:1] for k, v in spec.param_space.items()}
            spec.search, spec.n_iter = "grid", 1
            if isinstance(spec.estimator, RandomForestClassifier):
                spec.param_space["model__n_estimators"] = [50]
    return specs


def _clean_param_name(name: str) -> str:
    return name.replace("model__", "").replace("features__enabled", "engineered_features")


def _git_commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git not available
        return "unknown"


def _log_figure(fig, filename: str, also_save_to: Path | None = None) -> None:
    mlflow.log_figure(fig, f"plots/{filename}")
    if also_save_to is not None:
        also_save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(also_save_to, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _make_search(spec: ModelSpec, cv) -> GridSearchCV | RandomizedSearchCV:
    pipeline = build_pipeline(spec.estimator)
    common = dict(scoring=CV_SCORING, refit="roc_auc", cv=cv, n_jobs=-1, return_train_score=True, error_score="raise")
    if spec.search == "random":
        return RandomizedSearchCV(
            pipeline, spec.param_space, n_iter=spec.n_iter, random_state=config.RANDOM_STATE, **common
        )
    return GridSearchCV(pipeline, spec.param_space, **common)


def train_one(spec: ModelSpec, X_train, y_train, X_test, y_test, cv, tags: dict, persist: bool = True) -> ModelResult:
    """Tune one model family; ``persist`` also writes its plots to reports/figures."""
    search = _make_search(spec, cv)
    with mlflow.start_run(run_name=spec.name, tags={**tags, "model_family": spec.name, "stage": "tuning"}) as run:
        mlflow.set_tag("mlflow.note.content", spec.description)
        mlflow.log_params(
            {
                "model_family": spec.name,
                "search_strategy": spec.search,
                "cv_folds": config.CV_FOLDS,
                "refit_metric": "roc_auc",
                "n_candidates": spec.n_iter if spec.search == "random" else len(ParameterGrid(spec.param_space)),
            }
        )
        mlflow.log_dict({k: [str(x) for x in v] for k, v in spec.param_space.items()}, "search_space.json")

        started = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        cv_results = pd.DataFrame(search.cv_results_)
        # One nested run per evaluated candidate.
        for i, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"{spec.name}-trial-{i:02d}", nested=True, tags={**tags, "stage": "trial"}):
                mlflow.log_params({_clean_param_name(k): str(v) for k, v in row["params"].items()})
                trial_metrics = {"rank_roc_auc": float(row["rank_test_roc_auc"])}
                for metric in CV_SCORING:
                    trial_metrics[f"cv_{metric}_mean"] = float(row[f"mean_test_{metric}"])
                    trial_metrics[f"cv_{metric}_std"] = float(row[f"std_test_{metric}"])
                trial_metrics["train_roc_auc_mean"] = float(row["mean_train_roc_auc"])
                mlflow.log_metrics(trial_metrics)

        best_idx = search.best_index_
        cv_metrics = {}
        for metric in CV_SCORING:
            cv_metrics[f"cv_{metric}_mean"] = float(cv_results.loc[best_idx, f"mean_test_{metric}"])
            cv_metrics[f"cv_{metric}_std"] = float(cv_results.loc[best_idx, f"std_test_{metric}"])

        best = search.best_estimator_
        y_proba = best.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test, y_pred, y_proba).items()}

        best_params = {_clean_param_name(k): v for k, v in search.best_params_.items()}
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metrics({**cv_metrics, **test_metrics, "search_fit_seconds": fit_seconds})

        title = spec.name.replace("_", " ").title()
        fig_dir = config.FIGURES_DIR / "models" / spec.name

        def out(name: str) -> Path | None:
            return fig_dir / name if persist else None

        _log_figure(plot_roc(y_test, y_proba, f"ROC - {title}"), "roc_curve.png", out("roc_curve.png"))
        _log_figure(plot_pr(y_test, y_proba, f"Precision-Recall - {title}"), "pr_curve.png", out("pr_curve.png"))
        _log_figure(
            plot_confusion(y_test, y_pred, f"Confusion matrix - {title}"),
            "confusion_matrix.png",
            out("confusion_matrix.png"),
        )
        _log_figure(
            plot_calibration(y_test, y_proba, f"Calibration - {title}"),
            "calibration.png",
            out("calibration.png"),
        )
        importance = feature_importance(best)
        if importance is not None:
            _log_figure(
                plot_importance(importance, f"Top features - {title}"),
                "feature_importance.png",
                out("feature_importance.png"),
            )
            mlflow.log_dict(importance.round(5).to_dict(), "feature_importance.json")

        with tempfile.TemporaryDirectory() as tmp:
            cv_path = Path(tmp) / "cv_results.csv"
            cv_results.drop(columns=["params"]).to_csv(cv_path, index=False)
            mlflow.log_artifact(str(cv_path), artifact_path="tuning")
        mlflow.log_text(
            classification_report(y_test, y_pred, target_names=["no_disease", "disease"], zero_division=0),
            "classification_report.txt",
        )

        signature = infer_signature(X_train, best.predict_proba(X_train))
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature, input_example=X_train.head(3))

        logger.info(
            "%-20s cv_roc_auc=%.3f±%.3f test_roc_auc=%.3f test_acc=%.3f best=%s",
            spec.name,
            cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"],
            test_metrics["test_roc_auc"],
            test_metrics["test_accuracy"],
            best_params,
        )
        return ModelResult(spec.name, best, best_params, cv_metrics, test_metrics, run.info.run_id, y_proba)


def comparison_figures(results: list[ModelResult], y_test) -> tuple[plt.Figure, plt.Figure]:
    colors = ["#9e9e9e", "#264653", "#2a9d8f", "#e76f51", "#e9c46a"]
    roc_fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for color, res in zip(colors, results, strict=False):
        fpr, tpr, _ = roc_curve(y_test, res.y_proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{res.name} (AUC {res.test_metrics['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Hold-out ROC curves - all models")
    ax.legend(loc="lower right", fontsize=9)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    table = pd.DataFrame(
        {res.name: [res.cv_metrics[f"cv_{m}_mean"] for m in metrics] for res in results}, index=metrics
    )
    bar_fig, ax2 = plt.subplots(figsize=(9, 4.8))
    table.plot(kind="bar", ax=ax2, color=colors[: len(results)], width=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Mean 5-fold CV score")
    ax2.set_title("Cross-validated metrics by model")
    ax2.set_xticklabels(metrics, rotation=0)
    ax2.legend(fontsize=9, loc="lower right")
    return roc_fig, bar_fig


def export_champion(champion: ModelResult, X_train, results, tags: dict, register: bool) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion.pipeline, config.MODEL_FILE, compress=3)

    model_version = None
    if register:
        version = mlflow.register_model(f"runs:/{champion.run_id}/model", config.REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", version.version)
        client.update_model_version(
            config.REGISTERED_MODEL_NAME,
            version.version,
            description=f"{champion.name}; cv_roc_auc={champion.cv_metrics['cv_roc_auc_mean']:.3f}",
        )
        model_version = version.version

    if config.MLFLOW_MODEL_DIR.exists():
        import shutil

        shutil.rmtree(config.MLFLOW_MODEL_DIR)
    signature = infer_signature(X_train, champion.pipeline.predict_proba(X_train))
    mlflow.sklearn.save_model(
        champion.pipeline, str(config.MLFLOW_MODEL_DIR), signature=signature, input_example=X_train.head(3)
    )

    metadata = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_family": champion.name,
        "model_version": f"{__version__}+{champion.name}",
        "mlflow_run_id": champion.run_id,
        "mlflow_registry_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": tags.get("git_commit"),
        "dataset_sha256": tags.get("dataset_sha256"),
        "decision_threshold": 0.5,
        "features": config.FEATURE_ORDER,
        "best_params": {
            k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
            for k, v in champion.best_params.items()
        },
        "cv_metrics": champion.cv_metrics,
        "test_metrics": champion.test_metrics,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "all_models": {r.name: {**r.cv_metrics, **r.test_metrics} for r in results},
    }
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    return metadata


def run(quick: bool = False, register: bool = True) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = load_clean_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    logger.info("train=%d test=%d positive_rate=%.3f", len(X_train), len(X_test), y.mean())

    tags = {
        "git_commit": _git_commit(),
        "dataset_sha256": sha256_of(config.CLEAN_DATA_FILE),
        "quick_mode": str(quick),
        "triggered_by": "github-actions" if os.getenv("GITHUB_ACTIONS") else "local",
    }

    results = []
    for spec in model_specs(quick):
        results.append(train_one(spec, X_train, y_train, X_test, y_test, cv, tags, persist=not quick))

    candidates = [r for r in results if not r.name.startswith("baseline")]
    champion = max(candidates, key=lambda r: (r.cv_metrics["cv_roc_auc_mean"], r.cv_metrics["cv_accuracy_mean"]))
    logger.info("Champion model: %s (cv_roc_auc=%.3f)", champion.name, champion.cv_metrics["cv_roc_auc_mean"])

    rows = []
    for r in results:
        params = json.dumps(r.best_params, default=str)
        rows.append({"model": r.name, **r.cv_metrics, **r.test_metrics, "best_params": params})
    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)

    with mlflow.start_run(run_name="model-selection", tags={**tags, "stage": "selection"}):
        mlflow.log_params({"champion": champion.name, "selection_metric": "cv_roc_auc_mean"})
        mlflow.log_metrics({f"champion_{k}": v for k, v in {**champion.cv_metrics, **champion.test_metrics}.items()})
        roc_fig, bar_fig = comparison_figures(results, y_test)
        persist = not quick
        _log_figure(roc_fig, "roc_comparison.png", config.FIGURES_DIR / "09_roc_comparison.png" if persist else None)
        _log_figure(
            bar_fig, "cv_metric_comparison.png", config.FIGURES_DIR / "10_cv_metric_comparison.png" if persist else None
        )
        mlflow.log_table(comparison, "model_comparison.json")
        if quick:
            # Smoke-test mode never overwrites the exported production model or reports.
            return {"model_family": champion.name, "test_metrics": champion.test_metrics}
        metadata = export_champion(champion, X_train, results, tags, register)
        mlflow.log_dict(metadata, "champion_metadata.json")

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    (config.METRICS_DIR / "champion_metrics.json").write_text(
        json.dumps({"champion": champion.name, **champion.cv_metrics, **champion.test_metrics}, indent=2)
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart disease classifiers with MLflow tracking")
    parser.add_argument("--quick", action="store_true", help="single-candidate search for smoke testing")
    parser.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    args = parser.parse_args()
    # Initialise the tracking store before configuring logging: on a fresh sqlite store MLflow
    # runs alembic migrations, whose fileConfig() resets the root logger and disables ours.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    logger.disabled = False
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    metadata = run(quick=args.quick, register=not args.no_register)
    print(json.dumps({"champion": metadata["model_family"], **metadata["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
