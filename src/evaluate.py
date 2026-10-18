"""Metrics and diagnostic plots shared by training, notebooks and tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import transformed_feature_names  # noqa: E402

# Scorers used inside cross-validation. zero_division=0 keeps the majority
# baseline (which never predicts the positive class) from raising warnings.
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def plot_roc(y_true, y_proba, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#e76f51", lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig


def plot_pr(y_true, y_proba, title: str):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#264653", lw=2, label=f"Average precision = {ap:.3f}")
    ax.axhline(np.mean(y_true), color="grey", ls="--", lw=1, label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return fig


def plot_confusion(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    ax.grid(False)
    return fig


def plot_calibration(y_true, y_proba, title: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, "o-", color="#2a9d8f", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disease rate")
    ax.set_title(title)
    ax.legend()
    return fig


def feature_importance(pipeline) -> pd.Series | None:
    """Impurity importances for trees, |coefficients| for linear models."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_.ravel()
    else:
        return None
    return pd.Series(values, index=names).sort_values(key=np.abs, ascending=False)


def plot_importance(importance: pd.Series, title: str, top: int = 15):
    data = importance.head(top)[::-1]
    colors = ["#e76f51" if v > 0 else "#2a9d8f" for v in data.values]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(data.index, data.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Importance / coefficient")
    return fig
