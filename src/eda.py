"""Exploratory data analysis: summary tables and publication-quality figures.

python -m src.eda      # writes PNGs to reports/figures and tables to reports/metrics
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean_data  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}
PRETTY = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels (ca)",
    "cp": "Chest pain type",
    "restecg": "Resting ECG",
    "slope": "ST slope",
    "thal": "Thalassemia",
    "sex": "Sex (1 = male)",
    "fbs": "Fasting sugar >120",
    "exang": "Exercise angina",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold"})


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return str(path)


def plot_class_balance(df: pd.DataFrame) -> str:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for bar, value in zip(bars, counts.values, strict=False):
        ax.annotate(
            f"{value}  ({value / counts.sum():.1%})",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of the target")
    ax.set_ylim(0, counts.max() * 1.18)
    return _save(fig, "01_class_balance.png")


def plot_missing(raw_missing: dict[str, int], n_rows: int) -> str:
    cols = config.FEATURE_ORDER
    values = [raw_missing.get(c, 0) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(cols, values, color=["#e9c46a" if v else "#d9d9d9" for v in values])
    for i, v in enumerate(values):
        if v:
            ax.annotate(f"{v} ({v / n_rows:.1%})", (i, v), ha="center", va="bottom")
    ax.set_title("Missing values per feature in the raw file ('?' markers)")
    ax.set_ylabel("Missing rows")
    ax.set_ylim(0, max(values + [1]) * 1.4)
    return _save(fig, "02_missing_values.png")


def plot_histograms(df: pd.DataFrame) -> str:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        for label in (0, 1):
            subset = df.loc[df[config.TARGET] == label, col].dropna()
            discrete = col == "ca"
            sns.histplot(
                subset,
                ax=ax,
                color=PALETTE[label],
                label=LABELS[label],
                kde=not discrete,
                stat="density",
                alpha=0.45,
                bins=4 if discrete else 18,
                discrete=discrete,
            )
        ax.set_title(PRETTY[col])
        ax.set_xlabel("")
    axes[0, 0].legend()
    fig.suptitle("Distribution of continuous features by diagnosis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_feature_histograms.png")


def plot_correlation(df: pd.DataFrame) -> str:
    corr = df.astype(float).corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation heatmap (features and target)")
    return _save(fig, "04_correlation_heatmap.png")


def plot_target_correlation(df: pd.DataFrame) -> str:
    corr = df.astype(float).corr()[config.TARGET].drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=["#2a9d8f" if v < 0 else "#e76f51" for v in corr.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Correlation of each feature with the target")
    ax.set_xlabel("Pearson r with disease presence")
    return _save(fig, "05_target_correlation.png")


def plot_categorical_rates(df: pd.DataFrame) -> str:
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        table = pd.crosstab(df[col], df[config.TARGET], normalize="index")
        counts = df[col].value_counts().sort_index()
        table.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[0], PALETTE[1]], legend=False, width=0.75)
        ax.set_title(PRETTY[col])
        ax.set_xlabel("")
        ax.set_ylabel("Share of patients")
        ax.set_xticklabels([f"{int(float(i))}\n(n={counts[i]})" for i in table.index], rotation=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in (0, 1)]
    fig.legend(handles, [LABELS[0], LABELS[1]], loc="upper right")
    fig.suptitle("Disease rate within each category", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_categorical_disease_rate.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    long = df.melt(id_vars=config.TARGET, value_vars=cols)
    long["Diagnosis"] = long[config.TARGET].map(LABELS)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    for ax, col in zip(axes, cols, strict=False):
        sns.boxplot(
            data=long[long["variable"] == col],
            x="Diagnosis",
            y="value",
            hue="Diagnosis",
            palette={LABELS[0]: PALETTE[0], LABELS[1]: PALETTE[1]},
            ax=ax,
            legend=False,
        )
        ax.set_title(PRETTY[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Outlier view: continuous features by diagnosis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_boxplots.png")


def plot_age_vs_hr(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label in (0, 1):
        subset = df[df[config.TARGET] == label]
        ax.scatter(
            subset["age"], subset["thalach"], c=PALETTE[label], label=LABELS[label], alpha=0.7, edgecolor="white"
        )
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, "k--", lw=1, label="Age-predicted max HR (220 - age)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Max heart rate achieved (bpm)")
    ax.set_title("Heart-rate response vs age (motivates hr_achieved_ratio)")
    ax.legend()
    return _save(fig, "08_age_vs_max_hr.png")


def summary_tables(df: pd.DataFrame) -> dict:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    describe = df.describe().T.round(2)
    describe.to_csv(config.METRICS_DIR / "eda_summary_statistics.csv")
    by_class = df.groupby(config.TARGET).mean(numeric_only=True).T.round(2)
    by_class.columns = [LABELS[c] for c in by_class.columns]
    by_class.to_csv(config.METRICS_DIR / "eda_means_by_class.csv")
    corr = df.astype(float).corr()[config.TARGET].drop(config.TARGET).sort_values(key=np.abs, ascending=False)
    return {
        "rows": int(len(df)),
        "features": len(config.FEATURE_ORDER),
        "class_counts": {LABELS[int(k)]: int(v) for k, v in df[config.TARGET].value_counts().sort_index().items()},
        "top_target_correlations": corr.head(6).round(3).to_dict(),
    }


def run() -> dict:
    _style()
    df = load_clean_data()
    report_path = config.PROCESSED_DATA_DIR / "cleaning_report.json"
    cleaning = json.loads(report_path.read_text()) if report_path.exists() else {}
    figures = [
        plot_class_balance(df),
        plot_missing(cleaning.get("missing_in_raw", {}), len(df)),
        plot_histograms(df),
        plot_correlation(df),
        plot_target_correlation(df),
        plot_categorical_rates(df),
        plot_boxplots(df),
        plot_age_vs_hr(df),
    ]
    summary = summary_tables(df)
    summary["figures"] = [str(p) for p in figures]
    (config.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA summary: %s", json.dumps({k: v for k, v in summary.items() if k != "figures"}))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    run()
