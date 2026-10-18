"""Central configuration: paths, schema and constants shared by every stage.

Keeping the schema in one place means the cleaning code, the feature
pipeline, the API request model and the tests can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Primary + mirror locations of the UCI "processed Cleveland" file.
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
]
# SHA-256 of the canonical processed.cleveland.data file (303 rows).
RAW_DATA_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-risk")
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# The 14 attributes used by every published experiment on this dataset.
RAW_COLUMNS = [
    "age",  # age in years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl
    "restecg",  # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise induced angina
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",  # number of major vessels coloured by fluoroscopy (0-3)
    "thal",  # 3 normal, 6 fixed defect, 7 reversible defect
    "num",  # diagnosis of heart disease (0 = absent, 1-4 = present)
]
TARGET = "target"

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ENGINEERED_FEATURES = ["hr_achieved_ratio", "rate_pressure_index"]
# Model input order as the API receives it (13 clinical attributes).
FEATURE_ORDER = [c for c in RAW_COLUMNS if c != "num"]

# Allowed values for categorical codes; anything else is treated as missing.
CATEGORY_LEVELS = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

# Physiologically plausible ranges; values outside are set to NaN and imputed.
PLAUSIBLE_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 250),
    "chol": (100, 650),
    "thalach": (60, 230),
    "oldpeak": (0.0, 7.0),
}
