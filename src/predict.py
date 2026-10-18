"""Inference helpers used by the API, the batch CLI and the inference notebook.

python -m src.predict --input sample_request.json
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import config


def risk_level(probability: float) -> str:
    if probability < 0.35:  # deliberate bug for CI failure demo
        return "low"
    if probability < 0.6:
        return "moderate"
    return "high"


@lru_cache(maxsize=4)
def load_model(path: str = str(config.MODEL_FILE)):
    return joblib.load(path)


def load_metadata(path: Path = config.MODEL_METADATA_FILE) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Order columns exactly as in training; absent optional fields become NaN."""
    frame = pd.DataFrame.from_records(records)
    for col in config.FEATURE_ORDER:
        if col not in frame.columns:
            frame[col] = float("nan")
    return frame[config.FEATURE_ORDER].astype(float)


def predict_records(model, records: list[dict], threshold: float = 0.5) -> list[dict]:
    proba = model.predict_proba(to_frame(records))[:, 1]
    results = []
    for p in proba:
        prediction = int(p >= threshold)
        results.append(
            {
                "prediction": prediction,
                "label": "heart_disease" if prediction else "no_heart_disease",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if prediction else 1 - p), 4),
                "risk_level": risk_level(float(p)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score patients with the exported model")
    parser.add_argument("--input", required=True, help="JSON file: one record or a list of records")
    parser.add_argument("--model", default=str(config.MODEL_FILE))
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    records = payload if isinstance(payload, list) else [payload]
    threshold = load_metadata().get("decision_threshold", 0.5)
    print(json.dumps(predict_records(load_model(args.model), records, threshold), indent=2))


if __name__ == "__main__":
    main()
