"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset.

Run as a script to download the raw file and write the cleaned CSV:

    python -m src.data            # download (if needed) + clean
    python -m src.data --force    # re-download even if a valid copy exists
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            payload = archive.read("processed.cleveland.data")
    return payload


def download_raw_data(
    destination: Path = config.RAW_DATA_FILE,
    urls: list[str] | None = None,
    force: bool = False,
    retries: int = 3,
) -> Path:
    """Download the raw Cleveland file, trying each mirror with retries.

    An existing file is reused when its checksum matches, so repeated runs
    (and CI) do not hit the UCI servers unnecessarily.
    """
    destination = Path(destination)
    if destination.exists() and not force:
        if sha256_of(destination) == config.RAW_DATA_SHA256:
            logger.info("Raw data already present and verified: %s", config.display_path(destination))
            return destination
        logger.warning("Checksum mismatch for %s, downloading again", config.display_path(destination))

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls or config.DATA_URLS:
        for attempt in range(1, retries + 1):
            try:
                logger.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
                payload = _fetch(url)
                destination.write_bytes(payload)
                checksum = sha256_of(destination)
                if checksum != config.RAW_DATA_SHA256:
                    logger.warning("Downloaded file checksum %s differs from expected", checksum)
                logger.info("Saved %d bytes to %s", len(payload), config.display_path(destination))
                return destination
            except (requests.RequestException, KeyError, zipfile.BadZipFile) as exc:
                errors.append(f"{url}: {exc}")
                time.sleep(2 * attempt)
    raise RuntimeError("Could not download the dataset from any mirror:\n" + "\n".join(errors))


def load_raw_data(path: Path = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values in the source."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a typed, validated, de-duplicated frame and a cleaning report.

    Missing values are deliberately *not* imputed here: imputation lives in
    the model pipeline so it is fitted on training folds only (no leakage)
    and is applied identically at inference time.
    """
    df = raw.copy()
    report: dict = {"rows_in": int(len(df)), "missing_in_raw": {}}

    for col in config.RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    report["missing_in_raw"] = {c: int(n) for c, n in df.isna().sum().items() if n}

    # Out-of-range numerics become NaN (e.g. cholesterol recorded as 0).
    out_of_range = {}
    for col, (low, high) in config.PLAUSIBLE_RANGES.items():
        mask = df[col].notna() & ~df[col].between(low, high)
        if mask.any():
            out_of_range[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    report["out_of_range_set_to_nan"] = out_of_range

    # Unknown category codes become NaN.
    invalid_codes = {}
    for col, levels in config.CATEGORY_LEVELS.items():
        mask = df[col].notna() & ~df[col].isin(levels)
        if mask.any():
            invalid_codes[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    report["invalid_codes_set_to_nan"] = invalid_codes

    # Target: 0 = no disease, 1-4 = disease present -> binary.
    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    int_like = config.BINARY_FEATURES + ["cp", "restecg", "slope", "ca", "thal", "age"]
    for col in int_like:
        df[col] = df[col].astype("Int64")

    report["rows_out"] = int(len(df))
    report["missing_after_cleaning"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["class_balance"] = {str(k): int(v) for k, v in df[config.TARGET].value_counts().sort_index().items()}
    return df, report


def load_clean_data(path: Path = config.CLEAN_DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.FEATURE_ORDER].astype(float)
    y = df[config.TARGET].astype(int)
    return X, y


def run(force: bool = False) -> pd.DataFrame:
    raw_path = download_raw_data(force=force)
    clean, report = clean_data(load_raw_data(raw_path))
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_DATA_FILE, index=False)
    report_path = config.PROCESSED_DATA_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Clean dataset written to %s (%d rows)", config.display_path(config.CLEAN_DATA_FILE), len(clean))
    logger.info("Cleaning report: %s", json.dumps(report))
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean the UCI Heart Disease dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    run(force=args.force)


if __name__ == "__main__":
    main()
