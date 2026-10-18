#!/usr/bin/env bash
# Download the UCI Heart Disease (Cleveland) data and produce the cleaned CSV.
# Usage: ./scripts/download_data.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data "$@"
echo "Raw file:   data/raw/processed.cleveland.data"
echo "Clean file: data/processed/heart_disease_clean.csv"
