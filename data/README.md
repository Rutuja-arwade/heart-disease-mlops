# Data

| File | Description |
|---|---|
| `raw/processed.cleveland.data` | Original UCI file (303 rows, 14 columns, no header, `?` = missing). SHA-256 `a74b7efa…0427c8`. |
| `processed/heart_disease_clean.csv` | Cleaned dataset with header, typed columns and binary `target`. |
| `processed/cleaning_report.json` | What the cleaning step changed (missing counts, invalid values, duplicates, class balance). |

## Source

Heart Disease dataset, UCI Machine Learning Repository
(Janosi, Steinbrunn, Pfisterer, Detrano, 1988) — https://archive.ics.uci.edu/dataset/45/heart+disease.
Only the Cleveland subset is used: it is the one with complete attributes and the one used in
almost all published work on this dataset.

## Re-create from scratch

```bash
./scripts/download_data.sh          # or: python -m src.data
./scripts/download_data.sh --force  # ignore the cached copy and download again
```

The script tries the classic UCI path first and falls back to the zip archive on the new UCI site,
retrying each mirror three times. A cached file is reused only if its SHA-256 matches.

## Cleaning rules (`src/data.py`)

1. Parse `?` as missing; coerce every column to numeric.
2. Numeric values outside physiological ranges (e.g. cholesterol 0) become missing.
3. Category codes outside the documented levels become missing.
4. `num` (0–4) becomes `target` = 1 if `num > 0` (disease present), else 0.
5. Drop exact duplicate rows.

Missing values (4 in `ca`, 2 in `thal`) are **kept** in the clean CSV and imputed inside the model
pipeline (median / most-frequent). That way the imputer is fitted on training folds only, and the
API handles requests that omit `ca` or `thal` the same way.
