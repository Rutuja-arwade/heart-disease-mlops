"""Data acquisition and cleaning tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data


def test_raw_loader_marks_question_marks_as_missing(raw_frame):
    assert list(raw_frame.columns) == config.RAW_COLUMNS
    assert raw_frame["ca"].isna().sum() == 1
    assert raw_frame["thal"].isna().sum() == 2


def test_clean_binarises_target(raw_frame):
    clean, _ = data.clean_data(raw_frame)
    assert "num" not in clean.columns
    assert set(clean[config.TARGET].unique()) <= {0, 1}
    # num values 2 and 4 collapse to "disease present"
    assert clean[config.TARGET].tolist() == [0, 1, 1, 0, 0, 1]


def test_clean_removes_duplicates_and_reports(raw_frame):
    clean, report = data.clean_data(raw_frame)
    assert report["rows_in"] == 7
    assert report["duplicates_removed"] == 1
    assert report["rows_out"] == len(clean) == 6
    assert report["missing_in_raw"] == {"ca": 1, "thal": 2}


def test_clean_keeps_missing_for_pipeline_imputation(raw_frame):
    clean, report = data.clean_data(raw_frame)
    # Imputation is a pipeline concern (fit on train folds only), not a cleaning one.
    assert clean["ca"].isna().sum() == 1
    assert report["missing_after_cleaning"]["thal"] == 1


def test_out_of_range_and_invalid_codes_become_nan(raw_frame):
    raw_frame.loc[0, "chol"] = 0  # impossible cholesterol
    raw_frame.loc[1, "cp"] = 9  # unknown chest pain code
    clean, report = data.clean_data(raw_frame)
    assert np.isnan(clean.loc[0, "chol"])
    assert pd.isna(clean.loc[1, "cp"])
    assert report["out_of_range_set_to_nan"] == {"chol": 1}
    assert report["invalid_codes_set_to_nan"] == {"cp": 1}


def test_split_features_target_order(clean_df):
    X, y = data.split_features_target(clean_df)
    assert list(X.columns) == config.FEATURE_ORDER
    assert len(X) == len(y) == len(clean_df)
    assert y.dtype.kind == "i"


def test_real_dataset_shape_and_balance(clean_df):
    assert clean_df.shape == (303, 14)
    positive_rate = clean_df[config.TARGET].mean()
    assert 0.4 < positive_rate < 0.5


def test_download_reuses_verified_file(tmp_path, monkeypatch):
    target = tmp_path / "raw.data"
    target.write_bytes(config.RAW_DATA_FILE.read_bytes())

    def boom(*_args, **_kwargs):
        raise AssertionError("network must not be used when a verified copy exists")

    monkeypatch.setattr(data, "_fetch", boom)
    assert data.download_raw_data(target) == target


def test_download_falls_back_to_next_mirror(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, timeout=30):
        calls.append(url)
        if "bad" in url:
            raise data.requests.ConnectionError("mirror down")
        return b"1,2,3\n"

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    monkeypatch.setattr(data.time, "sleep", lambda _s: None)
    out = data.download_raw_data(tmp_path / "x.data", urls=["http://bad", "http://good"], force=True, retries=2)
    assert out.read_bytes() == b"1,2,3\n"
    assert calls == ["http://bad", "http://bad", "http://good"]


def test_download_raises_when_all_mirrors_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_fetch", lambda *_a, **_k: (_ for _ in ()).throw(data.requests.Timeout("t")))
    monkeypatch.setattr(data.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="Could not download"):
        data.download_raw_data(tmp_path / "x.data", urls=["http://a"], force=True, retries=1)
