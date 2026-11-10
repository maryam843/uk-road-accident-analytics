"""Tests for src.dashboard_exports (notebook result files and the CSV builders)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import dashboard_exports as dx


def test_missing_result_file_fails_with_a_clear_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"sarima_order\.json is missing; run notebooks/03_forecasting\.ipynb"):
        dx.sarima_order(tmp_path)
    with pytest.raises(FileNotFoundError, match="07_anomalies"):
        dx.result_path("anomalous_days_top10.csv", tmp_path)


def test_result_files_are_read(tmp_path: Path) -> None:
    (tmp_path / "sarima_order.json").write_text(
        json.dumps({"order": [1, 1, 1], "seasonal_order": [1, 0, 2, 12], "with_intercept": False}))
    assert dx.sarima_order(tmp_path) == ((1, 1, 1), (1, 0, 2, 12), False)
    pd.DataFrame({"year": [2005, 2007], "kaggle_rows": [10, 30]}).to_csv(tmp_path / "kaggle_rows_by_year.csv", index=False)
    yearly = pd.DataFrame({"collisions": [10, 20, 30]}, index=pd.Index([2005, 2006, 2007], name="year"))
    ras = pd.DataFrame({"collisions": [10, 20, 31]}, index=pd.Index([2005, 2006, 2007], name="year"))
    trust = dx.data_trust(yearly, ras, tmp_path)
    assert trust["kaggle_rows"].isna().tolist() == [False, True, False]
    assert trust["pipeline_minus_ras0101"].tolist() == [0, 0, -1]


def test_ksi_share_yearly_and_by_factor() -> None:
    yearly = pd.DataFrame({"collisions": [100, 200], "ksi_collisions": [10, 50],
                           "ksi_collisions_adjusted": [20.0, 60.0]}, index=pd.Index([2018, 2019], name="year"))
    share = dx.ksi_share_yearly(yearly)
    assert share["ksi_share_reported"].tolist() == [0.1, 0.25]
    assert share["ksi_share_adjusted"].tolist() == [0.2, 0.3]
    coll = pd.DataFrame({"speed_limit": pd.array([30, 30, 70, None], dtype="Int16"),
                         "urban_or_rural_area_label": ["Urban", "Urban", "Rural", None],
                         "light_conditions_label": ["Daylight"] * 4, "ksi": [1, 0, 1, 0]})
    out = dx.ksi_rate_by_factor(coll)
    for factor in dx.FACTORS:
        part = out[out["factor"] == factor]
        assert part["collisions"].sum() == 4 and part["ksi"].sum() == 2
    speed = out[out["factor"] == "speed limit (mph)"].set_index("level")
    assert speed.loc["30", "ksi_share"] == 0.5 and "Missing" in speed.index
    assert (out["wilson_low"] <= out["ksi_share"]).all() and (out["ksi_share"] <= out["wilson_high"]).all()


def test_write_csv_enforces_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"a": range(10)})
    assert dx.write_csv(frame, "small.csv", tmp_path).exists()
    monkeypatch.setattr(dx, "MAX_BYTES", 5)
    with pytest.raises(ValueError):
        dx.write_csv(frame, "big.csv", tmp_path)
