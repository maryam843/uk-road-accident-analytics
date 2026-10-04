"""Tests for src.dashboard_exports (notebook-output parsing and the CSV builders)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import dashboard_exports as dx


def _notebook_with_table(tmp_path: Path, frame: pd.DataFrame, text: str = "") -> Path:
    html = frame.style.format("{:,}").to_html()
    outputs = [{"output_type": "stream", "name": "stdout", "text": [text]},
               {"output_type": "display_data", "metadata": {}, "data": {"text/html": html.splitlines(True)}}]
    nb = {"cells": [{"cell_type": "code", "id": "7", "source": [], "outputs": outputs, "metadata": {},
                     "execution_count": 1}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    return path


def test_notebook_tables_reads_styler_output(tmp_path: Path) -> None:
    frame = pd.DataFrame({"rows": [198735, 0], "dups": [3, 34144]}, index=pd.Index([2005, 2012], name="year"))
    path = _notebook_with_table(tmp_path, frame, text="hello\n")
    parsed = dx.notebook_tables(path, "7")[0]
    assert list(parsed.columns) == ["rows", "dups"]
    assert list(parsed.index) == ["2005", "2012"]
    assert dx.to_number(parsed["dups"]).tolist() == [3, 34144]
    assert dx.notebook_text(path, "7") == "hello\n"
    with pytest.raises(KeyError):
        dx.notebook_text(path, "missing")


def test_notebook_tables_multiindex_columns(tmp_path: Path) -> None:
    cols = pd.MultiIndex.from_product([["Backtest", "Test"], ["MAE", "RMSE"]])
    frame = pd.DataFrame([[1.5, 2.5, 3.5, 4.5]], index=["SARIMA"], columns=cols)
    parsed = dx.notebook_tables(_notebook_with_table(tmp_path, frame), "7")[0]
    assert list(parsed.columns) == ["MAE", "RMSE", "MAE", "RMSE"]
    assert dx.to_number(parsed.iloc[:, 2]).loc["SARIMA"] == pytest.approx(3.5)


def test_to_number_handles_commas_signs_and_percent() -> None:
    got = dx.to_number(pd.Series(["198,735", "+23.46%", "60.9%", "-3", "x"]))
    assert got.iloc[:4].tolist() == pytest.approx([198735, 0.2346, 0.609, -3])
    assert np.isnan(got.iloc[4])


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
