"""Tests for src.traffic on a small synthetic AADF fixture (no download needed)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import traffic


def synthetic_aadf() -> pd.DataFrame:
    """Five count points over two years, with the spellings found in the real file."""
    base = [
        # id, road, category, type, easting, northing, length, aadf
        (1, "A1", "TA", "Major", 500_000, 200_000, 2.0, 10_000),
        (2, "A1", "TA", "Major", 500_000, 210_000, 1.0, 20_000),
        (3, "M6(T)", "TM", "Major", 400_000, 300_000, 5.0, 50_000),
        (4, "A57M", "PM", "Major", 380_000, 390_000, 0.5, 30_000),
        (5, "B1234", "MB", "Minor", 450_000, 250_000, np.nan, 2_000),
    ]
    rows = [(*r, year) for year in (2018, 2019) for r in base]
    df = pd.DataFrame(rows, columns=["count_point_id", "road_name", "road_category", "road_type",
                                     "easting", "northing", "link_length_km", "all_motor_vehicles",
                                     "year"])
    for col in set(traffic.AADF_COLUMNS) - set(df.columns):
        df[col] = "x" if col not in {"latitude", "longitude"} else 52.0
    df["extra_column_not_used"] = 1
    return df


@pytest.fixture()
def aadf_zip(tmp_path: Path) -> Path:
    path = tmp_path / "dft_traffic_counts_aadf.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("dft_traffic_counts_aadf.csv", synthetic_aadf().to_csv(index=False))
    return path


def test_load_filters_years_and_columns(aadf_zip: Path) -> None:
    df = traffic.load_aadf(aadf_zip, years=(2019, 2019))
    assert set(df.columns) == set(traffic.AADF_COLUMNS)
    assert (df["year"] == 2019).all() and len(df) == 5


def test_load_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    synthetic_aadf().drop(columns=["link_length_km"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="link_length_km"):
        traffic.load_aadf(path)


def test_normalise_road_name() -> None:
    names = pd.Series(["A57M", "a57(m)", "M6(T)", " M25 ", "A1"])
    assert traffic.normalise_road_name(names).tolist() == ["A57(M)", "A57(M)", "M6", "M25", "A1"]


def test_collision_road_name() -> None:
    cls = pd.Series([1, 2, 3, 3, 4, 3])
    num = pd.Series([25, 1, 1, 0, 123, None])
    got = traffic.collision_road_name(cls, num)
    assert got.iloc[:3].tolist() == ["M25", "A1(M)", "A1"]
    assert got.iloc[3:].isna().all()  # no number, B road, missing number


def test_clean_aadf_vehicle_km(aadf_zip: Path) -> None:
    df = traffic.clean_aadf(traffic.load_aadf(aadf_zip))
    major = df[df["is_major"]]
    assert major["usable"].all()
    first = df[(df["count_point_id"] == 1) & (df["year"] == 2018)].iloc[0]
    assert first["vehicle_km"] == pytest.approx(10_000 * 2.0 * 365)
    minor = df[~df["is_major"]]
    assert (~minor["usable"]).all() and minor["vehicle_km"].isna().all()
    assert set(df.loc[df["road_category"] == "TM", "road_group"]) == {"Motorway"}


def test_match_same_road_same_year_with_cap() -> None:
    points = traffic.clean_aadf(synthetic_aadf())
    points = points[points["usable"]].assign(road_name=lambda d: d["road_name_norm"])
    collisions = pd.DataFrame({
        "road_name": ["A1", "A1", "A1", "M6", "A2", None, "A1"],
        "easting": [500_100, 500_000, 500_000, 400_000, 500_000, 500_000, 500_000],
        "northing": [200_000, 209_000, 205_000, 301_000, 200_000, 200_000, 200_500],
        "year": [2019, 2019, 2019, 2018, 2019, 2019, 2017],
    }, index=[10, 11, 12, 13, 14, 15, 16])
    got = traffic.match_to_count_points(collisions, points, cap_m=2_000)
    assert got.loc[10, "count_point_id"] == 1 and got.loc[10, "distance_m"] == pytest.approx(100)
    assert got.loc[11, "count_point_id"] == 2  # nearest A1 point, not the first one
    assert got.loc[12, "reason"] == "beyond distance cap" and pd.isna(got.loc[12, "count_point_id"])
    assert got.loc[12, "distance_m"] == pytest.approx(5_000)
    assert got.loc[13, "count_point_id"] == 3  # M6 toll point matches motorway 6
    assert got.loc[14, "reason"] == "no same-road count point"
    assert got.loc[15, "reason"] == "missing road number"
    assert got.loc[16, "reason"] == "no same-road count point"  # no 2017 counts in the fixture
    wider = traffic.match_to_count_points(collisions, points, cap_m=5_000)
    assert wider.loc[12, "reason"] == "matched"
