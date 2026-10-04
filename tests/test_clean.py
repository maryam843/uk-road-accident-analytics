"""Unit tests for src.clean, run against the synthetic fixture in tests/fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import clean, config
from tests.fixtures.make_fixture import ras0101_sheets, write_ods

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def code_list() -> pd.DataFrame:
    return clean.load_code_list(FIXTURES / "data_guide_codes.csv")


@pytest.fixture(scope="module")
def raw_collisions() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "collisions_fixture.csv", dtype=str)


@pytest.fixture(scope="module")
def collisions(code_list: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Small chunks so the chunked path (several chunks) is exercised.
    return clean.process_collisions(FIXTURES / "collisions_fixture.csv", code_list, chunksize=500)


@pytest.fixture(scope="module")
def casualties(code_list: pd.DataFrame, collisions: tuple[pd.DataFrame, dict]) -> tuple[pd.DataFrame, dict]:
    coll, _ = collisions
    return clean.process_casualties(
        FIXTURES / "casualties_fixture.csv", code_list, coll["collision_index"], chunksize=700
    )


# --- column normalisation ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Accident_Index", "collision_index"),
        ("﻿accident_index", "collision_index"),
        ("accident_year", "collision_year"),
        ("accident_reference", "collision_ref_no"),
        ("Accident_Severity", "collision_severity"),
        ("did_police_officer_attend_scene_of_accident", "did_police_officer_attend_scene_of_collision"),
        ("Local_Authority_(District)", "local_authority_district"),
        ("Light_Conditions", "light_conditions"),
        ("speed_limit", "speed_limit"),
        ("lsoa_of_accident_location", "lsoa_of_collision_location"),
        # spelling used in the real 2026 release header
        ("collision_adjusted_severity_serious", "collision_adjusted_serious"),
        ("collision_adjusted_severity_slight", "collision_adjusted_slight"),
        ("casualty_adjusted_severity_serious", "casualty_adjusted_serious"),
        ("casualty_adjusted_severity_slight", "casualty_adjusted_slight"),
    ],
)
def test_normalise_column_name(raw: str, expected: str) -> None:
    assert clean.normalise_column_name(raw) == expected


def test_build_column_map_prefers_collision_spelling_and_coalesces() -> None:
    header = ["did_police_officer_attend_scene_of_accident", "did_police_officer_attend_scene_of_collision"]
    mapping = clean.build_column_map(header)
    assert mapping["did_police_officer_attend_scene_of_collision"] == "did_police_officer_attend_scene_of_collision"
    assert mapping["did_police_officer_attend_scene_of_accident"].endswith("__dup1")
    df = pd.DataFrame({"a": [1.0, None, None], "b": [9.0, 2.0, None]}).rename(
        columns={"a": "x", "b": "x__dup1"}
    )
    out = clean.coalesce_duplicates(df)
    assert list(out.columns) == ["x"]
    assert out["x"].tolist()[:2] == [1.0, 2.0]
    assert pd.isna(out["x"].iloc[2])


def test_fixture_header_is_normalised_and_legacy_values_coalesced(
    collisions: tuple[pd.DataFrame, dict], raw_collisions: pd.DataFrame
) -> None:
    coll, _ = collisions
    assert not any("accident" in c for c in coll.columns)
    assert not any("__dup" in c for c in coll.columns)
    legacy = raw_collisions["did_police_officer_attend_scene_of_accident"].notna()
    assert legacy.any()
    ids = raw_collisions.loc[legacy, "collision_index"]
    kept = coll[coll["collision_index"].isin(ids)]
    # Values that only existed in the legacy column survive (unless they were -1).
    assert kept["did_police_officer_attend_scene_of_collision"].notna().any()


# --- code list and missing codes ---------------------------------------------------------
def test_code_list_joins_duplicate_codes(code_list: pd.DataFrame) -> None:
    rows = code_list[(code_list["field"] == "casualty_type") & (code_list["code"] == 23)]
    assert len(rows) == 1
    assert "/" in rows["label"].iloc[0]


def test_missing_codes_become_nan_not_category(code_list: pd.DataFrame) -> None:
    values = pd.Series(["1", "-1", "9", "2", None])  # 9 = "Unknown" weather: kept as a category
    codes, labels, stats = clean.decode_field(values, code_list, "collision", "weather_conditions")
    assert codes.isna().tolist() == [False, True, False, False, True]
    assert stats["missing_to_nan"] == 1
    assert "Data missing or out of range" not in set(labels.dropna())
    assert labels.iloc[2] == "Unknown"


def test_self_reported_unknown_is_missing(code_list: pd.DataFrame) -> None:
    values = pd.Series([4, 9, -1])  # junction_control 9 = "unknown (self reported)"
    codes, _, stats = clean.decode_field(values, code_list, "collision", "junction_control")
    assert codes.tolist()[0] == 4
    assert codes.isna().tolist() == [False, True, True]
    assert stats["missing_to_nan"] == 2


def test_unmapped_code_is_counted(code_list: pd.DataFrame) -> None:
    _, labels, stats = clean.decode_field(pd.Series([1, 77]), code_list, "collision", "road_type")
    assert stats["unmapped_codes"] == 1
    assert pd.isna(labels.iloc[1])


def test_no_missing_codes_left_after_processing(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, stats = collisions
    for field in clean.COLLISION_CODED_FIELDS:
        assert not (coll[field] == -1).any(), field
        assert "Data missing or out of range" not in set(coll[f"{field}_label"].dropna())
    assert sum(s["missing_to_nan"] for s in stats["decode"].values()) > 0


# --- year filter, dedup, speed limits, coordinates -----------------------------------------
def test_year_filter(collisions: tuple[pd.DataFrame, dict], raw_collisions: pd.DataFrame) -> None:
    coll, _ = collisions
    raw_years = raw_collisions["collision_year"].astype(int)
    assert raw_years.min() < config.YEAR_MIN and raw_years.max() > config.YEAR_MAX
    assert coll["collision_year"].between(config.YEAR_MIN, config.YEAR_MAX).all()
    assert coll["date"].dt.year.eq(coll["collision_year"]).all()


def test_filter_years_bounds() -> None:
    df = pd.DataFrame({"collision_year": [2004, 2005, 2019, 2020]})
    assert clean.filter_years(df)["collision_year"].tolist() == [2005, 2019]


def test_dedup_counts_exact_and_conflicting() -> None:
    df = pd.DataFrame(
        {"collision_index": ["a", "a", "b", "b", "c"], "sev": [1, 1, 2, 3, 3]}
    )
    out, stats = clean.dedupe_collisions(df)
    assert out["collision_index"].tolist() == ["a", "b", "c"]
    assert stats == {
        "duplicate_index_rows_dropped": 2,
        "exact_duplicate_rows": 1,
        "conflicting_duplicate_rows": 1,
    }


def test_fixture_dedup(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, stats = collisions
    assert coll["collision_index"].is_unique
    assert stats["dedup"]["exact_duplicate_rows"] > 0
    assert stats["dedup"]["conflicting_duplicate_rows"] > 0


def test_speed_limit_validation() -> None:
    values = pd.Series(["20", "30", "15", "99", "-1", None, "70", "10"])
    cleaned, n_invalid = clean.validate_speed_limit(values)
    assert cleaned.tolist()[:2] == [20, 30]
    assert cleaned.isna().tolist() == [False, False, True, True, True, True, False, True]
    assert n_invalid == 4  # 15, 99, -1, 10 (the blank was already missing)


def test_fixture_speed_limits_valid(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, stats = collisions
    assert set(coll["speed_limit"].dropna().unique()) <= set(config.VALID_SPEED_LIMITS)
    assert stats["invalid_speed_limit"] > 0


def test_coordinates_flagged_not_imputed(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, _ = collisions
    bad = ~coll["coord_valid"]
    assert bad.any()
    # Invalid coordinates are left as they were (missing or 0), never filled in.
    assert (coll.loc[bad, "latitude"].isna() | (coll.loc[bad, "latitude"] == 0)).all()
    assert coll.loc[coll["coord_valid"], "latitude"].between(*config.GB_LAT_RANGE).all()


def test_datetime_and_ksi(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, stats = collisions
    assert stats["weekday_code_vs_date_mismatch"] == 0
    has_time = coll["hour"].notna()
    assert (coll.loc[has_time, "datetime"].dt.hour == coll.loc[has_time, "hour"]).all()
    assert stats["datetime"]["missing_or_bad_times"] == int((~has_time).sum()) > 0
    assert (coll["ksi"] == coll["collision_severity"].isin([1, 2]).astype(int)).all()


# --- casualties and aggregates -------------------------------------------------------------
def test_casualties_only_for_kept_collisions(
    collisions: tuple[pd.DataFrame, dict], casualties: tuple[pd.DataFrame, dict]
) -> None:
    coll, _ = collisions
    cas, stats = casualties
    assert cas["collision_index"].isin(coll["collision_index"]).all()
    assert not cas["collision_index"].str.contains("ORPHAN").any()
    assert not cas.duplicated(subset=clean.CASUALTY_KEY).any()
    assert stats["duplicate_casualty_rows_dropped"] > 0


def test_monthly_series_has_no_gaps(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, _ = collisions
    monthly = clean.monthly_series(coll)
    assert len(monthly) == 12 * (config.YEAR_MAX - config.YEAR_MIN + 1)
    assert pd.infer_freq(monthly.index) == "MS"
    assert monthly["collisions"].sum() == len(coll)
    parts = monthly[["fatal_collisions", "serious_collisions", "slight_collisions"]].sum(axis=1)
    assert (parts == monthly["collisions"]).all()


def test_yearly_aggregates(
    collisions: tuple[pd.DataFrame, dict], casualties: tuple[pd.DataFrame, dict]
) -> None:
    coll, _ = collisions
    cas, _ = casualties
    yearly = clean.yearly_aggregates(coll, cas)
    assert yearly.index.min() >= config.YEAR_MIN and yearly.index.max() <= config.YEAR_MAX
    assert yearly["collisions"].sum() == len(coll)
    assert yearly["casualties"].sum() == len(cas)
    assert (yearly["ksi_collisions_adjusted"] >= yearly["fatal_collisions"]).all()


def test_ras0101_reader_roundtrip(tmp_path: Path) -> None:
    yearly = pd.DataFrame(
        {
            "collisions": [145571, 138660],
            "fatal_collisions": [1637, 1608],
            "ksi_collisions": [20000, 19000],
            "ksi_collisions_adjusted": [25000.5, 24000.25],
            "killed": [1754, 1713],
            "casualties": [195723, 183670],
        },
        index=pd.Index([2012, 2013], name="year"),
    )
    path = tmp_path / "ras0101.ods"
    write_ods(path, ras0101_sheets(yearly))
    out = clean.load_ras0101(path)
    assert out.loc[2012, "collisions"] == 145571
    assert out.loc[2013, "ksi_collisions_adjusted"] == pytest.approx(24000.25)
    assert out.loc[2013, "killed"] == 1713
    assert pd.isna(out.loc[1926, "fatal_collisions"])  # "[x]" -> NaN


def test_dashboard_data(
    collisions: tuple[pd.DataFrame, dict], casualties: tuple[pd.DataFrame, dict], tmp_path: Path
) -> None:
    coll, _ = collisions
    cas, _ = casualties
    sizes = clean.write_dashboard_data(coll, cas, tmp_path)
    expected = {"monthly_collisions_by_severity.csv", "yearly_totals.csv", "by_speed_limit.csv",
                "by_light_conditions.csv", "by_road_type.csv", "by_region.csv",
                "fatal_collision_points.csv"}
    assert set(sizes) == expected
    assert all(size < config.DASHBOARD_MAX_BYTES for size in sizes.values())
    yearly = pd.read_csv(tmp_path / "yearly_totals.csv")
    parts = yearly[["fatal_collisions", "serious_collisions", "slight_collisions"]].sum(axis=1)
    assert (parts == yearly["collisions"]).all()  # the DAX reconciliation check must be 0
    for stem in ("by_speed_limit", "by_light_conditions", "by_road_type", "by_region"):
        table = pd.read_csv(tmp_path / f"{stem}.csv")
        assert table["collisions"].sum() == len(coll), stem
        assert pd.to_datetime(table["year_start"]).dt.month.eq(1).all()
    points = pd.read_csv(tmp_path / "fatal_collision_points.csv")
    assert points["latitude"].between(*config.GB_LAT_RANGE).all()


def test_real_release_header_spelling_is_read(tmp_path: Path, code_list: pd.DataFrame) -> None:
    """The 2026 release writes *_adjusted_severity_*; the values must still be loaded."""
    raw = pd.read_csv(FIXTURES / "collisions_fixture.csv", dtype=str).rename(
        columns={
            "collision_adjusted_serious": "collision_adjusted_severity_serious",
            "collision_adjusted_slight": "collision_adjusted_severity_slight",
        }
    )
    path = tmp_path / "collisions_2026_header.csv"
    raw.to_csv(path, index=False)
    coll, _ = clean.process_collisions(path, code_list)
    assert coll["collision_adjusted_serious"].notna().all()
    assert coll["collision_adjusted_slight"].notna().all()


def test_road_number_and_grid_reference_kept(collisions: tuple[pd.DataFrame, dict]) -> None:
    coll, _ = collisions
    for col in ("first_road_number", "location_easting_osgr", "location_northing_osgr"):
        assert str(coll[col].dtype) == "Int32"
        assert (coll[col].dropna() >= 0).all()  # negative missing codes became NA
