"""Tests for feature construction, year splits and the preprocessing contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import clean, features

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    code_list = clean.load_code_list(FIXTURES / "data_guide_codes.csv")
    coll, _ = clean.process_collisions(FIXTURES / "collisions_fixture.csv", code_list)
    return features.build_model_frame(coll)


def test_model_frame_has_only_intended_columns(frame: pd.DataFrame) -> None:
    expected = {"collision_index", "year", "ksi", *features.CATEGORICAL_FEATURES,
                *features.ABLATION_NUMERIC_FEATURES}
    assert set(frame.columns) == expected
    leaky = {"did_police_officer_attend_scene_of_collision", "number_of_casualties",
             "casualty_severity", "collision_adjusted_serious", "collision_injury_based"}
    assert not leaky & set(frame.columns)


def test_region_mapping_merges_scotland_codes() -> None:
    regions = features.region_from_police_force(pd.Series([92, 99, 1, 63, 999]))
    assert regions.tolist()[:4] == ["Scotland", "Scotland", "London", "Wales"]
    assert pd.isna(regions.iloc[4])


def test_split_by_year_is_disjoint(frame: pd.DataFrame) -> None:
    splits = features.split_by_year(frame)
    assert splits["train"]["year"].between(2005, 2015).all()
    assert (splits["valid"]["year"] == 2016).all()
    assert splits["test"]["year"].between(2017, 2019).all()
    assert sum(len(s) for s in splits.values()) == len(frame)


def test_assert_disjoint_catches_overlap() -> None:
    a = pd.DataFrame({"collision_index": ["x", "y"]})
    b = pd.DataFrame({"collision_index": ["y", "z"]})
    with pytest.raises(AssertionError):
        features.assert_disjoint({"train": a, "test": b})


def test_preprocessor_covers_every_feature(frame: pd.DataFrame) -> None:
    cat = list(features.CATEGORICAL_FEATURES)
    num = features.ABLATION_NUMERIC_FEATURES
    pre = features.make_preprocessor(cat, num, min_frequency=1)
    X = pre.fit_transform(frame[cat + num])
    names = features.check_feature_coverage(pre, cat, num)
    assert X.shape[1] == len(names)
    pre_cat = features.make_preprocessor(cat[:-1], min_frequency=1).fit(frame[cat])
    with pytest.raises(AssertionError):
        features.check_feature_coverage(pre_cat, cat)
