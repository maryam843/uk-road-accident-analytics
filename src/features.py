"""Feature construction and year-based splits for the KSI severity model.

Only information available *before* the collision is used as a feature. Fields that
are recorded after the event, or that are derived from the outcome, are listed in
:data:`EXCLUDED_FROM_MODEL` together with the reason.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

# Police force code -> region. Police force areas do not match ONS regions exactly
# (e.g. Thames Valley, Humberside), so this is an approximate, police-force-based region.
POLICE_FORCE_REGION: dict[int, str] = {
    1: "London", 48: "London",
    3: "North West", 4: "North West", 5: "North West", 6: "North West", 7: "North West",
    10: "North East", 11: "North East", 17: "North East",
    12: "Yorkshire and the Humber", 13: "Yorkshire and the Humber",
    14: "Yorkshire and the Humber", 16: "Yorkshire and the Humber",
    20: "West Midlands", 21: "West Midlands", 22: "West Midlands", 23: "West Midlands",
    30: "East Midlands", 31: "East Midlands", 32: "East Midlands", 33: "East Midlands",
    34: "East Midlands",
    35: "East of England", 36: "East of England", 37: "East of England",
    40: "East of England", 41: "East of England", 42: "East of England",
    43: "South East", 44: "South East", 45: "South East", 46: "South East", 47: "South East",
    50: "South West", 52: "South West", 53: "South West", 54: "South West", 55: "South West",
    60: "Wales", 61: "Wales", 62: "Wales", 63: "Wales",
    91: "Scotland", 92: "Scotland", 93: "Scotland", 94: "Scotland", 95: "Scotland",
    96: "Scotland", 97: "Scotland", 98: "Scotland", 99: "Scotland",
}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Pre-event categorical features (model input name -> source column in processed data).
CATEGORICAL_FEATURES: dict[str, str] = {
    "hour": "hour",
    "weekday": "date",
    "month": "date",
    "speed_limit": "speed_limit",
    "road_type": "road_type_label",
    "first_road_class": "first_road_class_label",
    "junction_detail": "junction_detail_historic_label",
    "light_conditions": "light_conditions_label",
    "weather_conditions": "weather_conditions_label",
    "road_surface_conditions": "road_surface_conditions_label",
    "urban_or_rural_area": "urban_or_rural_area_label",
    "region": "police_force",
}
# Used only in the ablation run: the vehicle count is fixed at the moment of impact,
# so it is not a pre-trip feature, but it is not an outcome either.
ABLATION_NUMERIC_FEATURES: list[str] = ["number_of_vehicles"]

EXCLUDED_FROM_MODEL: dict[str, str] = {
    "did_police_officer_attend_scene_of_collision": "recorded after the event; police are "
    "more likely to attend serious collisions, so it leaks the label",
    "number_of_casualties": "outcome of the collision; more casualties means more chances "
    "that one is killed or seriously injured",
    "casualty table (all fields)": "describes the injured people, i.e. the outcome itself",
    "collision_adjusted_serious / collision_adjusted_slight": "derived from the severity label",
    "collision_injury_based": "describes how severity was recorded (CRASH/COPA), not the "
    "collision; it would let the model learn the reporting system instead of risk",
    "enhanced_collision_severity": "outcome, and only collected from 2023",
    "police_force (raw code)": "replaced by region: Scottish codes changed in 2019 (test years)",
    "special_conditions_at_site, carriageway_hazards": "not in the agreed feature list",
    "latitude / longitude": "not in the agreed feature list; spatial work is notebook 05",
}

TARGET = "ksi"


def region_from_police_force(codes: pd.Series) -> pd.Series:
    """Map police force codes to an approximate region (NaN if unknown)."""
    return codes.astype("Int64").map(POLICE_FORCE_REGION).astype("object")


def build_model_frame(collisions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per collision with model features, ``ksi``, ``year`` and the index.

    All categorical features are returned as ``object`` strings with ``np.nan`` for
    missing values (the encoder later adds an explicit missing-indicator column).
    """
    date = collisions["date"]
    out = pd.DataFrame(index=collisions.index)
    out["collision_index"] = collisions["collision_index"].astype(str)
    out["year"] = collisions["collision_year"].astype(int)
    out["hour"] = collisions["hour"].astype("Int64").map(lambda h: f"{int(h):02d}", na_action="ignore")
    out["weekday"] = date.dt.dayofweek.map(lambda d: WEEKDAY_NAMES[int(d)], na_action="ignore")
    out["month"] = date.dt.month.map(lambda m: f"{int(m):02d}", na_action="ignore")
    out["speed_limit"] = collisions["speed_limit"].astype("Int64").map(
        lambda v: str(int(v)), na_action="ignore"
    )
    for feature, source in CATEGORICAL_FEATURES.items():
        if feature in out.columns:
            continue
        if feature == "region":
            out[feature] = region_from_police_force(collisions[source])
        else:
            out[feature] = collisions[source].astype("object")
    for feature in ABLATION_NUMERIC_FEATURES:
        out[feature] = collisions[feature].astype("float64")
    for feature in CATEGORICAL_FEATURES:
        out[feature] = out[feature].astype("object").where(out[feature].notna(), np.nan)
    out[TARGET] = collisions[TARGET].astype("Int64")
    dropped = int(out[TARGET].isna().sum())
    if dropped:
        logger.warning("Dropping %d collisions with missing severity", dropped)
    return out.loc[out[TARGET].notna()].astype({TARGET: int}).reset_index(drop=True)


def split_by_year(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split into train (2005-2015), valid (2016) and test (2017-2019).

    Raises:
        AssertionError: if any collision index appears in more than one split.
    """
    spans = {
        "train": config.CLF_TRAIN_YEARS,
        "valid": config.CLF_VALID_YEARS,
        "test": config.CLF_TEST_YEARS,
    }
    splits = {name: frame.loc[frame["year"].between(*span)].copy() for name, span in spans.items()}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, pd.DataFrame], key: str = "collision_index") -> None:
    """Assert that no value of ``key`` occurs in two splits (and none twice in one)."""
    seen: dict[str, str] = {}
    for name, part in splits.items():
        assert part[key].is_unique, f"duplicate {key} inside {name}"
        overlap = set(part[key]) & seen.keys()
        assert not overlap, f"{len(overlap)} {key} values in both {name} and {seen[next(iter(overlap))]}"
        seen.update(dict.fromkeys(part[key], name))


def make_preprocessor(
    categorical: list[str], numeric: list[str] | None = None, min_frequency: int = 50
) -> ColumnTransformer:
    """ColumnTransformer with every input column listed explicitly.

    Categoricals are one-hot encoded (missing values get their own indicator column,
    categories rarer than ``min_frequency`` are pooled, unseen categories at predict time
    go to the pooled column). Numerics are median-imputed and standardised.
    ``remainder="drop"`` is safe because :func:`check_feature_coverage` asserts that every
    intended column produced output features.
    """
    numeric = numeric or []
    transformers: list[tuple[str, object, list[str]]] = [
        (
            "cat",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_frequency,
                sparse_output=True,
            ),
            list(categorical),
        )
    ]
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                list(numeric),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def check_feature_coverage(
    preprocessor: ColumnTransformer, categorical: list[str], numeric: list[str] | None = None
) -> list[str]:
    """Assert every intended input column produced at least one output feature.

    Returns:
        The output feature names.
    """
    names = list(preprocessor.get_feature_names_out())
    for col in categorical:
        assert any(n.startswith(f"cat__{col}_") for n in names), f"{col} missing from output"
    for col in numeric or []:
        assert f"num__{col}" in names, f"{col} missing from output"
    return names
